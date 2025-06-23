"""
LENS Rollout Generator - Implements gated training with 5-step camera exploration.

This module extends the OTA rollout generator with LENS-specific gated training logic:
- Phase A: 5-step camera exploration with intrinsic rewards
- Phase B: Manipulation from best view with task rewards
- Best-of-N strategy using segmentation-based rewards
"""

import copy
import numpy as np
import torch
from multiprocessing import Value

from yarr.agents.agent import Agent
from yarr.envs.env import Env

from arm.ota.rollout_generator import OtaRolloutGenerator
from arm.custom.reward_wrappers import SegmentationReward
from arm import utils


class LensRolloutGenerator(OtaRolloutGenerator):
    """
    LENS rollout generator implementing gated training with camera exploration.
    
    Extends OtaRolloutGenerator with:
    1. 5-step camera exploration phase with intrinsic rewards
    2. Best-of-N viewpoint selection based on segmentation rewards
    3. Manipulation phase from optimal viewpoint
    """
    
    def __init__(self,
                 scene_bounds,
                 viewpoint_agent_bounds=None,
                 viewpoint_resolution=None,
                 viewpoint_env_bounds=None,
                 viewpoint_align=True,
                 reach_reward=0.02,
                 rollout=False,
                 device="cpu",
                 # LENS-specific parameters
                 target_object_id=211,
                 camera_exploration_steps=5,
                 r_vis_weight=0.02,
                 r_struct_weight=0.01,
                 p_smooth_weight=0.005):
        
        # Initialize parent OTA rollout generator
        super().__init__(
            scene_bounds=scene_bounds,
            viewpoint_agent_bounds=viewpoint_agent_bounds,
            viewpoint_resolution=viewpoint_resolution,
            viewpoint_env_bounds=viewpoint_env_bounds,
            viewpoint_align=viewpoint_align,
            reach_reward=reach_reward,
            rollout=rollout,
            device=device
        )
        
        # LENS-specific parameters
        self._target_object_id = target_object_id
        self._camera_exploration_steps = camera_exploration_steps
        
        # Initialize LENS segmentation reward wrapper
        self._lens_reward = SegmentationReward(
            scene_bound=scene_bounds,
            target_object_id=target_object_id,
            r_vis_weight=r_vis_weight,
            r_struct_weight=r_struct_weight,
            p_smooth_weight=p_smooth_weight,
            device=device
        )
        
    def act_and_execute(self, step_signal: Value, env: Env, agent: Agent,
                       timesteps: int, eval: bool, obs_dict_tp0: dict, 
                       final: bool = False, viewpoint_align: bool = True):
        """
        LENS gated training implementation with camera exploration.
        
        Phase A: 5-step camera exploration with intrinsic rewards
        Phase B: Manipulation from best view with task rewards
        """
        
        # LENS Phase A: Camera Exploration with Best-of-N Strategy
        best_intrinsic_reward = -float('inf')
        best_view_obs = None
        best_camera_action = None
        best_vp_world_cart_pose = None
        best_attention_output = None
        
        # Store initial state for restoration
        initial_obs_dict = copy.deepcopy(obs_dict_tp0)
        
        for exploration_step in range(self._camera_exploration_steps):
            # Prepare NBV observation input (same as OTA)
            nbv_input_obs_dict = {}
            for obs_key in ['active_point_cloud', 'low_dim_state', 'active_rgb']:
                nbv_input_obs_dict[obs_key] = []
            
            for step_index in range(timesteps):
                # Process observation (identical to OTA preprocessing)
                gp_world_pos_tp0 = obs_dict_tp0["gripper_pose"][step_index][:3]
                gp_world_quat_tp0 = self._normalize_quaternion(obs_dict_tp0["gripper_pose"][step_index][3:])
                vp_world_cart_pos_tp0 = obs_dict_tp0["active_cam_pose"][step_index][:3]
                time_state_tp0 = obs_dict_tp0["low_dim_state"][step_index][[-1]] if env._time_in_state else np.array([])
                gp_tip_state_tp0 = np.concatenate([obs_dict_tp0["low_dim_state"][step_index][[0]],
                                                 obs_dict_tp0["low_dim_state"][step_index][8:10]])
                pc_world_tp0 = obs_dict_tp0["active_point_cloud"][step_index]
                
                if gp_world_quat_tp0[-1] < 0:
                    gp_world_quat_tp0 = -gp_world_quat_tp0
                
                # Apply viewpoint alignment if needed (same as OTA)
                if viewpoint_align:
                    wtl_rot_tp0 = self._world_to_local_rotation(vp_world_cart_pos_tp0)
                    gp_local_pos_tp0, gp_local_quat_tp0, _ = self._world_pose_to_local_pose(
                        wtl_rot_tp0, gp_world_pos_tp0, gp_world_quat_tp0)
                    pc_local_tp0 = self._world_pc_to_local_pc(vp_world_cart_pos_tp0, pc_world_tp0)
                    
                    # Build local state
                    norm_vp_world_spher_tp0 = self._viewpoint_normalize(
                        self._world_cart_to_spher(vp_world_cart_pos_tp0))
                    low_dim_state_local_tp0 = np.concatenate([
                        gp_tip_state_tp0, gp_local_pos_tp0, gp_local_quat_tp0,
                        norm_vp_world_spher_tp0[self._viewpoint_axes_movable], time_state_tp0], axis=-1)
                    
                    nbv_input_obs_dict["active_point_cloud"].append(pc_local_tp0.astype(np.float32))
                    nbv_input_obs_dict["low_dim_state"].append(low_dim_state_local_tp0.astype(np.float32))
                else:
                    # World coordinates (fallback)
                    norm_vp_world_spher_tp0 = self._viewpoint_normalize(
                        self._world_cart_to_spher(vp_world_cart_pos_tp0))
                    low_dim_state_world_tp0 = np.concatenate([
                        gp_tip_state_tp0, gp_world_pos_tp0, gp_world_quat_tp0,
                        norm_vp_world_spher_tp0[self._viewpoint_axes_movable], time_state_tp0], axis=-1)
                    
                    nbv_input_obs_dict["active_point_cloud"].append(pc_world_tp0.astype(np.float32))
                    nbv_input_obs_dict["low_dim_state"].append(low_dim_state_world_tp0.astype(np.float32))
                
                nbv_input_obs_dict["active_rgb"].append(obs_dict_tp0["active_rgb"][step_index])
                nbv_input_obs_dict["attention_input"] = [np.zeros((3,), dtype=np.float32)]
            
            # Get camera action from NBV agent (continuous deltas)
            prepped_data = {k: torch.tensor(np.array(v)[None], device=self._device) 
                           for k, v in nbv_input_obs_dict.items()}
            nbv_output = agent.act(step=step_signal.value, observation=prepped_data, deterministic=eval)
            
            # Extract camera deltas and compute new viewpoint
            camera_deltas = nbv_output.action[:2] if len(nbv_output.action) >= 2 else [0, 0]  # [Δθ, Δφ]
            
            # Current viewpoint in spherical coordinates
            _, vp_world_spher_tp0 = self._world_cart_to_spher(vp_world_cart_pos_tp0)
            
            # Apply camera deltas (following rollout generator pattern)
            vp_world_spher_goal = np.copy(vp_world_spher_tp0)
            if viewpoint_align:
                vp_world_spher_goal[1] = camera_deltas[0]  # Absolute θ
                vp_world_spher_goal[2] += camera_deltas[1]  # Delta φ
            else:
                vp_world_spher_goal[1] += camera_deltas[0]  # Delta θ  
                vp_world_spher_goal[2] += camera_deltas[1]  # Delta φ
            
            # Clip to viewpoint bounds
            vp_world_spher_goal = np.clip(vp_world_spher_goal, 
                                        self._viewpoint_env_bounds[:3], 
                                        self._viewpoint_env_bounds[3:])
            
            # Convert to world cartesian coordinates
            vp_world_cart_pose_goal = self._spher_to_world_pose(vp_world_spher_goal)
            
            # Execute camera movement
            tp0_1_transition = env.step(vp_world_cart_pose_goal, 'vision', final, eval)
            new_obs = tp0_1_transition.observation
            
            # Compute intrinsic rewards using LENS segmentation rewards
            if 'active_mask' in new_obs and 'active_depth' in new_obs:
                intrinsic_rewards = self._lens_reward.compute_intrinsic_rewards(
                    obs_tp0={'active_depth': obs_dict_tp0['active_depth'][0], 
                            'active_mask': obs_dict_tp0.get('active_mask', [np.zeros_like(obs_dict_tp0['active_depth'][0])])[0]},
                    obs_tp1={'active_depth': new_obs['active_depth'], 
                            'active_mask': new_obs.get('active_mask', np.zeros_like(new_obs['active_depth']))},
                    target_object_id=self._target_object_id,
                    action=np.array(camera_deltas)
                )
                
                total_intrinsic_reward = intrinsic_rewards['total_intrinsic_reward']
            else:
                # Fallback: use exploration-based reward
                total_intrinsic_reward = -np.linalg.norm(camera_deltas) * 0.01
            
            # Best-of-N selection
            if total_intrinsic_reward > best_intrinsic_reward:
                best_intrinsic_reward = total_intrinsic_reward
                best_view_obs = copy.deepcopy(new_obs)
                best_camera_action = camera_deltas
                best_vp_world_cart_pose = vp_world_cart_pose_goal
                best_attention_output = nbv_output.observation_elements.get('attention_output', np.array([0, 0, 0]))
            
            # Update observation for next exploration step
            obs_dict_tp0 = {k: [np.array(v, dtype=self._get_type(v))] * timesteps 
                           for k, v in new_obs.items()}
        
        # LENS Phase B: Manipulation from Best View
        # Move camera to best viewpoint
        env.step(best_vp_world_cart_pose, 'vision', final, eval)
        
        # Prepare observation with best attention output
        best_obs_dict = {k: [np.array(v, dtype=self._get_type(v))] * timesteps 
                        for k, v in best_view_obs.items()}
        best_obs_dict['attention_input'] = [best_attention_output] * timesteps
        
        # Execute manipulation using OTA's NBP logic
        return self._execute_manipulation_phase(
            step_signal, env, agent, timesteps, eval, best_obs_dict, 
            best_camera_action, best_intrinsic_reward, final, viewpoint_align)
    
    def _execute_manipulation_phase(self, step_signal, env, agent, timesteps, eval, 
                                   obs_dict_tp0_1, camera_action, intrinsic_reward, 
                                   final, viewpoint_align):
        """Execute manipulation phase using OTA NBP logic."""
        
        # This method implements the standard OTA manipulation logic
        # but with LENS rewards integrated
        
        # Prepare NBP input (identical to OTA)
        nbp_input_obs_dict = {}
        for obs_key in ['active_point_cloud', 'low_dim_state', 'active_rgb', 'attention_input']:
            nbp_input_obs_dict[obs_key] = []
        
        attention_world_pos_goal = obs_dict_tp0_1['attention_input'][0]
        
        for step_index in range(timesteps):
            # Standard OTA NBP preprocessing
            gp_world_pos_tp0_1 = obs_dict_tp0_1["gripper_pose"][step_index][:3]
            gp_world_quat_tp0_1 = self._normalize_quaternion(obs_dict_tp0_1["gripper_pose"][step_index][3:])
            vp_world_cart_pos_tp0_1 = obs_dict_tp0_1["active_cam_pose"][step_index][:3]
            time_state_tp0_1 = obs_dict_tp0_1["low_dim_state"][step_index][[-1]] if env._time_in_state else np.array([])
            gp_tip_state_tp0_1 = obs_dict_tp0_1["low_dim_state"][step_index][:3]
            pc_world_tp0_1 = obs_dict_tp0_1["active_point_cloud"][step_index]
            attention_world_pos_tp0_1 = obs_dict_tp0_1['attention_input'][step_index]
            
            if gp_world_quat_tp0_1[-1] < 0:
                gp_world_quat_tp0_1 = -gp_world_quat_tp0_1
            
            # Apply viewpoint alignment (same as OTA)
            if viewpoint_align:
                wtl_rot_tp0_1 = self._world_to_local_rotation(vp_world_cart_pos_tp0_1)
                gp_local_pos_tp0_1, gp_local_quat_tp0_1, _ = self._world_pose_to_local_pose(
                    wtl_rot_tp0_1, gp_world_pos_tp0_1, gp_world_quat_tp0_1)
                pc_local_tp0_1 = self._world_pc_to_local_pc(vp_world_cart_pos_tp0_1, pc_world_tp0_1)
                attention_local_pos_tp0_1, _, _ = self._world_pose_to_local_pose(
                    wtl_rot_tp0_1, attention_world_pos_tp0_1, np.array([0,0,0,1]))
                
                norm_vp_world_spher_tp0_1 = self._viewpoint_normalize(
                    self._world_cart_to_spher(vp_world_cart_pos_tp0_1))
                low_dim_state_local_tp0_1 = np.concatenate([
                    gp_tip_state_tp0_1, gp_local_pos_tp0_1, gp_local_quat_tp0_1,
                    norm_vp_world_spher_tp0_1[self._viewpoint_axes_movable], time_state_tp0_1], axis=-1)
                
                nbp_input_obs_dict["active_point_cloud"].append(pc_local_tp0_1.astype(np.float32))
                nbp_input_obs_dict["low_dim_state"].append(low_dim_state_local_tp0_1.astype(np.float32))
                nbp_input_obs_dict["attention_input"].append(attention_local_pos_tp0_1.astype(np.float32))
            else:
                norm_vp_world_spher_tp0_1 = self._viewpoint_normalize(
                    self._world_cart_to_spher(vp_world_cart_pos_tp0_1))
                low_dim_state_world_tp0_1 = np.concatenate([
                    gp_tip_state_tp0_1, gp_world_pos_tp0_1, gp_world_quat_tp0_1,
                    norm_vp_world_spher_tp0_1[self._viewpoint_axes_movable], time_state_tp0_1], axis=-1)
                
                nbp_input_obs_dict["active_point_cloud"].append(pc_world_tp0_1.astype(np.float32))
                nbp_input_obs_dict["low_dim_state"].append(low_dim_state_world_tp0_1.astype(np.float32))
                nbp_input_obs_dict["attention_input"].append(attention_world_pos_tp0_1.astype(np.float32))
            
            nbp_input_obs_dict["active_rgb"].append(obs_dict_tp0_1["active_rgb"][step_index])
        
        # Get manipulation action from NBP agent
        prepped_data = {k: torch.tensor(np.array(v)[None], device=self._device) 
                       for k, v in nbp_input_obs_dict.items()}
        nbp_output = agent.act(step=step_signal.value, observation=prepped_data, deterministic=eval)
        
        # Apply manipulation action
        nbp_action = nbp_output.action
        if viewpoint_align:
            nbp_action = self._local_gripper_action_to_world_action(
                nbp_output.action, vp_world_cart_pos_tp0_1)
        
        tp1_transition = env.step(nbp_action, 'worker', final, eval)
        terminal = tp1_transition.terminal
        
        # Prepare observation elements for replay storage
        nbv_agent_obs_elems = {'attention_output': attention_world_pos_goal,
                              'viewpoint_coordinate': camera_action,
                              'translation_idxs': np.array([0, 0, 0]),  # Placeholder
                              'rot_grip_action_indicies': np.array([0, 0])}  # Placeholder
        
        nbp_agent_obs_elems = {k: np.array(v) for k, v in nbp_output.observation_elements.items()}
        
        # Build replay elements (following OTA pattern)
        obs_and_replay_elems = {}
        
        # Add LENS-specific elements
        obs_and_replay_elems.update({
            'active_point_cloud_layer_0': nbp_input_obs_dict['active_point_cloud'][0],
            'low_dim_state_layer_0': nbp_input_obs_dict['low_dim_state'][0],
            'active_rgb_layer_0': nbp_input_obs_dict['active_rgb'][0],
            'attention_output_layer_0': attention_world_pos_goal,
            'viewpoint_coordinate_layer_0': camera_action,
            'translation_idxs_layer_0': nbv_agent_obs_elems['translation_idxs'],
            'rot_grip_action_indicies_layer_0': nbv_agent_obs_elems['rot_grip_action_indicies']
        })
        
        # Add NBP elements
        for k, v in nbp_agent_obs_elems.items():
            if k in ['attention_output', 'viewpoint_coordinate', 'translation_idxs', 'rot_grip_action_indicies']:
                obs_and_replay_elems[f'{k}_layer_1'] = v
        
        # Compute combined rewards (AuxReward + LENS intrinsic)
        information_gain, roi_entropy_tp0, roi_entropy_tp0_1, occ_ratio_tp0_1 = self._aux_reward.update_grid(
            target_point=attention_world_pos_goal,
            extrinsics_tp0=obs_dict_tp0_1["active_camera_extrinsics"][0],
            extrinsics_tp0_1=obs_dict_tp0_1["active_camera_extrinsics"][0],  # Same viewpoint
            depth_tp0=obs_dict_tp0_1["active_depth"][0],
            depth_tp0_1=obs_dict_tp0_1["active_depth"][0],  # Same viewpoint
            pc_tp0=obs_dict_tp0_1["active_point_cloud"][0],
            pc_tp0_1=obs_dict_tp0_1["active_point_cloud"][0],  # Same viewpoint
            roi_size=0.25)
        
        # Standard OTA reward computation
        gripper_action_result = tp1_transition.observation["gripper_pose"][:3]
        gripper_pos_goal = attention_world_pos_goal
        gripper_attention_distance = np.linalg.norm(gripper_pos_goal - gripper_action_result)
        reach_reward = -self._reach_reward
        roi_reachable = gripper_attention_distance <= 0.22
        roi_non_empty = occ_ratio_tp0_1 > 0.0
        
        if roi_reachable:
            if roi_non_empty:
                reach_reward = self._reach_reward
            else:
                reach_reward = 0
        
        focus_reward = information_gain
        
        # Add LENS intrinsic reward to focus reward
        total_focus_reward = focus_reward + intrinsic_reward
        
        obs_and_replay_elems.update({
            "reach_reward": np.array([reach_reward]),
            "focus_reward": np.array([total_focus_reward])
        })
        
        # Update transition info
        tp1_transition.info.update({
            "roi_entropy": roi_entropy_tp0_1,
            "roi_reachable": float(roi_reachable),
            "roi_non_empty": float(roi_non_empty),
            "information_gain": information_gain / (roi_entropy_tp0 + 1e-10),
            "viewpoint_tp0_1": best_vp_world_cart_pose[:3] if 'best_vp_world_cart_pose' in locals() else [0, 0, 0],
            "lens_intrinsic_reward": intrinsic_reward
        })
        
        return obs_and_replay_elems, nbp_output.action, tp1_transition, terminal
    
    # Helper methods using OTA utilities
    def _normalize_quaternion(self, quat):
        """Normalize quaternion using OTA utils."""
        return utils.normalize_quaternion(quat)
    
    def _world_cart_to_spher(self, cart_pos):
        """Convert world cartesian to spherical using OTA utils."""
        _, spher_coord = utils.world_cart_to_disc_world_spher(
            world_cartesion_position=cart_pos,
            bounds=self._viewpoint_agent_bounds,
            spher_res=self._viewpoint_resolution)
        return spher_coord
    
    def _spher_to_world_pose(self, spher_coord):
        """Convert spherical to world pose using OTA utils."""
        _, _, _, world_pose = utils.local_spher_to_world_pose(
            local_viewpoint_spher_coord=spher_coord)
        return world_pose
    
    def _viewpoint_normalize(self, spher_coord):
        """Normalize viewpoint coordinates using OTA utils."""
        return utils.viewpoint_normlize(spher_coord, self._viewpoint_env_bounds)
    
    def _world_to_local_rotation(self, world_pos):
        """World to local rotation matrix using OTA utils."""
        return utils.world_to_local_rotation(world_viewpoint_position=world_pos)
    
    def _world_pose_to_local_pose(self, rotation_matrix, world_pos, world_quat):
        """Transform world pose to local pose using OTA utils."""
        return utils.world_pose_to_local_pose(
            world_to_local_rotation=rotation_matrix,
            world_point_position=world_pos,
            world_point_quat=world_quat)
    
    def _world_pc_to_local_pc(self, viewpoint_pos, world_pc):
        """Transform world point cloud to local using OTA utils."""
        return utils.world_pc_to_local_pc(
            world_viewpoint_position=viewpoint_pos,
            world_points=world_pc)
    
    def _local_gripper_action_to_world_action(self, local_action, viewpoint_pos):
        """Transform local gripper action to world using OTA utils."""
        return utils.local_gripper_action_to_world_action(
            local_gripper_action=local_action,
            world_viewpoint_position=viewpoint_pos)