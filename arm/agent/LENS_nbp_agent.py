
from typing import List

import torch
from yarr.agents.agent import Agent, ActResult, Summary

import numpy as np
from rlbench.backend.const import TABLE_COORD

from arm import utils
from arm.agent.LENS_nbv_agent import LensNBVAgent

NAME = 'LensNBPAgent'
GAMMA = 0.99
NSTEP = 1
REPLAY_ALPHA = 0.7
REPLAY_BETA = 0.5


class LensNBPAgent(Agent):

    def __init__(self,
                 qattention_agents: List[LensNBVAgent],
                 rotation_resolution: float,
                 viewpoint_agent_bounds: List[float],
                 viewpoint_env_bounds: List[float],
                 viewpoint_resolution: List[float],
                 camera_names: List[str],
                 rotation_prediction_depth: int = 0,
                 ):
        super(LensNBPAgent, self).__init__()
        self._qattention_agents = qattention_agents
        self._rotation_resolution = rotation_resolution
        self._camera_names = camera_names
        self._rotation_prediction_depth = rotation_prediction_depth
        # r,theta,phi
        self._viewpoint_agent_bounds = np.array(viewpoint_agent_bounds)
        self._viewpoint_resolution = np.array(viewpoint_resolution)
        self._viewpoint_env_bounds = np.array(viewpoint_env_bounds)

    def build(self, training: bool, device=None) -> None:
        self._device = device
        if self._device is None:
            self._device = torch.device('cpu')
        for qa in self._qattention_agents:
            qa.build(training, device)
            
        self._act_depth = 0
        #self._observation_elements = {}
        self._translation_results = []
        self._rot_grip_results = [] 
        self._infos = {}
        

    def update(self, step: int, replay_sample: dict) -> dict:
        priorities = 0
        for qa in self._qattention_agents:
            update_dict = qa.update(step, replay_sample)
            priorities += update_dict['priority']
            replay_sample.update(update_dict)
            
        return {
            'priority': (priorities) ** REPLAY_ALPHA,
        }
        
    def reset(self):
        self._act_depth = 0
        #self._observation_elements = {}
        self._translation_results = []
        self._infos = {}
        
    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:

        observation_elements = {}
        infos = {}
        #translation_results = []
        
        act_results = self._qattention_agents[self._act_depth].\
            act(step, observation, deterministic)
            

        attention_output = act_results.observation_elements['attention_output'].cpu().numpy()
        observation_elements['attention_output'] = attention_output[0]
        
        viewpoint_coordinate = act_results.observation_elements['viewpoint_coordinate'].cpu().numpy()
        observation_elements['viewpoint_coordinate'] = viewpoint_coordinate[0]
        

        translation_idxs, rot_grip_idxs = act_results.action
        observation_elements['translation_idxs'] = translation_idxs[0].cpu().numpy()



        observation_elements['prev_layer_voxel_grid'] \
            = act_results.observation_elements['prev_layer_voxel_grid'][0].cpu().numpy()
        

        infos.update(act_results.info)
        
            
        rgai = rot_grip_idxs[0].cpu().numpy() if rot_grip_idxs is not None else np.array([],dtype=np.int32)

        observation_elements['rot_grip_action_indicies'] = rgai
        

        if self._act_depth < len(self._qattention_agents)-1:

            reference_fixation = np.array([0,0,0])
            continuous_action = np.concatenate([observation_elements['viewpoint_coordinate'],
                                                reference_fixation],axis=-1)
        else:
            # LENS Phase 6: Generate continuous end-effector deltas for manipulation
            # Convert discrete rotation indices to continuous pose deltas
            if len(rgai) >= 4:  # Has rotation and gripper data
                # Convert discrete indices to continuous deltas with exploration noise
                rotation_indices = rgai[:-1].astype(np.float32)  # Remove gripper, keep rotations
                gripper_action = rgai[-1:].astype(np.float32)    # Gripper action
                
                # Add exploration noise for continuous control
                rotation_noise = np.random.normal(0.0, 0.3, size=rotation_indices.shape)
                rotation_indices = rotation_indices + rotation_noise
                
                # Scale to small rotation deltas: ±2.5° (half of OTA's 5° resolution)
                max_rotation_index = 360 // self._rotation_resolution
                normalized_rotations = (rotation_indices / max_rotation_index) * 2.0 - 1.0  # [-1, 1]
                rotation_deltas_deg = normalized_rotations * 2.5  # ±2.5° deltas
                
                # Convert rotation deltas to quaternion deltas (small rotations)
                rotation_deltas_rad = np.radians(rotation_deltas_deg)
                # For small angles, quaternion ≈ [sin(θ/2), sin(φ/2), sin(ψ/2), cos(θ/2)]
                # But for deltas, we use small angle approximation: [θ/2, φ/2, ψ/2, 1]
                quat_deltas = np.concatenate([rotation_deltas_rad / 2.0, [1.0]])  # Small rotation quaternion
                quat_deltas = quat_deltas / np.linalg.norm(quat_deltas)  # Normalize
                
                continuous_action = np.concatenate([
                    observation_elements['attention_output'],  # Position target (3D)
                    quat_deltas,                              # Rotation deltas (4D) 
                    gripper_action])                          # Gripper action (1D)
            else:
                # Fallback: use attention output with default orientation and gripper
                continuous_action = np.concatenate([
                    observation_elements['attention_output'],
                    np.array([0, 0, 0, 1]),  # No rotation delta (identity quaternion)
                    np.array([0])])          # Default gripper action


        if self._act_depth < len(self._qattention_agents)-1:
            self._act_depth += 1
        else:
            self._act_depth = 0
            
        return ActResult(

            continuous_action,

            observation_elements=observation_elements,
            info=self._infos
        )
        
        

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for qa in self._qattention_agents:
            summaries.extend(qa.update_summaries())
        return summaries

    def act_summaries(self) -> List[Summary]:
        s = []
        for qa in self._qattention_agents:
            s.extend(qa.act_summaries())
        return s

    def load_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.load_weights(savedir)

    def save_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.save_weights(savedir)
