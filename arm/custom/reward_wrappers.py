"""
LENS Reward Wrappers - Segmentation-based reward computation extending AuxReward patterns.

This module extends the AuxReward patterns with segmentation-based rewards using active_mask
and active_depth observations for target object visibility tracking.

Implements the reward specifications from LENS_Implementation_Plan.md:
- R_vis: Depth-weighted visibility reward using segmentation masks
- R_struct: Structural richness based on depth variation within target object
- P_smooth: Smoothness penalty based on action magnitude
"""

import numpy as np
from typing import Dict, Any
from arm.ota.aux_task.aux_reward import AuxReward


class SegmentationReward(AuxReward):
    """
    Segmentation-based reward wrapper that extends AuxReward with target object visibility tracking.
    
    Uses active_mask observations to compute visibility and structural rewards for target objects.
    Follows the exact same patterns as AuxReward for compatibility.
    """
    
    def __init__(self,
                 scene_bound,
                 target_object_id: int = 211,
                 r_vis_weight: float = 0.02,
                 r_struct_weight: float = 0.01,
                 p_smooth_weight: float = 0.005,
                 world_pc_tp0=None,
                 world_pc_tp0_1=None,
                 voxel_size=100,
                 batch_size=1,
                 max_num_coords=1000000,
                 feature_size=0,
                 device="cpu") -> None:
        """
        Initialize segmentation reward wrapper.
        
        Args:
            scene_bound: Scene boundary coordinates (same as AuxReward)
            target_object_id: Target object ID for segmentation (default: 211)
            r_vis_weight: Weight for visibility reward component
            r_struct_weight: Weight for structural reward component
            p_smooth_weight: Weight for smoothness penalty component
            Other args: Same as AuxReward for compatibility
        """
        # Initialize parent AuxReward with exact same parameters
        super().__init__(
            scene_bound=scene_bound,
            world_pc_tp0=world_pc_tp0,
            world_pc_tp0_1=world_pc_tp0_1,
            voxel_size=voxel_size,
            batch_size=batch_size,
            max_num_coords=max_num_coords,
            feature_size=feature_size,
            device=device
        )
        
        # LENS-specific parameters
        self._target_object_id = target_object_id
        self._r_vis_weight = r_vis_weight
        self._r_struct_weight = r_struct_weight
        self._p_smooth_weight = p_smooth_weight
        
        # Cache for observations (following LENS plan specifications)
        self._active_mask_tp0 = None
        self._active_mask_tp0_1 = None
        self._active_depth_tp0 = None
        self._active_depth_tp0_1 = None
    
    def update_observation_data(self,
                               active_mask_tp0: np.ndarray,
                               active_mask_tp0_1: np.ndarray,
                               active_depth_tp0: np.ndarray,
                               active_depth_tp0_1: np.ndarray) -> None:
        """
        Update observation data for two time points (following LENS plan specifications).
        
        Args:
            active_mask_tp0: Segmentation mask at time point 0 [H, W]
            active_mask_tp0_1: Segmentation mask at time point 1 [H, W]
            active_depth_tp0: Depth image at time point 0 [H, W]
            active_depth_tp0_1: Depth image at time point 1 [H, W]
        """
        # Convert to numpy arrays (following LENS plan pattern)
        self._active_mask_tp0 = np.array(active_mask_tp0)
        self._active_mask_tp0_1 = np.array(active_mask_tp0_1)
        self._active_depth_tp0 = np.array(active_depth_tp0)
        self._active_depth_tp0_1 = np.array(active_depth_tp0_1)
    
    def compute_visibility_reward(self, obs_tp0: Dict[str, Any], obs_tp1: Dict[str, Any], target_object_id: int) -> float:
        """
        Compute visibility reward using depth-weighted approach (LENS plan specification).
        
        Args:
            obs_tp0: Observation at time point 0 with 'active_depth' and 'active_mask' keys
            obs_tp1: Observation at time point 1 with 'active_depth' and 'active_mask' keys
            target_object_id: Target object ID for segmentation
            
        Returns:
            Visibility reward (area_tp1 - area_tp0)
        """
        # Extract observations using verified observation keys
        depth_tp0 = obs_tp0['active_depth']
        depth_tp1 = obs_tp1['active_depth']
        seg_tp0 = obs_tp0['active_mask']
        seg_tp1 = obs_tp1['active_mask']
        
        # Create target object masks
        mask_tp0 = (seg_tp0 == target_object_id)
        mask_tp1 = (seg_tp1 == target_object_id)
        
        # Extract depths for target object pixels
        obj_depths_tp0 = depth_tp0[mask_tp0]
        obj_depths_tp1 = depth_tp1[mask_tp1]
        
        # Compute depth-weighted areas (LENS plan specification)
        area_tp0 = len(obj_depths_tp0) * (np.mean(obj_depths_tp0) ** 2) if len(obj_depths_tp0) > 0 else 0
        area_tp1 = len(obj_depths_tp1) * (np.mean(obj_depths_tp1) ** 2) if len(obj_depths_tp1) > 0 else 0
        
        return area_tp1 - area_tp0
    
    def compute_structure_reward(self, observation: Dict[str, Any], target_object_id: int) -> float:
        """
        Compute structural richness reward (LENS plan specification).
        
        Args:
            observation: Observation with 'active_depth' and 'active_mask' keys
            target_object_id: Target object ID for segmentation
            
        Returns:
            Structural richness score based on depth variation within target object
        """
        depth = observation['active_depth']
        segmentation = observation['active_mask']
        
        # Create target object mask
        mask = (segmentation == target_object_id)
        obj_depths = depth[mask]
        
        # Require minimum number of pixels for meaningful calculation
        if len(obj_depths) < 10:
            return 0.0
        
        # Structural richness as depth variation (LENS plan specification)
        return np.std(obj_depths) / (np.mean(obj_depths) + 1e-8)
    
    def compute_smoothness_penalty(self, action: np.ndarray) -> float:
        """
        Compute smoothness penalty based on action magnitude (LENS plan specification).
        
        Args:
            action: Action vector (camera deltas or end-effector deltas)
            
        Returns:
            Smoothness penalty (negative value)
        """
        return -np.linalg.norm(action) * self._p_smooth_weight
    
    def compute_intrinsic_rewards(self, obs_tp0: Dict[str, Any], obs_tp1: Dict[str, Any], 
                                 target_object_id: int, action: np.ndarray = None) -> Dict[str, float]:
        """
        Compute intrinsic rewards following LENS plan specifications.
        
        Args:
            obs_tp0: Observation at time point 0
            obs_tp1: Observation at time point 1  
            target_object_id: Target object ID for segmentation
            action: Action taken (for smoothness penalty)
            
        Returns:
            Dictionary with reward components following LENS plan
        """
        # R_vis: Depth-weighted visibility reward (LENS plan specification)
        r_vis_raw = self.compute_visibility_reward(obs_tp0, obs_tp1, target_object_id)
        r_vis = self._r_vis_weight * r_vis_raw
        
        # R_struct: Structural richness reward (LENS plan specification)
        r_struct_raw = self.compute_structure_reward(obs_tp1, target_object_id)
        r_struct = self._r_struct_weight * r_struct_raw
        
        # P_smooth: Smoothness penalty (LENS plan specification)
        p_smooth = 0.0
        if action is not None:
            p_smooth = self.compute_smoothness_penalty(action)
        
        # Total intrinsic reward
        total_intrinsic_reward = r_vis + r_struct + p_smooth
        
        return {
            'R_vis_raw': r_vis_raw,
            'R_vis': r_vis,
            'R_struct_raw': r_struct_raw,
            'R_struct': r_struct,
            'P_smooth': p_smooth,
            'total_intrinsic_reward': total_intrinsic_reward
        }
    
    def compute_combined_rewards(self,
                                obs_tp0: Dict[str, Any],
                                obs_tp1: Dict[str, Any],
                                target_point,
                                extrinsics_tp0,
                                extrinsics_tp0_1,
                                depth_tp0,
                                depth_tp0_1,
                                pc_tp0,
                                pc_tp0_1,
                                action: np.ndarray = None,
                                roi_size: float = 0.15) -> Dict[str, float]:
        """
        Compute combined AuxReward and LENS intrinsic rewards.
        
        Args:
            obs_tp0: Observation dict at time point 0 (with 'active_depth', 'active_mask')
            obs_tp1: Observation dict at time point 1 (with 'active_depth', 'active_mask')
            target_point: Target point for ROI computation (same as AuxReward)
            extrinsics_tp0/tp0_1: Camera extrinsics at both time points
            depth_tp0/tp0_1: Depth images at both time points
            pc_tp0/pc_tp0_1: Point clouds at both time points
            action: Action taken (for smoothness penalty)
            roi_size: ROI size for AuxReward computation
            
        Returns:
            Dictionary with all reward components
        """
        # Compute AuxReward information gain (preserving exact AuxReward patterns)
        information_gain, roi_entropy_tp0, roi_entropy_tp0_1, occ_ratio_tp0_1 = self.update_grid(
            target_point=target_point,
            extrinsics_tp0=extrinsics_tp0,
            extrinsics_tp0_1=extrinsics_tp0_1,
            depth_tp0=depth_tp0,
            depth_tp0_1=depth_tp0_1,
            pc_tp0=pc_tp0,
            pc_tp0_1=pc_tp0_1,
            roi_size=roi_size
        )
        
        # Compute LENS intrinsic rewards (following LENS plan specifications)
        intrinsic_rewards = self.compute_intrinsic_rewards(
            obs_tp0, obs_tp1, self._target_object_id, action
        )
        
        # Combine all rewards
        combined_rewards = {
            # AuxReward components (preserved exactly)
            'information_gain': information_gain,
            'roi_entropy_tp0': roi_entropy_tp0,
            'roi_entropy_tp0_1': roi_entropy_tp0_1,
            'occupancy_ratio_tp0_1': occ_ratio_tp0_1,
            
            # LENS intrinsic components
            **intrinsic_rewards,
            
            # Total combined reward
            'total_combined_reward': information_gain + intrinsic_rewards['total_intrinsic_reward']
        }
        
        return combined_rewards


def create_lens_reward_wrapper(cfg, scene_bound, device="cpu") -> SegmentationReward:
    """
    Factory function to create LENS reward wrapper from config.
    
    Args:
        cfg: LENS configuration object
        scene_bound: Scene boundary coordinates
        device: Compute device
        
    Returns:
        Configured SegmentationReward instance
    """
    return SegmentationReward(
        scene_bound=scene_bound,
        target_object_id=cfg.target_object_id,
        r_vis_weight=cfg.R_vis_weight,
        r_struct_weight=cfg.R_struct_weight,
        p_smooth_weight=cfg.P_smooth_weight,
        device=device
    )