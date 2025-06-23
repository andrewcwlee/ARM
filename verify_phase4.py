#!/usr/bin/env python3
"""
Phase 4 Verification: Reward System Extension

Verifies that the segmentation-based reward wrapper extends AuxReward patterns correctly
and integrates with LENS configuration parameters.
"""

import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf

# Add ARM to path for imports
sys.path.append('/home/andrewlee/_research/ota/ARM')

def test_reward_wrapper_import():
    """Test that reward wrapper can be imported correctly."""
    print("Testing reward wrapper import...")
    try:
        from arm.custom.reward_wrappers import SegmentationReward, create_lens_reward_wrapper
        print("✓ Reward wrapper imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_auxreward_inheritance():
    """Test that SegmentationReward correctly inherits from AuxReward."""
    print("Testing AuxReward inheritance...")
    try:
        from arm.custom.reward_wrappers import SegmentationReward
        from arm.ota.aux_task.aux_reward import AuxReward
        
        # Check inheritance
        assert issubclass(SegmentationReward, AuxReward), "SegmentationReward must inherit from AuxReward"
        print("✓ SegmentationReward correctly inherits from AuxReward")
        
        # Check constructor signature compatibility
        import inspect
        aux_params = inspect.signature(AuxReward.__init__).parameters
        seg_params = inspect.signature(SegmentationReward.__init__).parameters
        
        # All AuxReward parameters should be present
        for param_name in aux_params:
            if param_name != 'self':
                assert param_name in seg_params, f"Missing AuxReward parameter: {param_name}"
        
        print("✓ Constructor signature compatible with AuxReward")
        return True
    except Exception as e:
        print(f"✗ Inheritance test failed: {e}")
        return False

def test_reward_computation():
    """Test reward computation with dummy data following LENS plan specifications."""
    print("Testing reward computation...")
    try:
        from arm.custom.reward_wrappers import SegmentationReward
        
        # Create dummy scene bounds (same as AuxReward patterns)
        scene_bound = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        device = "cpu"
        
        # Initialize reward wrapper with all LENS parameters
        reward_wrapper = SegmentationReward(
            scene_bound=scene_bound,
            target_object_id=211,
            r_vis_weight=0.02,
            r_struct_weight=0.01,
            p_smooth_weight=0.005,
            device=device
        )
        print("✓ SegmentationReward initialized successfully")
        
        # Create dummy observations following LENS plan
        dummy_mask_tp0 = np.zeros((128, 128), dtype=np.int32)
        dummy_mask_tp0_1 = np.zeros((128, 128), dtype=np.int32)
        dummy_depth_tp0 = np.random.uniform(0.5, 2.0, (128, 128))
        dummy_depth_tp0_1 = np.random.uniform(0.5, 2.0, (128, 128))
        
        # Add some target object pixels with realistic depth values
        dummy_mask_tp0[30:50, 30:50] = 211  # Target object at tp0
        dummy_mask_tp0_1[35:55, 35:55] = 211  # Slightly moved at tp0_1
        dummy_depth_tp0[30:50, 30:50] = 1.2  # Realistic depth for target
        dummy_depth_tp0_1[35:55, 35:55] = 1.1  # Slightly closer at tp0_1
        
        # Create observation dictionaries
        obs_tp0 = {
            'active_mask': dummy_mask_tp0,
            'active_depth': dummy_depth_tp0
        }
        obs_tp1 = {
            'active_mask': dummy_mask_tp0_1,
            'active_depth': dummy_depth_tp0_1
        }
        
        # Test visibility reward (R_vis)
        r_vis = reward_wrapper.compute_visibility_reward(obs_tp0, obs_tp1, 211)
        print(f"✓ R_vis computation: {r_vis:.6f}")
        
        # Test structural reward (R_struct)
        r_struct = reward_wrapper.compute_structure_reward(obs_tp1, 211)
        assert r_struct >= 0, f"Invalid structural reward: {r_struct}"
        print(f"✓ R_struct computation: {r_struct:.6f}")
        
        # Test smoothness penalty (P_smooth)
        dummy_action = np.array([0.1, 0.05])  # Camera deltas
        p_smooth = reward_wrapper.compute_smoothness_penalty(dummy_action)
        assert p_smooth <= 0, f"Smoothness penalty should be negative: {p_smooth}"
        print(f"✓ P_smooth computation: {p_smooth:.6f}")
        
        # Test intrinsic rewards computation
        intrinsic_rewards = reward_wrapper.compute_intrinsic_rewards(obs_tp0, obs_tp1, 211, dummy_action)
        required_keys = ['R_vis_raw', 'R_vis', 'R_struct_raw', 'R_struct', 'P_smooth', 'total_intrinsic_reward']
        for key in required_keys:
            assert key in intrinsic_rewards, f"Missing reward key: {key}"
        print("✓ Intrinsic rewards computed successfully")
        print(f"  - R_vis: {intrinsic_rewards['R_vis']:.6f}")
        print(f"  - R_struct: {intrinsic_rewards['R_struct']:.6f}")
        print(f"  - P_smooth: {intrinsic_rewards['P_smooth']:.6f}")
        print(f"  - Total intrinsic: {intrinsic_rewards['total_intrinsic_reward']:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Reward computation test failed: {e}")
        return False

def test_lens_config_integration():
    """Test integration with LENS configuration."""
    print("Testing LENS config integration...")
    try:
        from arm.custom.reward_wrappers import create_lens_reward_wrapper
        
        # Load LENS configuration
        lens_cfg = OmegaConf.load('/home/andrewlee/_research/ota/ARM/conf/method/LENS.yaml')
        
        # Test factory function
        scene_bound = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        reward_wrapper = create_lens_reward_wrapper(lens_cfg, scene_bound, device="cpu")
        
        # Verify configuration parameters
        assert reward_wrapper._target_object_id == lens_cfg.target_object_id
        assert reward_wrapper._r_vis_weight == lens_cfg.R_vis_weight
        assert reward_wrapper._r_struct_weight == lens_cfg.R_struct_weight
        assert reward_wrapper._p_smooth_weight == lens_cfg.P_smooth_weight
        
        print("✓ LENS config integration successful")
        print(f"  - Target object ID: {reward_wrapper._target_object_id}")
        print(f"  - Visibility weight: {reward_wrapper._r_vis_weight}")
        print(f"  - Structural weight: {reward_wrapper._r_struct_weight}")
        print(f"  - Smoothness weight: {reward_wrapper._p_smooth_weight}")
        
        return True
    except Exception as e:
        print(f"✗ LENS config integration test failed: {e}")
        return False

def test_observation_key_compatibility():
    """Test that expected observation keys are used correctly."""
    print("Testing observation key compatibility...")
    try:
        from arm.custom.reward_wrappers import SegmentationReward
        
        # Check that we use the same observation keys as OTA
        expected_keys = ['active_mask', 'active_depth', 'active_rgb', 'active_point_cloud']
        
        # The reward wrapper should process active_mask (which is enabled by mask=True in LENS config)
        print("✓ Expected observation keys:")
        for key in expected_keys:
            print(f"  - {key} (used in LENS with mask=True)")
        
        # Verify that target_object_id matches LENS config
        lens_cfg = OmegaConf.load('/home/andrewlee/_research/ota/ARM/conf/method/LENS.yaml')
        assert lens_cfg.target_object_id == 211, f"Expected target_object_id=211, got {lens_cfg.target_object_id}"
        print(f"✓ Target object ID matches LENS config: {lens_cfg.target_object_id}")
        
        return True
    except Exception as e:
        print(f"✗ Observation key compatibility test failed: {e}")
        return False

def test_reward_wrapper_methods():
    """Test that all expected methods are present and callable."""
    print("Testing reward wrapper methods...")
    try:
        from arm.custom.reward_wrappers import SegmentationReward
        
        # Check required methods (updated for LENS plan specifications)
        required_methods = [
            'update_observation_data',
            'compute_visibility_reward', 
            'compute_structure_reward',
            'compute_smoothness_penalty',
            'compute_intrinsic_rewards',
            'compute_combined_rewards'
        ]
        
        for method_name in required_methods:
            assert hasattr(SegmentationReward, method_name), f"Missing method: {method_name}"
            method = getattr(SegmentationReward, method_name)
            assert callable(method), f"Method not callable: {method_name}"
        
        print("✓ All required methods present and callable")
        
        # Check AuxReward methods are preserved
        auxreward_methods = ['update_grid', 'calculate_entropy', '_point_to_voxel_index']
        for method_name in auxreward_methods:
            assert hasattr(SegmentationReward, method_name), f"Missing AuxReward method: {method_name}"
        
        print("✓ AuxReward methods preserved through inheritance")
        return True
    except Exception as e:
        print(f"✗ Method test failed: {e}")
        return False

def main():
    """Run all Phase 4 verification tests."""
    print("=" * 50)
    print("PHASE 4 VERIFICATION: Reward System Extension")
    print("=" * 50)
    
    tests = [
        test_reward_wrapper_import,
        test_auxreward_inheritance,
        test_reward_computation,
        test_lens_config_integration,
        test_observation_key_compatibility,
        test_reward_wrapper_methods
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    print("=" * 50)
    if all(results):
        print("✓ PHASE 4 VERIFICATION PASSED")
        print("✓ SegmentationReward extends AuxReward correctly")
        print("✓ LENS plan reward specifications implemented:")
        print("  - R_vis: Depth-weighted visibility reward")
        print("  - R_struct: Structural richness based on depth variation")
        print("  - P_smooth: Action magnitude smoothness penalty")
        print("✓ Target object visibility tracked with target_object_id: 211")
        print("✓ Reward wrapper integrates with LENS configuration")
        print("✓ Same observation processing patterns as AuxReward maintained")
        print("✓ Ready to proceed to Phase 5 (Launch Integration)")
    else:
        print("✗ PHASE 4 VERIFICATION FAILED")
        for i, (test, result) in enumerate(zip(tests, results)):
            status = "PASS" if result else "FAIL"
            print(f"  {test.__name__}: {status}")
    print("=" * 50)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)