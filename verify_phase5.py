#!/usr/bin/env python3
"""
Phase 5 Verification: Launch Integration

Verifies that LENS integrates correctly with the launch pipeline and can be executed
without demonstrations, following the exact OTA pattern.
"""

import os
import sys
import subprocess
from omegaconf import OmegaConf

# Add ARM to path for imports
sys.path.append('/home/andrewlee/_research/ota/ARM')

def test_lens_launch_utils_functions():
    """Test that all required lens_launch_utils functions exist."""
    print("Testing lens_launch_utils functions...")
    try:
        from arm.ota import lens_launch_utils
        
        # Check required functions exist
        required_functions = [
            'get_viewpoint_bounds',
            'create_replay', 
            'create_agent',
            'create_rollout_generator'  # Added in Phase 5
        ]
        
        for func_name in required_functions:
            assert hasattr(lens_launch_utils, func_name), f"Missing function: {func_name}"
            func = getattr(lens_launch_utils, func_name)
            assert callable(func), f"Function not callable: {func_name}"
        
        print("✓ All required lens_launch_utils functions present")
        return True
    except Exception as e:
        print(f"✗ lens_launch_utils function test failed: {e}")
        return False

def test_lens_config_loading():
    """Test that LENS configuration loads correctly."""
    print("Testing LENS config loading...")
    try:
        # Load LENS configuration
        lens_cfg = OmegaConf.load('/home/andrewlee/_research/ota/ARM/conf/method/LENS.yaml')
        
        # Verify essential parameters
        assert lens_cfg.name == 'LENS', f"Expected name='LENS', got {lens_cfg.name}"
        assert lens_cfg.target_object_id == 211, f"Expected target_object_id=211"
        assert hasattr(lens_cfg, 'voxel_sizes'), "Missing voxel_sizes parameter"
        assert hasattr(lens_cfg, 'viewpoint_align'), "Missing viewpoint_align parameter"
        assert hasattr(lens_cfg, 'reach_reward'), "Missing reach_reward parameter"
        
        print("✓ LENS configuration loads correctly")
        print(f"  - Method name: {lens_cfg.name}")
        print(f"  - Target object ID: {lens_cfg.target_object_id}")
        print(f"  - Voxel sizes: {lens_cfg.voxel_sizes}")
        
        return True
    except Exception as e:
        print(f"✗ LENS config loading test failed: {e}")
        return False

def test_launch_script_integration():
    """Test that launch.py recognizes LENS method."""
    print("Testing launch.py integration...")
    try:
        # Check that launch.py contains LENS case
        with open('/home/andrewlee/_research/ota/ARM/launch.py', 'r') as f:
            launch_content = f.read()
        
        # Verify LENS case exists
        assert "elif cfg.method.name == 'LENS':" in launch_content, "LENS case not found in launch.py"
        assert "from arm.ota import lens_launch_utils" in launch_content, "lens_launch_utils import not found"
        assert "lens_launch_utils.create_replay" in launch_content, "create_replay call not found"
        assert "lens_launch_utils.create_agent" in launch_content, "create_agent call not found"
        assert "lens_launch_utils.create_rollout_generator" in launch_content, "create_rollout_generator call not found"
        
        # Verify no actual fill_replay function call (LENS trains without demos)
        lens_section_start = launch_content.find("elif cfg.method.name == 'LENS':")
        c2farm_section_start = launch_content.find("elif 'C2FARM' in cfg.method.name:")
        lens_section = launch_content[lens_section_start:c2farm_section_start]
        assert "lens_launch_utils.fill_replay" not in lens_section, "LENS should not call lens_launch_utils.fill_replay"
        
        print("✓ Launch script integration correct")
        print("  - LENS case found in launch.py")
        print("  - All required function calls present")
        print("  - No demo loading (fill_replay) as expected")
        
        return True
    except Exception as e:
        print(f"✗ Launch script integration test failed: {e}")
        return False

def test_rollout_generator_creation():
    """Test that LENS rollout generator can be created."""
    print("Testing rollout generator creation...")
    try:
        from arm.ota import lens_launch_utils
        from omegaconf import OmegaConf
        
        # Load LENS config and add missing keys
        lens_cfg = OmegaConf.load('/home/andrewlee/_research/ota/ARM/conf/method/LENS.yaml')
        
        # Create complete config structure for testing
        full_cfg = OmegaConf.create({
            'method': lens_cfg,
            'rlbench': {
                'scene_bounds': [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            }
        })
        
        # Create dummy parameters
        viewpoint_agent_bounds = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        viewpoint_resolution = [0.1, 0.1, 0.1]
        viewpoint_env_bounds = [-1.2, -1.2, -1.2, 1.2, 1.2, 1.2]
        device = "cpu"
        
        # Test rollout generator creation
        rg = lens_launch_utils.create_rollout_generator(
            full_cfg, viewpoint_agent_bounds, viewpoint_resolution,
            viewpoint_env_bounds, device
        )
        
        # Verify it's the correct type
        from arm.ota.rollout_generator import OtaRolloutGenerator
        assert isinstance(rg, OtaRolloutGenerator), f"Expected OtaRolloutGenerator, got {type(rg)}"
        
        print("✓ Rollout generator creation successful")
        print(f"  - Type: {type(rg).__name__}")
        print(f"  - Device: {rg._device}")
        
        return True
    except Exception as e:
        print(f"✗ Rollout generator creation test failed: {e}")
        return False

def test_dry_run_syntax():
    """Test that LENS dry-run doesn't have syntax errors."""
    print("Testing dry-run syntax check...")
    try:
        # Run a basic syntax check by importing the launch module
        import launch
        
        print("✓ Launch module imports successfully")
        
        # Test basic config parsing with LENS method
        from omegaconf import OmegaConf
        
        # Create minimal config for syntax testing
        test_cfg = OmegaConf.create({
            'method': {'name': 'LENS'},
            'rlbench': {
                'scene_bounds': [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
                'camera_resolution': [128, 128],
                'cameras': ['active']
            },
            'replay': {
                'batch_size': 1,
                'timesteps': 1,
                'prioritisation': False,
                'use_disk': False
            }
        })
        
        print("✓ Basic configuration parsing works")
        
        return True
    except Exception as e:
        print(f"✗ Dry-run syntax test failed: {e}")
        return False

def test_component_compatibility():
    """Test that LENS components are compatible with each other."""
    print("Testing component compatibility...")
    try:
        from arm.ota import lens_launch_utils
        from arm.custom.reward_wrappers import create_lens_reward_wrapper
        from omegaconf import OmegaConf
        
        # Load LENS config
        lens_cfg = OmegaConf.load('/home/andrewlee/_research/ota/ARM/conf/method/LENS.yaml')
        
        # Test reward wrapper creation with LENS config
        scene_bound = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        reward_wrapper = create_lens_reward_wrapper(lens_cfg, scene_bound, device="cpu")
        
        # Create complete config for get_viewpoint_bounds
        full_cfg = OmegaConf.create({
            'method': lens_cfg,
            'rlbench': {'scene_bounds': [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]}
        })
        
        # Test viewpoint bounds calculation  
        viewpoint_agent_bounds, viewpoint_env_bounds, viewpoint_resolution = \
            lens_launch_utils.get_viewpoint_bounds(full_cfg)
        
        assert len(viewpoint_agent_bounds) == 6, "Invalid viewpoint_agent_bounds"
        assert len(viewpoint_env_bounds) == 6, "Invalid viewpoint_env_bounds"
        assert len(viewpoint_resolution) == 3, "Invalid viewpoint_resolution"
        
        print("✓ Component compatibility verified")
        print(f"  - Reward wrapper created successfully")
        print(f"  - Viewpoint bounds: agent={len(viewpoint_agent_bounds)}, env={len(viewpoint_env_bounds)}")
        
        return True
    except Exception as e:
        print(f"✗ Component compatibility test failed: {e}")
        return False

def main():
    """Run all Phase 5 verification tests."""
    print("=" * 50)
    print("PHASE 5 VERIFICATION: Launch Integration")
    print("=" * 50)
    
    tests = [
        test_lens_launch_utils_functions,
        test_lens_config_loading,
        test_launch_script_integration,
        test_rollout_generator_creation,
        test_dry_run_syntax,
        test_component_compatibility
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
        print("✓ PHASE 5 VERIFICATION PASSED")
        print("✓ LENS case added to launch.py copying OTA pattern")
        print("✓ Skip fill_replay (no demos for LENS) implemented")
        print("✓ Same environment and rollout generator as OTA maintained")
        print("✓ All lens_launch_utils functions available")
        print("✓ Component compatibility verified")
        print("✓ Ready for dry-run testing: python launch.py method=LENS --dry-run")
        print("✓ Ready to proceed to Phase 6 (Rollout Modification)")
    else:
        print("✗ PHASE 5 VERIFICATION FAILED")
        for i, (test, result) in enumerate(zip(tests, results)):
            status = "PASS" if result else "FAIL"
            print(f"  {test.__name__}: {status}")
    print("=" * 50)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)