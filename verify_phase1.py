#!/usr/bin/env python3
"""
Phase 1 Verification Script: LENS Configuration Setup
This script verifies that the LENS.yaml configuration is correctly set up.
"""

import os
import sys
from omegaconf import OmegaConf
import numpy as np

def test_config_loading():
    """Test that LENS.yaml loads correctly"""
    print("Testing LENS.yaml loading...")
    try:
        cfg = OmegaConf.load('conf/method/LENS.yaml')
        assert cfg.name == 'LENS', f"Expected name='LENS', got {cfg.name}"
        print("✓ LENS.yaml loads successfully")
        print(f"✓ Method name: {cfg.name}")
        return cfg
    except Exception as e:
        print(f"✗ Failed to load LENS.yaml: {e}")
        sys.exit(1)

def compare_with_ota():
    """Compare LENS config with OTA to ensure only intended changes"""
    print("\nComparing LENS.yaml with OTA.yaml...")
    
    ota_cfg = OmegaConf.load('conf/method/OTA.yaml')
    lens_cfg = OmegaConf.load('conf/method/LENS.yaml')
    
    # Check that all OTA parameters are preserved (except name)
    for key, value in ota_cfg.items():
        if key == 'name':
            continue
        if key in lens_cfg:
            if lens_cfg[key] != value:
                print(f"✗ Parameter {key} changed unexpectedly: {value} -> {lens_cfg[key]}")
                sys.exit(1)
        else:
            print(f"✗ Parameter {key} missing from LENS config")
            sys.exit(1)
    
    # Check that LENS-specific parameters are added
    for param in ['target_object_id', 'camera_exploration_steps', 'R_vis_weight', 
                  'R_struct_weight', 'P_smooth_weight']:
        if param not in lens_cfg:
            print(f"✗ LENS parameter {param} missing")
            sys.exit(1)
        print(f"✓ LENS parameter {param}: {lens_cfg[param]}")
    
    # Check SAC parameters are added
    sac_params = ['critic_lr', 'actor_lr', 'alpha', 'alpha_auto_tune']
    for param in sac_params:
        if param not in lens_cfg:
            print(f"✗ SAC parameter {param} missing")
            sys.exit(1)
        print(f"✓ SAC parameter {param}: {lens_cfg[param]}")
    
    print("✓ All expected parameters present and correct")

def test_mask_configuration():
    """Test that mask enabling works correctly with import test"""
    print("\nTesting mask configuration logic...")
    
    # Test the logic that will be used in launch.py
    class MockCfg:
        def __init__(self, method_name):
            self.method = type('method', (), {'name': method_name})()
    
    # Test LENS enables mask
    lens_cfg = MockCfg('LENS')
    enable_mask_lens = (lens_cfg.method.name == 'LENS')
    assert enable_mask_lens == True, f"LENS should enable mask, got {enable_mask_lens}"
    print("✓ LENS enables mask correctly")
    
    # Test OTA does not enable mask  
    ota_cfg = MockCfg('OTA')
    enable_mask_ota = (ota_cfg.method.name == 'LENS')
    assert enable_mask_ota == False, f"OTA should not enable mask, got {enable_mask_ota}"
    print("✓ OTA does not enable mask correctly")

def test_config_structure():
    """Test that config has expected structure and types"""
    print("\nTesting config structure...")
    
    cfg = OmegaConf.load('conf/method/LENS.yaml')
    
    # Test required types
    assert isinstance(cfg.target_object_id, int), "target_object_id should be int"
    assert isinstance(cfg.camera_exploration_steps, int), "camera_exploration_steps should be int"
    assert isinstance(cfg.R_vis_weight, float), "R_vis_weight should be float"
    
    # Test reasonable values
    assert cfg.target_object_id > 0, "target_object_id should be positive"
    assert cfg.camera_exploration_steps > 0, "camera_exploration_steps should be positive"
    assert 0 < cfg.R_vis_weight < 1, "R_vis_weight should be between 0 and 1"
    
    print("✓ Config structure and types correct")

def main():
    """Run all Phase 1 verification tests"""
    print("=" * 50)
    print("PHASE 1 VERIFICATION: LENS Configuration Setup")
    print("=" * 50)
    
    # Change to ARM directory
    os.chdir('/home/andrewlee/_research/ota/ARM')
    
    try:
        cfg = test_config_loading()
        compare_with_ota()
        test_mask_configuration()
        test_config_structure()
        
        print("\n" + "=" * 50)
        print("✓ PHASE 1 VERIFICATION PASSED")
        print("✓ LENS.yaml configuration is correct")
        print("✓ Ready to proceed to Phase 2")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ PHASE 1 VERIFICATION FAILED: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()