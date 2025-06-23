#!/usr/bin/env python3
"""
Phase 2 Verification Script: LENS Launch Utils
This script verifies that lens_launch_utils.py is correctly set up.
"""

import os
import sys
import re
import difflib
from typing import List

def test_file_exists():
    """Test that lens_launch_utils.py exists"""
    print("Testing lens_launch_utils.py existence...")
    
    if not os.path.exists('arm/ota/lens_launch_utils.py'):
        print("✗ lens_launch_utils.py does not exist")
        sys.exit(1)
    
    print("✓ lens_launch_utils.py exists")

def compare_function_signatures():
    """Compare function signatures between OTA and LENS launch utils"""
    print("\nComparing function signatures...")
    
    def extract_functions(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        return re.findall(r'^def ([^(]+\([^)]*\)):.*?(?=^def|\Z)', content, re.MULTILINE | re.DOTALL)
    
    ota_functions = extract_functions('arm/ota/launch_utils.py')
    lens_functions = extract_functions('arm/ota/lens_launch_utils.py')
    
    # Extract just the signatures (first line)
    ota_sigs = [func.split('\n')[0] for func in ota_functions]
    lens_sigs = [func.split('\n')[0] for func in lens_functions]
    
    if len(ota_sigs) != len(lens_sigs):
        print(f"✗ Function count mismatch: OTA={len(ota_sigs)}, LENS={len(lens_sigs)}")
        sys.exit(1)
    
    for i, (ota_sig, lens_sig) in enumerate(zip(ota_sigs, lens_sigs)):
        if ota_sig != lens_sig:
            print(f"✗ Function signature {i+1} differs:")
            print(f"  OTA:  {ota_sig}")
            print(f"  LENS: {lens_sig}")
            sys.exit(1)
    
    print(f"✓ All {len(ota_sigs)} function signatures match exactly")

def test_imports():
    """Test that imports are correctly modified"""
    print("\nTesting import modifications...")
    
    with open('arm/ota/lens_launch_utils.py', 'r') as f:
        content = f.read()
    
    # Check that LENS agent imports are present
    if 'from arm.agent.LENS_nbv_agent import LensNBVAgent' not in content:
        print("✗ Missing LensNBVAgent import")
        sys.exit(1)
    
    if 'from arm.agent.LENS_nbp_agent import LensNBPAgent' not in content:
        print("✗ Missing LensNBPAgent import")
        sys.exit(1)
    
    # Check that OTA agent imports are removed
    if 'from arm.ota.qattention_agent import QAttentionAgent' in content:
        print("✗ OTA QAttentionAgent import still present")
        sys.exit(1)
    
    if 'from arm.ota.qattention_stack_agent import QAttentionStackAgent' in content:
        print("✗ OTA QAttentionStackAgent import still present")
        sys.exit(1)
    
    print("✓ Import modifications correct")

def test_agent_instantiation():
    """Test that agent instantiations are correctly modified"""
    print("\nTesting agent instantiation modifications...")
    
    with open('arm/ota/lens_launch_utils.py', 'r') as f:
        content = f.read()
    
    # Check that LENS agents are instantiated
    if 'qattention_agent = LensNBVAgent(' not in content:
        print("✗ LensNBVAgent not instantiated")
        sys.exit(1)
    
    if 'rotation_agent = LensNBPAgent(' not in content:
        print("✗ LensNBPAgent not instantiated") 
        sys.exit(1)
    
    # Check that OTA agents are not instantiated
    if 'QAttentionAgent(' in content:
        print("✗ OTA QAttentionAgent still instantiated")
        sys.exit(1)
    
    if 'QAttentionStackAgent(' in content:
        print("✗ OTA QAttentionStackAgent still instantiated")
        sys.exit(1)
    
    print("✓ Agent instantiation modifications correct")

def test_preserved_logic():
    """Test that all other logic is preserved exactly"""
    print("\nTesting preserved logic...")
    
    def normalize_content(file_path):
        """Remove agent-related lines for comparison"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        normalized = []
        for line in lines:
            # Skip import lines that differ
            if ('qattention_agent import' in line or 
                'qattention_stack_agent import' in line or
                'LENS_nbv_agent import' in line or
                'LENS_nbp_agent import' in line):
                continue
            
            # Normalize agent instantiation lines
            if 'qattention_agent = ' in line:
                line = line.replace('QAttentionAgent(', 'AGENT_CLASS(')
                line = line.replace('LensNBVAgent(', 'AGENT_CLASS(')
            elif 'rotation_agent = ' in line:
                line = line.replace('QAttentionStackAgent(', 'STACK_AGENT_CLASS(')
                line = line.replace('LensNBPAgent(', 'STACK_AGENT_CLASS(')
            
            normalized.append(line)
        
        return normalized
    
    ota_normalized = normalize_content('arm/ota/launch_utils.py')
    lens_normalized = normalize_content('arm/ota/lens_launch_utils.py')
    
    if ota_normalized != lens_normalized:
        print("✗ Logic differs between OTA and LENS versions")
        print("Differences:")
        for i, (ota_line, lens_line) in enumerate(zip(ota_normalized, lens_normalized)):
            if ota_line != lens_line:
                print(f"  Line {i+1}:")
                print(f"    OTA:  {ota_line.rstrip()}")
                print(f"    LENS: {lens_line.rstrip()}")
                if i > 10:  # Limit output
                    print("    ... (more differences)")
                    break
        sys.exit(1)
    
    print("✓ All non-agent logic preserved exactly")

def test_constants_and_globals():
    """Test that constants and global variables are preserved"""
    print("\nTesting constants and globals...")
    
    def extract_constants(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find module-level constants
        constants = re.findall(r'^([A-Z_][A-Z0-9_]*)\s*=\s*(.+)$', content, re.MULTILINE)
        return constants
    
    ota_constants = extract_constants('arm/ota/launch_utils.py')
    lens_constants = extract_constants('arm/ota/lens_launch_utils.py')
    
    if ota_constants != lens_constants:
        print("✗ Constants differ between versions")
        print(f"  OTA: {ota_constants}")
        print(f"  LENS: {lens_constants}")
        sys.exit(1)
    
    print(f"✓ All {len(ota_constants)} constants preserved")

def test_import_compatibility():
    """Test that LENS launch utils can be imported (basic syntax check)"""
    print("\nTesting import compatibility...")
    
    try:
        # This will fail because the agents don't exist yet, but it tests syntax
        import ast
        with open('arm/ota/lens_launch_utils.py', 'r') as f:
            content = f.read()
        
        # Parse the file to check for syntax errors
        ast.parse(content)
        print("✓ lens_launch_utils.py has valid Python syntax")
        
    except SyntaxError as e:
        print(f"✗ Syntax error in lens_launch_utils.py: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✓ lens_launch_utils.py syntax is valid (import will fail until agents exist: {e})")

def main():
    """Run all Phase 2 verification tests"""
    print("=" * 50)
    print("PHASE 2 VERIFICATION: LENS Launch Utils")
    print("=" * 50)
    
    # Change to ARM directory
    os.chdir('/home/andrewlee/_research/ota/ARM')
    
    try:
        test_file_exists()
        compare_function_signatures()
        test_imports()
        test_agent_instantiation()
        test_preserved_logic()
        test_constants_and_globals()
        test_import_compatibility()
        
        print("\n" + "=" * 50)
        print("✓ PHASE 2 VERIFICATION PASSED")
        print("✓ lens_launch_utils.py correctly copies OTA with minimal changes")
        print("✓ Only agent imports and instantiations modified")
        print("✓ All other logic preserved exactly")
        print("✓ Ready to proceed to Phase 3 (Agent Creation)")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ PHASE 2 VERIFICATION FAILED: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()