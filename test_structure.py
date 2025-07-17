#!/usr/bin/env python3
"""Test script to validate the FastMoE package structure."""

import os
import sys

def test_package_structure():
    """Test that all required files are present."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = [
        'fastmoe/__init__.py',
        'fastmoe/models/__init__.py',
        'fastmoe/models/qwen3_moe.py',
        'fastmoe/models/interfaces.py',
        'fastmoe/models/utils.py',
        'fastmoe/serve/__init__.py',
        'fastmoe/serve/launch_server.py',
        'benchmarks/mtbench/benchmark.sh',
        'run_qwen3_benchmarks.sh',
        'requirements.txt',
        'setup.py',
        'README.md',
        '.gitignore'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("All required files are present!")
        return True

def test_import_structure():
    """Test the import structure without dependencies."""
    print("Testing import structure...")
    
    # Test that we can at least import the module files as text
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check that Python files have valid syntax
    python_files = [
        'fastmoe/__init__.py',
        'fastmoe/models/__init__.py',
        'fastmoe/models/interfaces.py',
        'fastmoe/models/utils.py',
        'fastmoe/serve/__init__.py',
        'setup.py'
    ]
    
    for file_path in python_files:
        full_path = os.path.join(base_dir, file_path)
        try:
            with open(full_path, 'r') as f:
                compile(f.read(), full_path, 'exec')
            print(f"  ✓ {file_path} - syntax OK")
        except SyntaxError as e:
            print(f"  ✗ {file_path} - syntax error: {e}")
            return False
    
    return True

def test_scripts_executable():
    """Test that scripts are executable."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        'benchmarks/mtbench/benchmark.sh',
        'run_qwen3_benchmarks.sh'
    ]
    
    for script in scripts:
        full_path = os.path.join(base_dir, script)
        if not os.access(full_path, os.X_OK):
            print(f"  ✗ {script} - not executable")
            return False
        else:
            print(f"  ✓ {script} - executable")
    
    return True

def main():
    """Main test function."""
    print("FastMoE Package Structure Test")
    print("=" * 40)
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Import Structure", test_import_structure),
        ("Script Permissions", test_scripts_executable)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("All tests PASSED! ✓")
        return 0
    else:
        print("Some tests FAILED! ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())