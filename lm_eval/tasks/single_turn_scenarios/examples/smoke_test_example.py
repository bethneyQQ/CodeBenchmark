#!/usr/bin/env python3
"""
Smoke test example for single_turn_scenarios task.

This script performs basic validation to ensure the task is properly configured
and can load data correctly.
"""

import sys
import os
import json
from pathlib import Path

# Add lm_eval to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_task_registration():
    """Test that tasks are properly registered."""
    print("Testing task registration...")
    
    try:
        from lm_eval.tasks import TaskManager
        
        task_manager = TaskManager()
        all_tasks = task_manager.all_tasks
        
        # Check for main suite task
        suite_found = "single_turn_scenarios_suite" in all_tasks
        print(f"  Suite task registered: {suite_found}")
        
        # Check for individual scenario tasks
        expected_scenarios = [
            "single_turn_scenarios_code_completion",
            "single_turn_scenarios_bug_fix",
            "single_turn_scenarios_algorithm_implementation",
            "single_turn_scenarios_security"
        ]
        
        scenarios_found = 0
        for scenario in expected_scenarios:
            if scenario in all_tasks:
                scenarios_found += 1
                print(f"  {scenario}: ✓")
            else:
                print(f"  {scenario}: ✗")
        
        print(f"  Scenarios found: {scenarios_found}/{len(expected_scenarios)}")
        return suite_found and scenarios_found > 0
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_dataset_loading():
    """Test that the dataset can be loaded."""
    print("\nTesting dataset loading...")
    
    try:
        # Import utils from the task
        task_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(task_dir))
        
        from utils import load_dataset, validate_problem_schema
        
        # Test basic dataset loading
        dataset = load_dataset()
        print(f"  Dataset loaded: {len(dataset)} problems")
        
        # Test a few samples
        if len(dataset) > 0:
            sample = dataset[0]
            is_valid = validate_problem_schema(sample)
            print(f"  Sample validation: {is_valid}")
            print(f"  Sample ID: {sample.get('id', 'N/A')}")
            print(f"  Sample scenario: {sample.get('scenario', 'N/A')}")
            print(f"  Sample language: {sample.get('language', 'N/A')}")
        
        return len(dataset) > 0
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_metadata_filtering():
    """Test metadata filtering functionality."""
    print("\nTesting metadata filtering...")
    
    try:
        task_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(task_dir))
        
        from utils import load_dataset, filter_by_metadata
        
        # Load full dataset
        full_dataset = load_dataset()
        full_count = len(full_dataset)
        print(f"  Full dataset: {full_count} problems")
        
        # Test scenario filtering
        if full_count > 0:
            scenarios = set(item['scenario'] for item in full_dataset)
            print(f"  Available scenarios: {sorted(scenarios)}")
            
            if 'code_completion' in scenarios:
                filtered = filter_by_metadata(full_dataset, {'scenario': 'code_completion'})
                print(f"  Code completion filtered: {len(filtered)} problems")
            
            # Test language filtering
            languages = set(item['language'] for item in full_dataset)
            print(f"  Available languages: {sorted(languages)}")
            
            if 'python' in languages:
                filtered = filter_by_metadata(full_dataset, {'language': 'python'})
                print(f"  Python filtered: {len(filtered)} problems")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_configuration_files():
    """Test that configuration files are present and valid."""
    print("\nTesting configuration files...")
    
    try:
        task_dir = Path(__file__).parent.parent
        
        # Check for essential files
        essential_files = [
            "problems.jsonl",
            "context_configs.json",
            "single_turn_scenarios_suite.yaml",
            "utils.py",
            "metrics.py"
        ]
        
        files_found = 0
        for file_name in essential_files:
            file_path = task_dir / file_name
            if file_path.exists():
                files_found += 1
                print(f"  {file_name}: ✓")
            else:
                print(f"  {file_name}: ✗")
        
        # Check YAML files
        yaml_files = list(task_dir.glob("*.yaml"))
        print(f"  YAML task files: {len(yaml_files)}")
        
        return files_found == len(essential_files)
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    """Run all smoke tests."""
    print("Single Turn Scenarios - Smoke Test")
    print("=" * 40)
    
    tests = [
        ("Task Registration", test_task_registration),
        ("Dataset Loading", test_dataset_loading),
        ("Metadata Filtering", test_metadata_filtering),
        ("Configuration Files", test_configuration_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "PASS" if result else "FAIL"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: ERROR - {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 40)
    print(f"Smoke Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The task appears to be properly configured.")
    else:
        print("✗ Some tests failed. Check the configuration and dependencies.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)