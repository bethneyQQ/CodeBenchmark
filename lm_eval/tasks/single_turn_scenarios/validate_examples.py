#!/usr/bin/env python3
"""
Validate Usage Examples

This script validates that the usage examples in README.md are correct
and can be executed successfully.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_api_keys():
    """Check if API keys are available."""
    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'DASHSCOPE_API_KEY': os.getenv('DASHSCOPE_API_KEY')
    }
    
    available_keys = {k: v for k, v in api_keys.items() if v}
    
    print("🔑 API Key Status:")
    for key, value in api_keys.items():
        status = "✅ Available" if value else "❌ Not set"
        print(f"  {key}: {status}")
    
    return available_keys

def test_task_availability():
    """Test if single_turn_scenarios tasks are available."""
    print("\n📋 Testing Task Availability:")
    
    try:
        from lm_eval.tasks import TaskManager
        tm = TaskManager()
        
        expected_tasks = [
            'single_turn_scenarios_code_completion',
            'single_turn_scenarios_bug_fix',
            'single_turn_scenarios_suite',
            'single_turn_scenarios_python'
        ]
        
        available_tasks = []
        missing_tasks = []
        
        for task in expected_tasks:
            if task in tm.all_tasks:
                available_tasks.append(task)
                print(f"  ✅ {task}")
            else:
                missing_tasks.append(task)
                print(f"  ❌ {task}")
        
        return len(missing_tasks) == 0
        
    except Exception as e:
        print(f"  ❌ Error loading tasks: {e}")
        return False

def test_model_examples():
    """Test basic model examples with minimal execution."""
    print("\n🤖 Testing Model Examples:")
    
    available_keys = check_api_keys()
    
    # Test examples based on available API keys
    test_cases = []
    
    if 'OPENAI_API_KEY' in available_keys:
        test_cases.append({
            'name': 'OpenAI GPT-3.5-turbo',
            'command': [
                'lm_eval', '--model', 'openai-chat',
                '--model_args', 'model=gpt-3.5-turbo',
                '--tasks', 'single_turn_scenarios_code_completion',
                '--limit', '1',
                '--show_config'
            ]
        })
    
    if 'ANTHROPIC_API_KEY' in available_keys:
        test_cases.append({
            'name': 'Anthropic Claude Haiku',
            'command': [
                'lm_eval', '--model', 'anthropic',
                '--model_args', 'model=claude-3-haiku-20240307',
                '--tasks', 'single_turn_scenarios_code_completion',
                '--limit', '1',
                '--show_config'
            ]
        })
    
    if 'DASHSCOPE_API_KEY' in available_keys:
        test_cases.append({
            'name': 'DashScope Qwen Turbo',
            'command': [
                'lm_eval', '--model', 'dashscope',
                '--model_args', 'model=qwen-turbo',
                '--tasks', 'single_turn_scenarios_code_completion',
                '--limit', '1',
                '--show_config'
            ]
        })
    
    # Always test HuggingFace model (no API key needed)
    test_cases.append({
        'name': 'HuggingFace DialoGPT (fallback)',
        'command': [
            'lm_eval', '--model', 'hf',
            '--model_args', 'pretrained=microsoft/DialoGPT-medium',
            '--tasks', 'single_turn_scenarios_code_completion',
            '--limit', '1',
            '--show_config'
        ]
    })
    
    results = []
    
    for test_case in test_cases:
        print(f"\n  Testing {test_case['name']}...")
        
        try:
            result = subprocess.run(
                test_case['command'],
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"    ✅ {test_case['name']}: SUCCESS")
                results.append(True)
            else:
                print(f"    ❌ {test_case['name']}: FAILED")
                print(f"    Error: {result.stderr[:200]}...")
                results.append(False)
                
        except subprocess.TimeoutExpired:
            print(f"    ⏰ {test_case['name']}: TIMEOUT")
            results.append(False)
        except Exception as e:
            print(f"    ❌ {test_case['name']}: ERROR - {e}")
            results.append(False)
    
    return all(results)

def test_metadata_filtering():
    """Test metadata filtering examples."""
    print("\n🔍 Testing Metadata Filtering:")
    
    # Test with a simple model that doesn't require API keys
    test_command = [
        'lm_eval', '--model', 'hf',
        '--model_args', 'pretrained=microsoft/DialoGPT-medium',
        '--tasks', 'single_turn_scenarios_code_completion',
        '--metadata', '{"difficulty":"simple"}',
        '--limit', '1',
        '--show_config'
    ]
    
    try:
        result = subprocess.run(
            test_command,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("  ✅ Metadata filtering: SUCCESS")
            return True
        else:
            print("  ❌ Metadata filtering: FAILED")
            print(f"  Error: {result.stderr[:200]}...")
            return False
            
    except Exception as e:
        print(f"  ❌ Metadata filtering: ERROR - {e}")
        return False

def test_configuration_validation():
    """Test configuration file validation."""
    print("\n⚙️  Testing Configuration Validation:")
    
    config_files = [
        'model_configs/dashscope.yaml',
        'model_configs/claude_code.yaml',
        'model_configs/openai.yaml'
    ]
    
    task_dir = Path(__file__).parent
    results = []
    
    for config_file in config_files:
        config_path = task_dir / config_file
        
        if config_path.exists():
            print(f"  ✅ {config_file}: EXISTS")
            
            # Try to load YAML
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check for required fields
                if 'model_name' in config:
                    print(f"    ✅ Valid YAML structure")
                    results.append(True)
                else:
                    print(f"    ❌ Missing required fields")
                    results.append(False)
                    
            except Exception as e:
                print(f"    ❌ YAML parsing error: {e}")
                results.append(False)
        else:
            print(f"  ❌ {config_file}: NOT FOUND")
            results.append(False)
    
    return all(results)

def main():
    """Run all validation tests."""
    print("🧪 Validating Single Turn Scenarios Usage Examples")
    print("=" * 60)
    
    tests = [
        ("API Keys", check_api_keys),
        ("Task Availability", test_task_availability),
        ("Configuration Files", test_configuration_validation),
        ("Metadata Filtering", test_metadata_filtering),
        ("Model Examples", test_model_examples)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test:")
        print("-" * 40)
        
        try:
            if test_name == "API Keys":
                # API keys test returns dict, not boolean
                available_keys = test_func()
                result = len(available_keys) > 0
                print(f"Result: {'✅ PASS' if result else '⚠️  WARNING - No API keys available'}")
            else:
                result = test_func()
                print(f"Result: {'✅ PASS' if result else '❌ FAIL'}")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("The usage examples in README.md should work correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")
        print("Some usage examples may need to be updated.")
        print("\nCommon issues:")
        print("- Missing API keys (set OPENAI_API_KEY, ANTHROPIC_API_KEY, or DASHSCOPE_API_KEY)")
        print("- Missing dependencies (run: pip install lm-eval)")
        print("- GPU memory issues (use smaller batch sizes or CPU)")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)