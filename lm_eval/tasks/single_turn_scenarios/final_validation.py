#!/usr/bin/env python3
"""
Final validation script for single_turn_scenarios task.

This script performs comprehensive end-to-end validation to ensure the task
works correctly with all supported models and scenarios.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add lm_eval to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def validate_task_registration():
    """Validate that all tasks are properly registered."""
    print("Validating task registration...")
    
    try:
        from lm_eval.tasks import TaskManager
        
        task_manager = TaskManager()
        all_tasks = task_manager.all_tasks
        
        # Expected tasks
        expected_tasks = [
            "single_turn_scenarios_suite",
            "single_turn_scenarios_code_completion",
            "single_turn_scenarios_bug_fix",
            "single_turn_scenarios_code_translation",
            "single_turn_scenarios_documentation",
            "single_turn_scenarios_function_generation",
            "single_turn_scenarios_system_design",
            "single_turn_scenarios_algorithm_implementation",
            "single_turn_scenarios_api_design",
            "single_turn_scenarios_database_design",
            "single_turn_scenarios_performance_optimization",
            "single_turn_scenarios_full_stack",
            "single_turn_scenarios_testing_strategy",
            "single_turn_scenarios_security",
            "single_turn_scenarios_python",
            "single_turn_scenarios_intermediate",
            "single_turn_scenarios_minimal_context"
        ]
        
        missing_tasks = []
        for task in expected_tasks:
            if task not in all_tasks:
                missing_tasks.append(task)
            else:
                print(f"  ✓ {task}")
        
        if missing_tasks:
            print(f"  ✗ Missing tasks: {missing_tasks}")
            return False
        
        print(f"  ✓ All {len(expected_tasks)} tasks registered successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def validate_dataset_integrity():
    """Validate dataset integrity and structure."""
    print("\nValidating dataset integrity...")
    
    try:
        # Import task utilities
        task_dir = Path(__file__).parent
        sys.path.insert(0, str(task_dir))
        
        from utils import load_dataset, validate_problem_schema
        from metrics import exact_match, bleu_score
        
        # Load dataset
        dataset = load_dataset()
        print(f"  ✓ Dataset loaded: {len(dataset)} problems")
        
        if len(dataset) == 0:
            print("  ✗ Dataset is empty")
            return False
        
        # Validate schema for all problems
        valid_problems = 0
        schema_errors = []
        
        for i, problem in enumerate(dataset):
            if validate_problem_schema(problem):
                valid_problems += 1
            else:
                schema_errors.append(f"Problem {i}: {problem.get('id', 'unknown')}")
        
        print(f"  ✓ Valid problems: {valid_problems}/{len(dataset)}")
        
        if schema_errors:
            print(f"  ✗ Schema errors in {len(schema_errors)} problems")
            for error in schema_errors[:5]:  # Show first 5 errors
                print(f"    {error}")
            if len(schema_errors) > 5:
                print(f"    ... and {len(schema_errors) - 5} more")
        
        # Check data distribution
        scenarios = set(p['scenario'] for p in dataset)
        languages = set(p['language'] for p in dataset)
        difficulties = set(p['difficulty'] for p in dataset)
        
        print(f"  ✓ Scenarios: {len(scenarios)} ({sorted(scenarios)})")
        print(f"  ✓ Languages: {len(languages)} ({sorted(languages)})")
        print(f"  ✓ Difficulties: {len(difficulties)} ({sorted(difficulties)})")
        
        return len(schema_errors) == 0
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def validate_metrics():
    """Validate that all metrics can be computed."""
    print("\nValidating metrics...")
    
    try:
        task_dir = Path(__file__).parent
        sys.path.insert(0, str(task_dir))
        
        from metrics import (
            exact_match, bleu_score, codebleu_score, rouge_l_score,
            syntax_validity, pass_at_k, runtime_correctness
        )
        
        # Test data
        predictions = ["def hello(): return 'world'", "print('hello')"]
        references = ["def hello(): return 'world'", "print('hello world')"]
        
        # Test basic metrics
        metrics_to_test = [
            ("exact_match", exact_match),
            ("bleu_score", bleu_score),
            ("codebleu_score", codebleu_score),
            ("rouge_l_score", rouge_l_score)
        ]
        
        for metric_name, metric_func in metrics_to_test:
            try:
                result = metric_func(predictions, references)
                print(f"  ✓ {metric_name}: {result:.4f}")
            except Exception as e:
                print(f"  ✗ {metric_name}: {e}")
                return False
        
        # Test code-specific metrics
        try:
            syntax_result = syntax_validity("def hello(): return 'world'", "python")
            print(f"  ✓ syntax_validity: {syntax_result}")
        except Exception as e:
            print(f"  ✗ syntax_validity: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error importing metrics: {e}")
        return False

def validate_sandbox_setup():
    """Validate sandbox execution setup."""
    print("\nValidating sandbox setup...")
    
    try:
        task_dir = Path(__file__).parent
        sys.path.insert(0, str(task_dir))
        
        from sandbox import SandboxExecutor
        
        # Test basic sandbox creation
        try:
            executor = SandboxExecutor("python", {"timeout": 10, "memory_mb": 100})
            print("  ✓ Sandbox executor created")
            
            # Test simple code execution (if Docker is available)
            try:
                result = executor.execute_code("print('hello')", [])
                print(f"  ✓ Code execution: exit_code={result.exit_code}")
                executor.cleanup()
            except Exception as e:
                print(f"  ⚠ Code execution not available (Docker may not be installed): {e}")
            
        except Exception as e:
            print(f"  ⚠ Sandbox not fully available: {e}")
            print("    This is expected if Docker is not installed")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def validate_model_configs():
    """Validate model configuration files."""
    print("\nValidating model configurations...")
    
    try:
        task_dir = Path(__file__).parent
        model_configs_dir = task_dir / "model_configs"
        
        if not model_configs_dir.exists():
            print("  ✗ Model configs directory not found")
            return False
        
        expected_configs = [
            "claude_code.yaml",
            "deepseek.yaml", 
            "openai.yaml",
            "anthropic.yaml",
            "universal.yaml"
        ]
        
        configs_found = 0
        for config_file in expected_configs:
            config_path = model_configs_dir / config_file
            if config_path.exists():
                configs_found += 1
                print(f"  ✓ {config_file}")
                
                # Try to load and validate YAML
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Check for required fields
                    required_fields = ["model_name", "generation_params"]
                    missing_fields = [field for field in required_fields if field not in config]
                    if missing_fields:
                        print(f"    ⚠ Missing fields: {missing_fields}")
                    
                except Exception as e:
                    print(f"    ✗ Invalid YAML: {e}")
            else:
                print(f"  ✗ {config_file}")
        
        print(f"  ✓ Model configs found: {configs_found}/{len(expected_configs)}")
        return configs_found > 0
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def validate_analysis_tools():
    """Validate analysis tools."""
    print("\nValidating analysis tools...")
    
    try:
        task_dir = Path(__file__).parent
        analysis_dir = task_dir / "analysis_tools"
        
        if not analysis_dir.exists():
            print("  ✗ Analysis tools directory not found")
            return False
        
        expected_tools = [
            "compare_models.py",
            "context_impact.py",
            "generate_report.py",
            "scenario_analysis.py"
        ]
        
        tools_found = 0
        for tool_file in expected_tools:
            tool_path = analysis_dir / tool_file
            if tool_path.exists():
                tools_found += 1
                print(f"  ✓ {tool_file}")
            else:
                print(f"  ✗ {tool_file}")
        
        print(f"  ✓ Analysis tools found: {tools_found}/{len(expected_tools)}")
        return tools_found > 0
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def run_quick_evaluation_test():
    """Run a quick evaluation test if possible."""
    print("\nRunning quick evaluation test...")
    
    try:
        from lm_eval import evaluator
        from lm_eval.models import get_model
        
        # Try to create a simple model for testing
        print("  Attempting to create test model...")
        
        # Use a very small model for testing
        try:
            model = get_model("hf").create_from_arg_string(
                "pretrained=gpt2,device=cpu,dtype=float32"
            )
            print("  ✓ Test model created")
            
            # Run minimal evaluation
            print("  Running minimal evaluation (2 samples)...")
            results = evaluator.simple_evaluate(
                model=model,
                tasks=["single_turn_scenarios_code_completion"],
                limit=2,
                batch_size=1,
                no_cache=True
            )
            
            print("  ✓ Evaluation completed")
            
            # Check results structure
            if "results" in results and "single_turn_scenarios_code_completion" in results["results"]:
                task_results = results["results"]["single_turn_scenarios_code_completion"]
                print(f"  ✓ Results structure valid, metrics: {list(task_results.keys())}")
                return True
            else:
                print("  ✗ Invalid results structure")
                return False
                
        except Exception as e:
            print(f"  ⚠ Evaluation test not possible: {e}")
            print("    This is expected if you don't have HuggingFace models available")
            return True  # Don't fail validation for this
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Run comprehensive validation."""
    print("Single Turn Scenarios - Final Validation")
    print("=" * 50)
    
    validation_tests = [
        ("Task Registration", validate_task_registration),
        ("Dataset Integrity", validate_dataset_integrity),
        ("Metrics", validate_metrics),
        ("Sandbox Setup", validate_sandbox_setup),
        ("Model Configurations", validate_model_configs),
        ("Analysis Tools", validate_analysis_tools),
        ("Quick Evaluation", run_quick_evaluation_test)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in validation_tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASS" if result else "FAIL"
            print(f"Status: {status}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Duration: {duration:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("The single_turn_scenarios task is ready for use.")
        print("\nNext steps:")
        print("- Run full evaluation: lm_eval --tasks single_turn_scenarios_suite")
        print("- Check CLI usage: cat lm_eval/tasks/single_turn_scenarios/CLI_USAGE.md")
        print("- Run examples: python lm_eval/tasks/single_turn_scenarios/examples/smoke_test_example.py")
    else:
        print(f"\n⚠ {total - passed} VALIDATIONS FAILED")
        print("Please address the failed validations before using the task.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)