#!/usr/bin/env python3
"""
Comprehensive Test Runner

Runs all tests across the single_turn_scenarios task including unit tests,
integration tests, performance tests, and security tests.

Requirements addressed: All requirements validation
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTestRunner:
    """Runs comprehensive test suite for single_turn_scenarios task."""
    
    def __init__(self):
        self.task_dir = Path(__file__).parent
        self.test_dir = self.task_dir / "tests"
        self.results = {}
        self.start_time = None
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test categories and return comprehensive results."""
        self.start_time = time.time()
        
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        test_categories = [
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Security Tests", self.run_security_tests),
            ("CLI Validation", self.run_cli_validation),
            ("Example Validation", self.run_example_validation),
            ("Final System Validation", self.run_final_validation)
        ]
        
        overall_success = True
        
        for category_name, test_function in test_categories:
            print(f"\n📋 Running {category_name}")
            print("-" * 40)
            
            try:
                category_result = test_function()
                self.results[category_name] = category_result
                
                if category_result.get("success", False):
                    print(f"✅ {category_name}: PASSED")
                else:
                    print(f"❌ {category_name}: FAILED")
                    overall_success = False
                    
            except Exception as e:
                logger.error(f"Error in {category_name}: {e}")
                self.results[category_name] = {
                    "success": False,
                    "error": str(e),
                    "tests_run": 0,
                    "tests_passed": 0
                }
                overall_success = False
        
        # Generate final report
        self.results["overall"] = {
            "success": overall_success,
            "duration": time.time() - self.start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.print_summary()
        return self.results
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run all unit tests."""
        unit_test_dir = self.test_dir / "unit"
        
        if not unit_test_dir.exists():
            return {"success": False, "error": "Unit test directory not found"}
        
        test_files = list(unit_test_dir.glob("test_*.py"))
        
        if not test_files:
            return {"success": False, "error": "No unit test files found"}
        
        results = {
            "success": True,
            "tests_run": 0,
            "tests_passed": 0,
            "test_results": {}
        }
        
        for test_file in test_files:
            print(f"  Running {test_file.name}...")
            
            try:
                # Run pytest on individual test file
                cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.task_dir)
                
                test_success = result.returncode == 0
                results["test_results"][test_file.name] = {
                    "success": test_success,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
                if test_success:
                    results["tests_passed"] += 1
                    print(f"    ✅ {test_file.name}")
                else:
                    results["success"] = False
                    print(f"    ❌ {test_file.name}")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:200]}...")
                
                results["tests_run"] += 1
                
            except Exception as e:
                results["success"] = False
                results["test_results"][test_file.name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"    ❌ {test_file.name}: {e}")
        
        return results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        integration_test_dir = self.test_dir / "integration"
        
        if not integration_test_dir.exists():
            return {"success": False, "error": "Integration test directory not found"}
        
        test_files = list(integration_test_dir.glob("test_*.py"))
        
        results = {
            "success": True,
            "tests_run": 0,
            "tests_passed": 0,
            "test_results": {}
        }
        
        for test_file in test_files:
            print(f"  Running {test_file.name}...")
            
            try:
                # Run integration test
                cmd = [sys.executable, str(test_file)]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.task_dir)
                
                test_success = result.returncode == 0
                results["test_results"][test_file.name] = {
                    "success": test_success,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
                if test_success:
                    results["tests_passed"] += 1
                    print(f"    ✅ {test_file.name}")
                else:
                    results["success"] = False
                    print(f"    ❌ {test_file.name}")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:200]}...")
                
                results["tests_run"] += 1
                
            except Exception as e:
                results["success"] = False
                results["test_results"][test_file.name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"    ❌ {test_file.name}: {e}")
        
        return results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        performance_test_dir = self.test_dir / "performance"
        
        if not performance_test_dir.exists():
            return {"success": False, "error": "Performance test directory not found"}
        
        test_files = list(performance_test_dir.glob("test_*.py"))
        
        results = {
            "success": True,
            "tests_run": 0,
            "tests_passed": 0,
            "test_results": {}
        }
        
        for test_file in test_files:
            print(f"  Running {test_file.name}...")
            
            try:
                # Run performance test
                cmd = [sys.executable, str(test_file)]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.task_dir, timeout=300)
                
                test_success = result.returncode == 0
                results["test_results"][test_file.name] = {
                    "success": test_success,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
                if test_success:
                    results["tests_passed"] += 1
                    print(f"    ✅ {test_file.name}")
                else:
                    results["success"] = False
                    print(f"    ❌ {test_file.name}")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:200]}...")
                
                results["tests_run"] += 1
                
            except subprocess.TimeoutExpired:
                results["success"] = False
                results["test_results"][test_file.name] = {
                    "success": False,
                    "error": "Test timed out after 300 seconds"
                }
                print(f"    ❌ {test_file.name}: Timeout")
            except Exception as e:
                results["success"] = False
                results["test_results"][test_file.name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"    ❌ {test_file.name}: {e}")
        
        return results
    
    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests."""
        security_test_dir = self.test_dir / "security"
        
        if not security_test_dir.exists():
            return {"success": False, "error": "Security test directory not found"}
        
        test_files = list(security_test_dir.glob("test_*.py"))
        
        results = {
            "success": True,
            "tests_run": 0,
            "tests_passed": 0,
            "test_results": {}
        }
        
        for test_file in test_files:
            print(f"  Running {test_file.name}...")
            
            try:
                # Run security test
                cmd = [sys.executable, str(test_file)]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.task_dir, timeout=120)
                
                test_success = result.returncode == 0
                results["test_results"][test_file.name] = {
                    "success": test_success,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
                if test_success:
                    results["tests_passed"] += 1
                    print(f"    ✅ {test_file.name}")
                else:
                    results["success"] = False
                    print(f"    ❌ {test_file.name}")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:200]}...")
                
                results["tests_run"] += 1
                
            except subprocess.TimeoutExpired:
                results["success"] = False
                results["test_results"][test_file.name] = {
                    "success": False,
                    "error": "Security test timed out after 120 seconds"
                }
                print(f"    ❌ {test_file.name}: Timeout")
            except Exception as e:
                results["success"] = False
                results["test_results"][test_file.name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"    ❌ {test_file.name}: {e}")
        
        return results
    
    def run_cli_validation(self) -> Dict[str, Any]:
        """Validate CLI commands and usage examples."""
        print("  Validating CLI commands...")
        
        results = {
            "success": True,
            "tests_run": 0,
            "tests_passed": 0,
            "cli_tests": {}
        }
        
        # Test CLI commands
        cli_tests = [
            {
                "name": "list_tasks",
                "command": [sys.executable, "-m", "lm_eval", "--tasks", "list"],
                "expected_in_output": "single_turn_scenarios"
            },
            {
                "name": "task_info",
                "command": [sys.executable, "-m", "lm_eval", "--tasks", "single_turn_scenarios_code_completion", "--help"],
                "expected_in_output": "single_turn_scenarios"
            }
        ]
        
        for test in cli_tests:
            print(f"    Testing {test['name']}...")
            results["tests_run"] += 1
            
            try:
                result = subprocess.run(
                    test["command"], 
                    capture_output=True, 
                    text=True, 
                    timeout=30,
                    cwd=self.task_dir.parent.parent.parent
                )
                
                success = (
                    result.returncode == 0 and 
                    test["expected_in_output"] in result.stdout
                )
                
                results["cli_tests"][test["name"]] = {
                    "success": success,
                    "stdout": result.stdout[:500],
                    "stderr": result.stderr[:500] if result.stderr else ""
                }
                
                if success:
                    results["tests_passed"] += 1
                    print(f"      ✅ {test['name']}")
                else:
                    results["success"] = False
                    print(f"      ❌ {test['name']}")
                
            except subprocess.TimeoutExpired:
                results["success"] = False
                results["cli_tests"][test["name"]] = {
                    "success": False,
                    "error": "CLI test timed out"
                }
                print(f"      ❌ {test['name']}: Timeout")
            except Exception as e:
                results["success"] = False
                results["cli_tests"][test["name"]] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"      ❌ {test['name']}: {e}")
        
        return results
    
    def run_example_validation(self) -> Dict[str, Any]:
        """Validate example scripts."""
        examples_dir = self.task_dir / "examples"
        
        if not examples_dir.exists():
            return {"success": False, "error": "Examples directory not found"}
        
        example_files = list(examples_dir.glob("*.py"))
        
        results = {
            "success": True,
            "tests_run": 0,
            "tests_passed": 0,
            "example_results": {}
        }
        
        for example_file in example_files:
            print(f"  Running {example_file.name}...")
            results["tests_run"] += 1
            
            try:
                # Run example script
                cmd = [sys.executable, str(example_file)]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.task_dir, timeout=60)
                
                test_success = result.returncode == 0
                results["example_results"][example_file.name] = {
                    "success": test_success,
                    "stdout": result.stdout[:500],
                    "stderr": result.stderr[:500] if result.stderr else ""
                }
                
                if test_success:
                    results["tests_passed"] += 1
                    print(f"    ✅ {example_file.name}")
                else:
                    results["success"] = False
                    print(f"    ❌ {example_file.name}")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:200]}...")
                
            except subprocess.TimeoutExpired:
                results["success"] = False
                results["example_results"][example_file.name] = {
                    "success": False,
                    "error": "Example timed out after 60 seconds"
                }
                print(f"    ❌ {example_file.name}: Timeout")
            except Exception as e:
                results["success"] = False
                results["example_results"][example_file.name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"    ❌ {example_file.name}: {e}")
        
        return results
    
    def run_final_validation(self) -> Dict[str, Any]:
        """Run the final validation script."""
        final_validation_script = self.task_dir / "final_validation.py"
        
        if not final_validation_script.exists():
            return {"success": False, "error": "Final validation script not found"}
        
        print("  Running final validation script...")
        
        try:
            cmd = [sys.executable, str(final_validation_script)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.task_dir, timeout=300)
            
            success = result.returncode == 0
            
            return {
                "success": success,
                "tests_run": 1,
                "tests_passed": 1 if success else 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "tests_run": 1,
                "tests_passed": 0,
                "error": "Final validation timed out after 300 seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "tests_run": 1,
                "tests_passed": 0,
                "error": str(e)
            }
    
    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("🏁 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        total_passed = 0
        
        for category, result in self.results.items():
            if category == "overall":
                continue
                
            tests_run = result.get("tests_run", 0)
            tests_passed = result.get("tests_passed", 0)
            success = result.get("success", False)
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {category}: {tests_passed}/{tests_run}")
            
            total_tests += tests_run
            total_passed += tests_passed
        
        overall = self.results.get("overall", {})
        duration = overall.get("duration", 0)
        
        print(f"\n📊 Overall Results:")
        print(f"   Total Tests: {total_passed}/{total_tests}")
        print(f"   Success Rate: {(total_passed/max(total_tests, 1)*100):.1f}%")
        print(f"   Duration: {duration:.2f} seconds")
        
        if overall.get("success", False):
            print("\n🎉 ALL TESTS PASSED!")
            print("The single_turn_scenarios task is fully validated and ready for production use.")
        else:
            print(f"\n⚠️  {total_tests - total_passed} TESTS FAILED")
            print("Please review and fix the failing tests before deployment.")
        
        # Save detailed results
        results_file = self.task_dir / "test_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n📄 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")
    
    def create_deployment_checklist(self) -> str:
        """Create a deployment checklist based on test results."""
        checklist = []
        
        checklist.append("# Single Turn Scenarios Deployment Checklist")
        checklist.append("")
        checklist.append("## Pre-Deployment Validation")
        
        for category, result in self.results.items():
            if category == "overall":
                continue
                
            success = result.get("success", False)
            status = "✅" if success else "❌"
            checklist.append(f"- {status} {category}")
            
            if not success:
                checklist.append(f"  - Review failed tests in {category}")
                if "error" in result:
                    checklist.append(f"  - Error: {result['error']}")
        
        checklist.append("")
        checklist.append("## Deployment Steps")
        checklist.append("- [ ] All tests passing")
        checklist.append("- [ ] Documentation updated")
        checklist.append("- [ ] Security audit completed")
        checklist.append("- [ ] Performance benchmarks acceptable")
        checklist.append("- [ ] CLI commands validated")
        checklist.append("- [ ] Example scripts working")
        checklist.append("")
        checklist.append("## Post-Deployment")
        checklist.append("- [ ] Monitor evaluation performance")
        checklist.append("- [ ] Validate with real model backends")
        checklist.append("- [ ] Check analysis tool compatibility")
        checklist.append("- [ ] Gather user feedback")
        
        return "\n".join(checklist)


def main():
    """Main function to run comprehensive tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--save-results", action="store_true", 
                       help="Save detailed results to JSON file")
    parser.add_argument("--create-checklist", action="store_true",
                       help="Create deployment checklist")
    parser.add_argument("--category", choices=[
        "unit", "integration", "performance", "security", "cli", "examples", "final"
    ], help="Run only specific test category")
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner()
    
    if args.category:
        # Run specific category
        category_map = {
            "unit": runner.run_unit_tests,
            "integration": runner.run_integration_tests,
            "performance": runner.run_performance_tests,
            "security": runner.run_security_tests,
            "cli": runner.run_cli_validation,
            "examples": runner.run_example_validation,
            "final": runner.run_final_validation
        }
        
        print(f"Running {args.category} tests only...")
        result = category_map[args.category]()
        
        if result.get("success", False):
            print(f"✅ {args.category.title()} tests passed")
            return 0
        else:
            print(f"❌ {args.category.title()} tests failed")
            return 1
    else:
        # Run all tests
        results = runner.run_all_tests()
        
        if args.create_checklist:
            checklist = runner.create_deployment_checklist()
            checklist_file = runner.task_dir / "deployment_checklist.md"
            with open(checklist_file, 'w') as f:
                f.write(checklist)
            print(f"\n📋 Deployment checklist created: {checklist_file}")
        
        return 0 if results["overall"]["success"] else 1


if __name__ == "__main__":
    exit(main())