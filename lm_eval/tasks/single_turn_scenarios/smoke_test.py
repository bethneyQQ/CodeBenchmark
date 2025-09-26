#!/usr/bin/env python3
"""
Single Turn Scenarios - Smoke Testing and Validation

This script performs comprehensive smoke tests to validate the environment
and ensure all components are working correctly.
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add the lm_eval path to sys.path for imports
current_dir = Path(__file__).parent
lm_eval_root = current_dir.parent.parent.parent
sys.path.insert(0, str(lm_eval_root))

try:
    import docker
    import yaml
    import jsonschema
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run the setup script first: ./setupEvaluationEnvironment.sh --install-deps")
    sys.exit(1)

@dataclass
class TestResult:
    """Represents the result of a single test."""
    name: str
    status: str  # "pass", "fail", "skip", "warning"
    message: str
    duration: float
    details: Optional[Dict[str, Any]] = None

@dataclass
class SmokeTestReport:
    """Comprehensive smoke test report."""
    timestamp: str
    environment: Dict[str, str]
    total_tests: int
    passed: int
    failed: int
    skipped: int
    warnings: int
    duration: float
    results: List[TestResult]

class SmokeTestRunner:
    """Main smoke test runner class."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
        self.current_dir = Path(__file__).parent
        
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test and capture the result."""
        print(f"🧪 Running {test_name}...")
        start_time = time.time()
        
        try:
            result = test_func(*args, **kwargs)
            duration = time.time() - start_time
            
            if isinstance(result, tuple):
                status, message, details = result
            else:
                status, message, details = "pass", str(result), None
                
            test_result = TestResult(
                name=test_name,
                status=status,
                message=message,
                duration=duration,
                details=details
            )
            
            # Print result
            status_emoji = {
                "pass": "✅",
                "fail": "❌", 
                "skip": "⏭️",
                "warning": "⚠️"
            }
            print(f"   {status_emoji.get(status, '❓')} {message} ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                name=test_name,
                status="fail",
                message=f"Exception: {str(e)}",
                duration=duration,
                details={"traceback": traceback.format_exc()}
            )
            print(f"   ❌ Exception: {str(e)} ({duration:.2f}s)")
        
        self.results.append(test_result)
        return test_result
    
    def test_environment_info(self) -> Tuple[str, str, Dict[str, Any]]:
        """Collect environment information."""
        env_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": str(Path.cwd()),
            "lm_eval_path": str(lm_eval_root),
            "task_path": str(self.current_dir)
        }
        
        # Check for required executables
        executables = ["python", "node", "java", "docker", "gcc", "go", "rustc"]
        for exe in executables:
            try:
                result = subprocess.run([exe, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                env_info[f"{exe}_version"] = result.stdout.strip().split('\n')[0]
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                env_info[f"{exe}_version"] = "Not available"
        
        return "pass", "Environment information collected", env_info
    
    def test_docker_connectivity(self) -> Tuple[str, str, Dict[str, Any]]:
        """Test Docker daemon connectivity."""
        try:
            client = docker.from_env()
            info = client.info()
            version = client.version()
            
            details = {
                "docker_version": version.get("Version", "Unknown"),
                "api_version": version.get("ApiVersion", "Unknown"),
                "containers_running": info.get("ContainersRunning", 0),
                "images_count": info.get("Images", 0)
            }
            
            return "pass", f"Docker daemon accessible (v{details['docker_version']})", details
            
        except docker.errors.DockerException as e:
            return "fail", f"Docker daemon not accessible: {str(e)}", {"error": str(e)}
        except Exception as e:
            return "fail", f"Docker connectivity error: {str(e)}", {"error": str(e)}
    
    def test_sandbox_images(self) -> Tuple[str, str, Dict[str, Any]]:
        """Test availability of sandbox Docker images."""
        try:
            client = docker.from_env()
            
            # Check for existing images or Dockerfiles
            docker_dir = self.current_dir / "docker"
            if not docker_dir.exists():
                return "skip", "Docker directory not found", {"docker_dir": str(docker_dir)}
            
            dockerfiles = list(docker_dir.glob("*.Dockerfile"))
            if not dockerfiles:
                return "skip", "No Dockerfiles found", {"docker_dir": str(docker_dir)}
            
            details = {
                "docker_dir": str(docker_dir),
                "dockerfiles_found": [f.name for f in dockerfiles],
                "images_checked": []
            }
            
            # Try to build a simple test image
            python_dockerfile = docker_dir / "python.Dockerfile"
            if python_dockerfile.exists():
                try:
                    # Build a test image
                    image, logs = client.images.build(
                        path=str(docker_dir),
                        dockerfile="python.Dockerfile",
                        tag="single-turn-test:python",
                        rm=True,
                        timeout=60
                    )
                    details["test_image_built"] = "single-turn-test:python"
                    
                    # Clean up test image
                    client.images.remove(image.id, force=True)
                    
                    return "pass", f"Sandbox images can be built ({len(dockerfiles)} Dockerfiles)", details
                    
                except docker.errors.BuildError as e:
                    return "warning", f"Image build failed: {str(e)}", details
                except Exception as e:
                    return "warning", f"Image test error: {str(e)}", details
            
            return "pass", f"Dockerfiles available ({len(dockerfiles)} found)", details
            
        except Exception as e:
            return "fail", f"Sandbox image test error: {str(e)}", {"error": str(e)}
    
    def test_dataset_integrity(self) -> Tuple[str, str, Dict[str, Any]]:
        """Test dataset file integrity and schema validation."""
        try:
            problems_file = self.current_dir / "problems.jsonl"
            if not problems_file.exists():
                return "fail", "problems.jsonl not found", {"file": str(problems_file)}
            
            # Load and validate problems
            problems = []
            line_count = 0
            
            with open(problems_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    if line.strip():
                        try:
                            problem = json.loads(line)
                            problems.append(problem)
                        except json.JSONDecodeError as e:
                            return "fail", f"JSON error at line {line_num}: {str(e)}", {
                                "file": str(problems_file),
                                "line": line_num,
                                "error": str(e)
                            }
            
            if not problems:
                return "fail", "No valid problems found in dataset", {
                    "file": str(problems_file),
                    "lines": line_count
                }
            
            # Basic schema validation
            required_fields = ["id", "title", "language", "scenario", "difficulty", "prompt"]
            schema_errors = []
            
            for i, problem in enumerate(problems[:5]):  # Check first 5 problems
                for field in required_fields:
                    if field not in problem:
                        schema_errors.append(f"Problem {i+1} missing field: {field}")
            
            details = {
                "file": str(problems_file),
                "total_problems": len(problems),
                "lines_processed": line_count,
                "schema_errors": schema_errors,
                "sample_scenarios": list(set(p.get("scenario", "unknown") for p in problems[:10])),
                "sample_languages": list(set(p.get("language", "unknown") for p in problems[:10]))
            }
            
            if schema_errors:
                return "warning", f"Dataset loaded with schema issues ({len(schema_errors)} errors)", details
            else:
                return "pass", f"Dataset valid ({len(problems)} problems)", details
                
        except Exception as e:
            return "fail", f"Dataset integrity test error: {str(e)}", {"error": str(e)}
    
    def test_configuration_files(self) -> Tuple[str, str, Dict[str, Any]]:
        """Test configuration file validity."""
        try:
            config_files = []
            details = {"files_checked": [], "errors": []}
            
            # Check YAML configuration files
            yaml_files = [
                "context_configs.json",
                "single_turn_scenarios_suite.yaml"
            ]
            
            for yaml_file in yaml_files:
                file_path = self.current_dir / yaml_file
                if file_path.exists():
                    try:
                        if yaml_file.endswith('.json'):
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                        else:
                            with open(file_path, 'r') as f:
                                data = yaml.safe_load(f)
                        
                        config_files.append(yaml_file)
                        details["files_checked"].append({
                            "file": yaml_file,
                            "status": "valid",
                            "keys": list(data.keys()) if isinstance(data, dict) else "non-dict"
                        })
                        
                    except (yaml.YAMLError, json.JSONDecodeError) as e:
                        details["errors"].append(f"{yaml_file}: {str(e)}")
                else:
                    details["errors"].append(f"{yaml_file}: File not found")
            
            # Check model configuration directory
            model_configs_dir = self.current_dir / "model_configs"
            if model_configs_dir.exists():
                model_configs = list(model_configs_dir.glob("*.yaml"))
                for config_file in model_configs:
                    try:
                        with open(config_file, 'r') as f:
                            yaml.safe_load(f)
                        details["files_checked"].append({
                            "file": f"model_configs/{config_file.name}",
                            "status": "valid"
                        })
                    except yaml.YAMLError as e:
                        details["errors"].append(f"model_configs/{config_file.name}: {str(e)}")
            
            if details["errors"]:
                return "warning", f"Configuration files have issues ({len(details['errors'])} errors)", details
            else:
                return "pass", f"Configuration files valid ({len(details['files_checked'])} checked)", details
                
        except Exception as e:
            return "fail", f"Configuration test error: {str(e)}", {"error": str(e)}
    
    def test_task_registration(self) -> Tuple[str, str, Dict[str, Any]]:
        """Test task registration with lm-eval."""
        try:
            # Try to import the task
            sys.path.insert(0, str(self.current_dir))
            
            try:
                from single_turn_scenarios import SingleTurnScenariosTask
                task_imported = True
            except ImportError as e:
                task_imported = False
                import_error = str(e)
            
            # Try to get task through TaskManager
            try:
                task_manager = TaskManager()
                available_tasks = task_manager.all_tasks
                
                single_turn_tasks = [task for task in available_tasks 
                                   if "single_turn" in task.lower()]
                
                details = {
                    "task_imported": task_imported,
                    "total_tasks": len(available_tasks),
                    "single_turn_tasks": single_turn_tasks[:10],  # First 10
                    "task_manager_working": True
                }
                
                if not task_imported:
                    details["import_error"] = import_error
                
                if single_turn_tasks:
                    return "pass", f"Task registration working ({len(single_turn_tasks)} single_turn tasks found)", details
                elif task_imported:
                    return "warning", "Task imported but not registered with TaskManager", details
                else:
                    return "fail", f"Task import failed: {import_error}", details
                    
            except Exception as e:
                details = {
                    "task_imported": task_imported,
                    "task_manager_error": str(e)
                }
                if not task_imported:
                    details["import_error"] = import_error
                
                return "fail", f"TaskManager error: {str(e)}", details
                
        except Exception as e:
            return "fail", f"Task registration test error: {str(e)}", {"error": str(e)}
    
    def test_sample_evaluation(self) -> Tuple[str, str, Dict[str, Any]]:
        """Test a minimal evaluation run."""
        try:
            # This is a basic test - we'll create a minimal mock evaluation
            # since running a full evaluation requires API keys and model access
            
            details = {
                "test_type": "mock_evaluation",
                "reason": "Full evaluation requires API keys and model access"
            }
            
            # Check if we can at least load the task configuration
            try:
                from utils import load_dataset, process_docs
                
                # Try to load a small sample of the dataset
                sample_data = load_dataset({"limit": 1})
                if sample_data and len(sample_data) > 0:
                    sample_doc = sample_data[0]
                    processed_doc = process_docs(sample_doc)
                    
                    details.update({
                        "dataset_loaded": True,
                        "sample_doc_keys": list(sample_doc.keys()),
                        "processed_doc_keys": list(processed_doc.keys()) if processed_doc else None
                    })
                    
                    return "pass", "Sample evaluation components working", details
                else:
                    return "warning", "Dataset loaded but empty", details
                    
            except ImportError as e:
                return "skip", f"Evaluation utilities not available: {str(e)}", details
            except Exception as e:
                return "warning", f"Sample evaluation test issue: {str(e)}", details
                
        except Exception as e:
            return "fail", f"Sample evaluation test error: {str(e)}", {"error": str(e)}
    
    def test_metrics_calculation(self) -> Tuple[str, str, Dict[str, Any]]:
        """Test metrics calculation functions."""
        try:
            from metrics import (
                exact_match, bleu_score, edit_distance_score,
                syntax_validity, pass_at_k
            )
            
            # Test basic metrics with sample data
            predictions = ["def hello(): return 'world'"]
            references = ["def hello(): return 'world'"]
            
            test_results = {}
            
            # Test exact match
            em_score = exact_match(predictions, references)
            test_results["exact_match"] = em_score
            
            # Test BLEU score
            try:
                bleu = bleu_score(predictions, references)
                test_results["bleu_score"] = bleu
            except Exception as e:
                test_results["bleu_error"] = str(e)
            
            # Test edit distance
            try:
                edit_dist = edit_distance_score(predictions, references)
                test_results["edit_distance"] = edit_dist
            except Exception as e:
                test_results["edit_distance_error"] = str(e)
            
            # Test syntax validity
            try:
                syntax_valid = syntax_validity(predictions[0], "python")
                test_results["syntax_validity"] = syntax_valid
            except Exception as e:
                test_results["syntax_validity_error"] = str(e)
            
            # Test pass@k
            try:
                pass_k = pass_at_k(predictions, ["assert hello() == 'world'"], k=1)
                test_results["pass_at_k"] = pass_k
            except Exception as e:
                test_results["pass_at_k_error"] = str(e)
            
            errors = [k for k in test_results.keys() if k.endswith("_error")]
            
            if errors:
                return "warning", f"Metrics partially working ({len(errors)} errors)", test_results
            else:
                return "pass", f"Metrics calculation working ({len(test_results)} metrics tested)", test_results
                
        except ImportError as e:
            return "skip", f"Metrics module not available: {str(e)}", {"error": str(e)}
        except Exception as e:
            return "fail", f"Metrics test error: {str(e)}", {"error": str(e)}
    
    def test_api_key_configuration(self) -> Tuple[str, str, Dict[str, Any]]:
        """Test API key configuration."""
        try:
            env_file = self.current_dir / ".env"
            env_template = self.current_dir / ".env.template"
            
            details = {
                "env_template_exists": env_template.exists(),
                "env_file_exists": env_file.exists(),
                "configured_keys": []
            }
            
            if not env_template.exists():
                return "warning", ".env.template not found", details
            
            if not env_file.exists():
                return "warning", ".env file not found (copy from .env.template)", details
            
            # Check configured API keys
            with open(env_file, 'r') as f:
                env_content = f.read()
            
            api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"]
            for key in api_keys:
                if key in env_content:
                    # Check if it's not the placeholder value
                    lines = env_content.split('\n')
                    for line in lines:
                        if line.startswith(f"{key}=") and "your_" not in line.lower():
                            details["configured_keys"].append(key)
                            break
            
            if details["configured_keys"]:
                return "pass", f"API keys configured ({len(details['configured_keys'])} keys)", details
            else:
                return "warning", "No API keys configured (evaluations will not work)", details
                
        except Exception as e:
            return "fail", f"API key test error: {str(e)}", {"error": str(e)}
    
    def generate_report(self) -> SmokeTestReport:
        """Generate comprehensive test report."""
        total_duration = time.time() - self.start_time
        
        # Count results by status
        status_counts = {"pass": 0, "fail": 0, "skip": 0, "warning": 0}
        for result in self.results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # Collect environment info
        env_info = {}
        for result in self.results:
            if result.name == "Environment Information" and result.details:
                env_info = result.details
                break
        
        report = SmokeTestReport(
            timestamp=datetime.now().isoformat(),
            environment=env_info,
            total_tests=len(self.results),
            passed=status_counts["pass"],
            failed=status_counts["fail"],
            skipped=status_counts["skip"],
            warnings=status_counts["warning"],
            duration=total_duration,
            results=self.results
        )
        
        return report
    
    def run_all_tests(self) -> SmokeTestReport:
        """Run all smoke tests."""
        print("🚀 Starting Single Turn Scenarios Smoke Tests")
        print("=" * 60)
        
        # Run all tests
        self.run_test("Environment Information", self.test_environment_info)
        self.run_test("Docker Connectivity", self.test_docker_connectivity)
        self.run_test("Sandbox Images", self.test_sandbox_images)
        self.run_test("Dataset Integrity", self.test_dataset_integrity)
        self.run_test("Configuration Files", self.test_configuration_files)
        self.run_test("Task Registration", self.test_task_registration)
        self.run_test("Sample Evaluation", self.test_sample_evaluation)
        self.run_test("Metrics Calculation", self.test_metrics_calculation)
        self.run_test("API Key Configuration", self.test_api_key_configuration)
        
        # Generate and return report
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print("🏁 Smoke Test Summary")
        print("=" * 60)
        
        print(f"✅ Passed: {report.passed}")
        print(f"❌ Failed: {report.failed}")
        print(f"⚠️  Warnings: {report.warnings}")
        print(f"⏭️  Skipped: {report.skipped}")
        print(f"⏱️  Duration: {report.duration:.2f}s")
        
        if report.failed > 0:
            print(f"\n❌ {report.failed} tests failed. Please check the issues above.")
            print("Run the setup script to resolve dependency issues:")
            print("  ./setupEvaluationEnvironment.sh --install-deps")
        elif report.warnings > 0:
            print(f"\n⚠️  {report.warnings} warnings found. System should work but may have limitations.")
        else:
            print("\n🎉 All tests passed! System is ready for evaluation.")
        
        return report

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Single Turn Scenarios Smoke Tests")
    parser.add_argument("--output", "-o", help="Output report file (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Run smoke tests
    runner = SmokeTestRunner()
    report = runner.run_all_tests()
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\n📄 Report saved to: {args.output}")
    
    # Exit with appropriate code
    if report.failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()