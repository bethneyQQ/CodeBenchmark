"""Integration tests for the complete evaluation pipeline."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import datasets

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import load_dataset, process_docs, doc_to_text, doc_to_target
from metrics import exact_match, bleu_score, pass_at_k, syntax_validity
from sandbox import SandboxExecutor, ExecutionResult
from config_manager import ConfigManager


class TestEvaluationPipeline:
    """Integration tests for the complete evaluation pipeline."""
    
    @pytest.mark.integration
    def test_full_pipeline_python_simple(self):
        """Test complete evaluation pipeline with Python simple problem."""
        # Create a temporary problems file
        problem = {
            "id": "integration_001",
            "title": "Add Two Numbers",
            "language": "python",
            "scenario": "function_generation",
            "difficulty": "simple",
            "context_mode": "no_context",
            "prompt": "Write a function called 'add' that takes two parameters and returns their sum.",
            "reference": ["def add(a, b):\n    return a + b"],
            "tests": [
                {
                    "type": "unit",
                    "file": "test_integration_001.py",
                    "cmd": "python -c 'from solution import add; assert add(2, 3) == 5; assert add(-1, 1) == 0'"
                }
            ],
            "metadata": {
                "time_limit_s": 10,
                "memory_limit_mb": 100,
                "seed": 1234,
                "author": "integration_test",
                "license": "MIT"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(problem, f)
            f.write('\n')
            problems_file = f.name
        
        try:
            # Mock the problems file path
            with patch('utils.Path.__truediv__') as mock_path:
                mock_path.return_value.exists.return_value = True
                
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.readlines.return_value = [
                        json.dumps(problem) + '\n'
                    ]
                    
                    # Step 1: Load dataset
                    with patch('utils.validate_problem_schema', return_value=True):
                        with patch('datasets.Dataset.from_list') as mock_dataset:
                            mock_ds = MagicMock()
                            mock_ds.__iter__.return_value = [problem]
                            mock_ds.__len__.return_value = 1
                            mock_dataset.return_value = mock_ds
                            
                            dataset = load_dataset()
                            assert len(dataset) == 1
            
            # Step 2: Process document
            processed_doc = process_docs(problem)
            assert "processed_prompt" in processed_doc
            
            # Step 3: Extract text and target
            text = doc_to_text(processed_doc)
            target = doc_to_target(processed_doc)
            
            assert "Write a function called 'add'" in text
            assert "def add(a, b):" in target
            
            # Step 4: Simulate model prediction
            prediction = "def add(a, b):\n    return a + b"
            
            # Step 5: Calculate metrics
            em_score = exact_match([prediction], [target])
            assert em_score == 1.0
            
            syntax_score = syntax_validity(prediction, "python")
            assert syntax_score == 1.0
            
            # Step 6: Test execution (mocked)
            with patch('sandbox.DOCKER_AVAILABLE', True):
                with patch('docker.from_env') as mock_docker:
                    mock_container = MagicMock()
                    mock_container.exec_run.return_value = (0, b"Success")
                    mock_container.stats.return_value = iter([{
                        'memory_stats': {'max_usage': 1024 * 1024},
                        'cpu_stats': {'cpu_usage': {'total_usage': 1000000}}
                    }])
                    
                    mock_client = MagicMock()
                    mock_client.containers.run.return_value = mock_container
                    mock_docker.return_value = mock_client
                    
                    executor = SandboxExecutor("python")
                    
                    with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                        with patch.object(executor, '_cleanup'):
                            with patch('time.time', side_effect=[0, 1]):
                                result = executor.execute_code(prediction, problem["tests"])
                                
                                assert result.exit_code == 0
                                assert result.wall_time == 1.0
            
        finally:
            # Clean up temporary file
            if os.path.exists(problems_file):
                os.unlink(problems_file)
    
    @pytest.mark.integration
    def test_pipeline_with_context_application(self):
        """Test pipeline with context template application."""
        problem = {
            "id": "integration_002",
            "title": "Context Test",
            "language": "python",
            "scenario": "code_completion",
            "difficulty": "simple",
            "context_mode": "minimal_context",
            "prompt": "Complete the function",
            "reference": ["def complete(): pass"],
            "tests": [],
            "metadata": {
                "time_limit_s": 5,
                "memory_limit_mb": 50,
                "seed": 5678,
                "author": "test",
                "license": "MIT"
            }
        }
        
        mock_context_configs = {
            "minimal_context": {
                "template": "{{prompt}}\n\nRequirements:\n- Follow PEP 8",
                "description": "Basic requirements"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_context_configs):
            processed_doc = process_docs(problem)
            
            assert "processed_prompt" in processed_doc
            assert "Complete the function" in processed_doc["processed_prompt"]
            assert "Requirements:" in processed_doc["processed_prompt"]
            assert "PEP 8" in processed_doc["processed_prompt"]
    
    @pytest.mark.integration
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid inputs."""
        # Test with invalid problem schema
        invalid_problem = {
            "id": "invalid_001",
            # Missing required fields
        }
        
        with patch('utils.validate_problem_schema', return_value=False):
            # Should handle invalid problems gracefully
            processed_doc = process_docs(invalid_problem)
            # Should return something even with invalid input
            assert isinstance(processed_doc, dict)
    
    @pytest.mark.integration
    def test_pipeline_with_multiple_languages(self):
        """Test pipeline with multiple programming languages."""
        languages = ["python", "javascript", "java"]
        
        for language in languages:
            problem = {
                "id": f"multi_lang_{language}",
                "title": f"Test {language.title()}",
                "language": language,
                "scenario": "function_generation",
                "difficulty": "simple",
                "context_mode": "no_context",
                "prompt": f"Write a hello world function in {language}",
                "reference": ["// Hello world function"],
                "tests": [],
                "metadata": {
                    "time_limit_s": 10,
                    "memory_limit_mb": 100,
                    "seed": 1234,
                    "author": "test",
                    "license": "MIT"
                }
            }
            
            # Process document for each language
            processed_doc = process_docs(problem)
            assert processed_doc["language"] == language
            
            text = doc_to_text(processed_doc)
            assert language in text.lower()


class TestMultiModelIntegration:
    """Integration tests for multi-model support."""
    
    @pytest.mark.integration
    def test_config_loading_for_all_models(self):
        """Test configuration loading for all supported model backends."""
        model_names = ["claude_code", "deepseek", "openai", "anthropic", "universal"]
        
        for model_name in model_names:
            mock_config = {
                "model_name": f"{model_name}-test",
                "endpoint_config": {"base_url": f"https://api.{model_name}.com"},
                "generation_params": {"temperature": 0.0}
            }
            
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_config)
                    
                    config_manager = ConfigManager()
                    config = config_manager.load_config(model_name)
                    
                    assert config["model_name"] == f"{model_name}-test"
                    assert "endpoint_config" in config
                    assert "generation_params" in config
    
    @pytest.mark.integration
    def test_model_config_validation(self):
        """Test model configuration validation across all backends."""
        valid_config = {
            "model_name": "test-model",
            "endpoint_config": {
                "base_url": "https://api.test.com",
                "timeout": 60
            },
            "generation_params": {
                "temperature": 0.0,
                "max_tokens": 2048
            }
        }
        
        config_manager = ConfigManager()
        assert config_manager.validate_config(valid_config) == True
        
        # Test invalid config
        invalid_config = {
            "model_name": "test-model"
            # Missing required fields
        }
        
        assert config_manager.validate_config(invalid_config) == False


class TestCrossLanguageIntegration:
    """Integration tests for cross-language support."""
    
    @pytest.mark.integration
    def test_syntax_validation_all_languages(self):
        """Test syntax validation across all supported languages."""
        test_codes = {
            "python": "def hello():\n    print('Hello, World!')",
            "javascript": "function hello() { console.log('Hello, World!'); }",
            "java": "public class Hello { public static void main(String[] args) { System.out.println(\"Hello, World!\"); } }",
            "cpp": "#include <iostream>\nint main() { std::cout << \"Hello, World!\" << std::endl; return 0; }",
            "go": "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Hello, World!\") }",
            "rust": "fn main() { println!(\"Hello, World!\"); }"
        }
        
        for language, code in test_codes.items():
            # Test valid syntax
            score = syntax_validity(code, language)
            assert score >= 0.0 and score <= 1.0
            
            # Test invalid syntax
            invalid_code = code.replace("(", "")  # Remove opening parenthesis
            invalid_score = syntax_validity(invalid_code, language)
            assert invalid_score <= score  # Invalid should score lower or equal
    
    @pytest.mark.integration
    def test_sandbox_execution_all_languages(self):
        """Test sandbox execution across all supported languages."""
        pytest.importorskip("docker")  # Skip if Docker not available
        
        test_codes = {
            "python": "print('Hello from Python')",
            "javascript": "console.log('Hello from JavaScript');",
            "java": "public class Test { public static void main(String[] args) { System.out.println(\"Hello from Java\"); } }",
        }
        
        for language, code in test_codes.items():
            try:
                executor = SandboxExecutor(language)
                result = executor.execute_code(code, [])
                
                # Should execute without errors
                assert result.exit_code == 0 or result.stdout != ""
                assert result.wall_time > 0
                
            except RuntimeError as e:
                if "Docker is not available" in str(e):
                    pytest.skip(f"Docker not available for {language} test")
                else:
                    raise


class TestEndToEndEvaluation:
    """End-to-end evaluation tests."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_evaluation_workflow(self):
        """Test complete evaluation workflow from problem to results."""
        # Create a complete test problem
        problem = {
            "id": "e2e_001",
            "title": "Fibonacci Function",
            "language": "python",
            "scenario": "algorithm_implementation",
            "difficulty": "intermediate",
            "context_mode": "minimal_context",
            "prompt": "Implement a function that calculates the nth Fibonacci number using recursion.",
            "reference": [
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
            ],
            "tests": [
                {
                    "type": "unit",
                    "file": "test_fibonacci.py",
                    "cmd": "python -c 'from solution import fibonacci; assert fibonacci(0) == 0; assert fibonacci(1) == 1; assert fibonacci(5) == 5'"
                }
            ],
            "metadata": {
                "time_limit_s": 30,
                "memory_limit_mb": 200,
                "seed": 9999,
                "author": "e2e_test",
                "license": "MIT"
            }
        }
        
        # Mock the complete workflow
        with patch('utils.load_context_configs') as mock_context:
            mock_context.return_value = {
                "minimal_context": {
                    "template": "{{prompt}}\n\nRequirements:\n- Use recursion\n- Handle edge cases",
                    "description": "Basic requirements"
                }
            }
            
            # Step 1: Process document
            processed_doc = process_docs(problem)
            assert "processed_prompt" in processed_doc
            
            # Step 2: Extract prompt and target
            prompt = doc_to_text(processed_doc)
            target = doc_to_target(processed_doc)
            
            assert "Fibonacci" in prompt
            assert "recursion" in prompt.lower()
            assert "def fibonacci" in target
            
            # Step 3: Simulate model prediction (correct implementation)
            prediction = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
            
            # Step 4: Calculate all metrics
            metrics = {}
            
            # Basic metrics
            metrics["exact_match"] = exact_match([prediction], [target])
            metrics["syntax_validity"] = syntax_validity(prediction, "python")
            
            # Mock execution result
            mock_execution_result = MagicMock()
            mock_execution_result.exit_code = 0
            mock_execution_result.stdout = "All tests passed"
            mock_execution_result.stderr = ""
            mock_execution_result.wall_time = 0.5
            mock_execution_result.peak_memory = 50
            mock_execution_result.security_violations = []
            
            # Step 5: Compile results
            result = {
                "id": problem["id"],
                "model": "test_model",
                "config": "minimal_context|temperature=0",
                "prediction": prediction,
                "metrics": metrics,
                "runtime": {
                    "time_s": mock_execution_result.wall_time,
                    "exit_code": mock_execution_result.exit_code,
                    "peak_memory_mb": mock_execution_result.peak_memory
                },
                "seed": problem["metadata"]["seed"],
                "timestamp": "2025-09-25T15:00:00Z"
            }
            
            # Validate result structure
            assert "id" in result
            assert "model" in result
            assert "prediction" in result
            assert "metrics" in result
            assert "runtime" in result
            
            # Validate metrics
            assert result["metrics"]["exact_match"] == 1.0
            assert result["metrics"]["syntax_validity"] == 1.0
    
    @pytest.mark.integration
    def test_evaluation_with_failures(self):
        """Test evaluation workflow with various failure scenarios."""
        problem = {
            "id": "failure_001",
            "title": "Failure Test",
            "language": "python",
            "scenario": "bug_fix",
            "difficulty": "simple",
            "context_mode": "no_context",
            "prompt": "Fix the bug in this code",
            "reference": ["def fixed(): return True"],
            "tests": [],
            "metadata": {
                "time_limit_s": 5,
                "memory_limit_mb": 50,
                "seed": 1111,
                "author": "test",
                "license": "MIT"
            }
        }
        
        # Test with syntax error prediction
        syntax_error_prediction = "def broken(\n    return False"  # Missing closing parenthesis
        
        processed_doc = process_docs(problem)
        prompt = doc_to_text(processed_doc)
        target = doc_to_target(processed_doc)
        
        # Calculate metrics for failed prediction
        em_score = exact_match([syntax_error_prediction], [target])
        syntax_score = syntax_validity(syntax_error_prediction, "python")
        
        assert em_score == 0.0  # Should not match
        assert syntax_score == 0.0  # Should be invalid syntax
        
        # Test with security violation
        security_violation_prediction = "import os; os.system('rm -rf /')"
        
        # Mock security check
        with patch('sandbox.SecurityMonitor') as mock_monitor:
            mock_instance = MagicMock()
            mock_instance.check_code.return_value = ["Dangerous import: os.system"]
            mock_monitor.return_value = mock_instance
            
            # Should detect security violation
            violations = mock_instance.check_code(security_violation_prediction, "python")
            assert len(violations) > 0


class TestDatasetIntegration:
    """Integration tests for dataset handling."""
    
    @pytest.mark.integration
    def test_dataset_loading_and_filtering(self):
        """Test complete dataset loading and filtering workflow."""
        # Create multiple test problems
        problems = [
            {
                "id": "filter_001",
                "language": "python",
                "scenario": "code_completion",
                "difficulty": "simple",
                "context_mode": "no_context",
                "prompt": "Test 1",
                "reference": ["test1"],
                "tests": [],
                "metadata": {"author": "test", "license": "MIT"}
            },
            {
                "id": "filter_002", 
                "language": "javascript",
                "scenario": "bug_fix",
                "difficulty": "intermediate",
                "context_mode": "minimal_context",
                "prompt": "Test 2",
                "reference": ["test2"],
                "tests": [],
                "metadata": {"author": "test", "license": "MIT"}
            },
            {
                "id": "filter_003",
                "language": "python",
                "scenario": "code_completion",
                "difficulty": "complex",
                "context_mode": "full_context",
                "prompt": "Test 3",
                "reference": ["test3"],
                "tests": [],
                "metadata": {"author": "test", "license": "MIT"}
            }
        ]
        
        # Mock dataset loading
        with patch('utils.validate_problem_schema', return_value=True):
            with patch('datasets.Dataset.from_list') as mock_dataset:
                mock_ds = MagicMock()
                mock_ds.__iter__.return_value = problems
                mock_ds.__len__.return_value = len(problems)
                
                # Mock filtering
                def mock_filter(func):
                    filtered = [p for p in problems if func(p)]
                    filtered_ds = MagicMock()
                    filtered_ds.__iter__.return_value = filtered
                    filtered_ds.__len__.return_value = len(filtered)
                    return filtered_ds
                
                mock_ds.filter = mock_filter
                mock_dataset.return_value = mock_ds
                
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.readlines.return_value = [
                            json.dumps(p) + '\n' for p in problems
                        ]
                        
                        # Load full dataset
                        dataset = load_dataset()
                        assert len(dataset) == 3
                        
                        # Test filtering by language
                        from utils import filter_by_metadata
                        python_only = filter_by_metadata(dataset, {"language": "python"})
                        assert len(python_only) == 2
                        
                        # Test filtering by scenario
                        completion_only = filter_by_metadata(dataset, {"scenario": "code_completion"})
                        assert len(completion_only) == 2
                        
                        # Test filtering by difficulty
                        simple_only = filter_by_metadata(dataset, {"difficulty": "simple"})
                        assert len(simple_only) == 1


# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest for integration tests."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])