"""Mock objects and fixtures for unit testing."""

from unittest.mock import MagicMock, Mock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import tempfile
import os
from pathlib import Path


@dataclass
class MockExecutionResult:
    """Mock execution result for testing."""
    stdout: str
    stderr: str
    exit_code: int
    wall_time: float
    peak_memory: int
    security_violations: List[str]


class MockSandboxExecutor:
    """Mock sandbox executor for testing."""
    
    def __init__(self, language: str = "python", limits: Dict = None):
        self.language = language
        self.limits = limits or {}
        self.execution_count = 0
        
    def execute_code(self, code: str, tests: List[Dict] = None) -> MockExecutionResult:
        """Mock code execution."""
        self.execution_count += 1
        
        # Simulate different execution scenarios based on code content
        if "syntax error" in code.lower():
            return MockExecutionResult(
                stdout="",
                stderr="SyntaxError: invalid syntax",
                exit_code=1,
                wall_time=0.1,
                peak_memory=10,
                security_violations=[]
            )
        
        if "import os" in code and "system" in code:
            return MockExecutionResult(
                stdout="",
                stderr="Security violation detected",
                exit_code=1,
                wall_time=0.1,
                peak_memory=10,
                security_violations=["Dangerous import: os.system"]
            )
        
        if "while True" in code:
            return MockExecutionResult(
                stdout="",
                stderr="Timeout exceeded",
                exit_code=124,
                wall_time=10.0,
                peak_memory=100,
                security_violations=[]
            )
        
        # Default successful execution
        return MockExecutionResult(
            stdout="Success",
            stderr="",
            exit_code=0,
            wall_time=0.5,
            peak_memory=50,
            security_violations=[]
        )
    
    def cleanup(self):
        """Mock cleanup."""
        pass


class MockDockerClient:
    """Mock Docker client for testing."""
    
    def __init__(self):
        self.containers = MockContainerManager()
        
    def from_env(self):
        return self


class MockContainerManager:
    """Mock Docker container manager."""
    
    def run(self, image: str, **kwargs) -> 'MockContainer':
        return MockContainer()


class MockContainer:
    """Mock Docker container."""
    
    def __init__(self):
        self.exec_results = []
        
    def exec_run(self, cmd: str, **kwargs) -> tuple:
        """Mock command execution in container."""
        if "pytest" in cmd:
            return (0, b"All tests passed")
        elif "python" in cmd:
            return (0, b"Success")
        else:
            return (0, b"Command executed")
    
    def stats(self, stream=True):
        """Mock container stats."""
        return iter([{
            'memory_stats': {'max_usage': 1024 * 1024},  # 1MB
            'cpu_stats': {'cpu_usage': {'total_usage': 1000000}}
        }])
    
    def stop(self):
        """Mock container stop."""
        pass
    
    def remove(self):
        """Mock container removal."""
        pass


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        
    def filter(self, function):
        """Mock dataset filtering."""
        filtered_data = [item for item in self.data if function(item)]
        return MockDataset(filtered_data)
    
    def map(self, function):
        """Mock dataset mapping."""
        mapped_data = [function(item) for item in self.data]
        return MockDataset(mapped_data)
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    @classmethod
    def from_list(cls, data: List[Dict]):
        return cls(data)


class MockConfigManager:
    """Mock configuration manager for testing."""
    
    def __init__(self):
        self.configs = {
            "test_model": {
                "model_name": "test-model",
                "endpoint_config": {"base_url": "https://api.test.com"},
                "generation_params": {"temperature": 0.0}
            },
            "universal": {
                "model_name": "universal",
                "endpoint_config": {"base_url": "https://api.universal.com"},
                "generation_params": {"temperature": 0.0}
            }
        }
    
    def load_config(self, model_name: str) -> Dict:
        """Mock config loading."""
        if model_name in self.configs:
            return self.configs[model_name]
        else:
            raise FileNotFoundError(f"Config not found: {model_name}")
    
    def get_config(self, model_name: str) -> Dict:
        """Mock config getting."""
        return self.load_config(model_name)
    
    def validate_config(self, config: Dict) -> bool:
        """Mock config validation."""
        required_fields = ["model_name", "endpoint_config", "generation_params"]
        return all(field in config for field in required_fields)


class MockMetricsCalculator:
    """Mock metrics calculator for testing."""
    
    def __init__(self):
        self.calculation_count = 0
        
    def calculate_all_metrics(self, prediction: str, reference: str, 
                            execution_result: MockExecutionResult = None) -> Dict[str, float]:
        """Mock comprehensive metrics calculation."""
        self.calculation_count += 1
        
        # Return mock metrics based on input
        if prediction == reference:
            return {
                "exact_match": 1.0,
                "bleu_score": 1.0,
                "codebleu_score": 1.0,
                "rouge_l_score": 1.0,
                "edit_distance_score": 1.0,
                "syntax_validity": 1.0,
                "pass_at_1": 1.0,
                "runtime_correctness": 1.0
            }
        else:
            return {
                "exact_match": 0.0,
                "bleu_score": 0.75,
                "codebleu_score": 0.80,
                "rouge_l_score": 0.70,
                "edit_distance_score": 0.65,
                "syntax_validity": 1.0,
                "pass_at_1": 0.8,
                "runtime_correctness": 0.9
            }


class MockFileSystem:
    """Mock file system for testing."""
    
    def __init__(self):
        self.files = {}
        self.directories = set()
        
    def create_file(self, path: str, content: str):
        """Create a mock file."""
        self.files[path] = content
        # Create parent directories
        parent = str(Path(path).parent)
        if parent != ".":
            self.directories.add(parent)
    
    def read_file(self, path: str) -> str:
        """Read a mock file."""
        if path in self.files:
            return self.files[path]
        else:
            raise FileNotFoundError(f"File not found: {path}")
    
    def exists(self, path: str) -> bool:
        """Check if mock file exists."""
        return path in self.files or path in self.directories
    
    def delete_file(self, path: str):
        """Delete a mock file."""
        if path in self.files:
            del self.files[path]


class MockTemporaryDirectory:
    """Mock temporary directory for testing."""
    
    def __init__(self):
        self.name = "/tmp/mock_temp_dir"
        self.cleanup_called = False
        
    def __enter__(self):
        return self.name
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_called = True


def create_sample_problem(problem_id: str = "test_001") -> Dict[str, Any]:
    """Create a sample problem for testing."""
    return {
        "id": problem_id,
        "title": "Test Problem",
        "language": "python",
        "scenario": "code_completion",
        "difficulty": "simple",
        "context_mode": "no_context",
        "prompt": "Write a function that adds two numbers",
        "reference": ["def add(a, b): return a + b"],
        "tests": [
            {
                "type": "unit",
                "file": f"test_{problem_id}.py",
                "cmd": f"python -m pytest test_{problem_id}.py"
            }
        ],
        "metadata": {
            "time_limit_s": 10,
            "memory_limit_mb": 100,
            "seed": 1234,
            "author": "test",
            "license": "MIT"
        }
    }


def create_sample_dataset(num_problems: int = 5) -> MockDataset:
    """Create a sample dataset for testing."""
    problems = []
    scenarios = ["code_completion", "bug_fix", "function_generation"]
    difficulties = ["simple", "intermediate", "complex"]
    languages = ["python", "javascript", "java"]
    
    for i in range(num_problems):
        problem = create_sample_problem(f"test_{i:03d}")
        problem["scenario"] = scenarios[i % len(scenarios)]
        problem["difficulty"] = difficulties[i % len(difficulties)]
        problem["language"] = languages[i % len(languages)]
        problems.append(problem)
    
    return MockDataset(problems)


def create_mock_context_configs() -> Dict[str, Dict[str, str]]:
    """Create mock context configurations."""
    return {
        "no_context": {
            "template": "{{prompt}}",
            "description": "Pure problem with no additional context"
        },
        "minimal_context": {
            "template": "{{prompt}}\n\nRequirements:\n- Follow best practices",
            "description": "Basic constraints and requirements"
        },
        "full_context": {
            "template": "Company Standards:\n{{company_standards}}\n\nProblem:\n{{prompt}}\n\nBest Practices:\n{{best_practices}}",
            "description": "Complete company standards and best practices"
        },
        "domain_context": {
            "template": "Domain: {{domain}}\n\nSpecialist Requirements:\n{{domain_requirements}}\n\nProblem:\n{{prompt}}",
            "description": "Domain-specific professional requirements"
        }
    }


def create_mock_model_configs() -> Dict[str, Dict[str, Any]]:
    """Create mock model configurations."""
    return {
        "claude_code": {
            "model_name": "claude-code-local",
            "endpoint_config": {
                "base_url": "https://api.anthropic.com",
                "timeout": 60,
                "rate_limit": 10
            },
            "generation_params": {
                "temperature": 0.0,
                "max_tokens": 2048,
                "top_p": 0.95
            }
        },
        "openai": {
            "model_name": "gpt-4",
            "endpoint_config": {
                "base_url": "https://api.openai.com",
                "timeout": 60,
                "rate_limit": 20
            },
            "generation_params": {
                "temperature": 0.0,
                "max_tokens": 2048
            }
        },
        "universal": {
            "model_name": "universal",
            "endpoint_config": {
                "base_url": "https://api.universal.com",
                "timeout": 30,
                "rate_limit": 5
            },
            "generation_params": {
                "temperature": 0.0,
                "max_tokens": 1024
            }
        }
    }


# Test fixtures that can be imported
SAMPLE_PROBLEMS = [create_sample_problem(f"test_{i:03d}") for i in range(10)]
SAMPLE_DATASET = create_sample_dataset(10)
MOCK_CONTEXT_CONFIGS = create_mock_context_configs()
MOCK_MODEL_CONFIGS = create_mock_model_configs()


class TestFixtures:
    """Collection of test fixtures and utilities."""
    
    @staticmethod
    def get_mock_execution_result(success: bool = True) -> MockExecutionResult:
        """Get a mock execution result."""
        if success:
            return MockExecutionResult(
                stdout="Success",
                stderr="",
                exit_code=0,
                wall_time=0.5,
                peak_memory=50,
                security_violations=[]
            )
        else:
            return MockExecutionResult(
                stdout="",
                stderr="Error occurred",
                exit_code=1,
                wall_time=0.1,
                peak_memory=10,
                security_violations=[]
            )
    
    @staticmethod
    def get_mock_sandbox_executor(language: str = "python") -> MockSandboxExecutor:
        """Get a mock sandbox executor."""
        return MockSandboxExecutor(language)
    
    @staticmethod
    def get_mock_dataset(size: int = 5) -> MockDataset:
        """Get a mock dataset."""
        return create_sample_dataset(size)
    
    @staticmethod
    def get_mock_config_manager() -> MockConfigManager:
        """Get a mock configuration manager."""
        return MockConfigManager()
    
    @staticmethod
    def get_mock_metrics_calculator() -> MockMetricsCalculator:
        """Get a mock metrics calculator."""
        return MockMetricsCalculator()


# Export commonly used mocks
__all__ = [
    'MockExecutionResult',
    'MockSandboxExecutor', 
    'MockDockerClient',
    'MockContainer',
    'MockDataset',
    'MockConfigManager',
    'MockMetricsCalculator',
    'MockFileSystem',
    'MockTemporaryDirectory',
    'create_sample_problem',
    'create_sample_dataset',
    'create_mock_context_configs',
    'create_mock_model_configs',
    'TestFixtures',
    'SAMPLE_PROBLEMS',
    'SAMPLE_DATASET',
    'MOCK_CONTEXT_CONFIGS',
    'MOCK_MODEL_CONFIGS'
]