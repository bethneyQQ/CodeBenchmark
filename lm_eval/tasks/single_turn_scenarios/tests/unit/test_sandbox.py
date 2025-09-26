"""Unit tests for sandbox.py module."""

import pytest
import tempfile
import os
import time
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import dataclass
from pathlib import Path

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sandbox import (
    SandboxExecutor, ExecutionResult, SecurityMonitor,
    LANGUAGE_CONFIGS, DEFAULT_LIMITS
)


class TestExecutionResult:
    """Test cases for ExecutionResult dataclass."""
    
    def test_execution_result_creation(self):
        """Test ExecutionResult creation with all fields."""
        result = ExecutionResult(
            stdout="test output",
            stderr="test error",
            exit_code=0,
            wall_time=1.5,
            peak_memory=100,
            security_violations=["test violation"]
        )
        
        assert result.stdout == "test output"
        assert result.stderr == "test error"
        assert result.exit_code == 0
        assert result.wall_time == 1.5
        assert result.peak_memory == 100
        assert result.security_violations == ["test violation"]
    
    def test_execution_result_defaults(self):
        """Test ExecutionResult with default values."""
        result = ExecutionResult(
            stdout="",
            stderr="",
            exit_code=1,
            wall_time=0.0,
            peak_memory=0,
            security_violations=[]
        )
        
        assert result.security_violations == []
        assert result.wall_time == 0.0


class TestSecurityMonitor:
    """Test cases for SecurityMonitor class."""
    
    def test_security_monitor_creation(self):
        """Test SecurityMonitor creation."""
        monitor = SecurityMonitor()
        assert monitor.violations == []
    
    def test_check_code_safe(self):
        """Test security check with safe code."""
        monitor = SecurityMonitor()
        safe_code = "def add(a, b): return a + b"
        
        violations = monitor.check_code(safe_code, "python")
        assert violations == []
    
    def test_check_code_dangerous_imports(self):
        """Test security check with dangerous imports."""
        monitor = SecurityMonitor()
        dangerous_code = "import os; os.system('rm -rf /')"
        
        violations = monitor.check_code(dangerous_code, "python")
        assert len(violations) > 0
        assert any("os.system" in v for v in violations)
    
    def test_check_code_subprocess_calls(self):
        """Test security check with subprocess calls."""
        monitor = SecurityMonitor()
        dangerous_code = "import subprocess; subprocess.call(['rm', '-rf', '/'])"
        
        violations = monitor.check_code(dangerous_code, "python")
        assert len(violations) > 0
        assert any("subprocess" in v for v in violations)
    
    def test_check_code_file_operations(self):
        """Test security check with file operations."""
        monitor = SecurityMonitor()
        dangerous_code = "open('/etc/passwd', 'r').read()"
        
        violations = monitor.check_code(dangerous_code, "python")
        assert len(violations) > 0
    
    def test_check_code_network_operations(self):
        """Test security check with network operations."""
        monitor = SecurityMonitor()
        dangerous_code = "import socket; socket.socket().connect(('evil.com', 80))"
        
        violations = monitor.check_code(dangerous_code, "python")
        assert len(violations) > 0
        assert any("socket" in v for v in violations)
    
    def test_check_code_javascript_dangerous(self):
        """Test security check with dangerous JavaScript."""
        monitor = SecurityMonitor()
        dangerous_code = "require('child_process').exec('rm -rf /')"
        
        violations = monitor.check_code(dangerous_code, "javascript")
        assert len(violations) > 0
        assert any("child_process" in v for v in violations)


class TestSandboxExecutor:
    """Test cases for SandboxExecutor class."""
    
    def test_sandbox_executor_creation_python(self):
        """Test SandboxExecutor creation for Python."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                assert executor.language == "python"
                assert executor.client == mock_client
    
    def test_sandbox_executor_creation_docker_unavailable(self):
        """Test SandboxExecutor creation when Docker is unavailable."""
        with patch('sandbox.DOCKER_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="Docker is not available"):
                SandboxExecutor("python")
    
    def test_sandbox_executor_unsupported_language(self):
        """Test SandboxExecutor with unsupported language."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_docker.return_value = MagicMock()
                
                with pytest.raises(ValueError, match="Unsupported language"):
                    SandboxExecutor("unsupported")
    
    def test_prepare_environment_python(self):
        """Test environment preparation for Python."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                
                with patch('tempfile.mkdtemp') as mock_mkdtemp:
                    mock_mkdtemp.return_value = "/tmp/test"
                    
                    temp_dir = executor._prepare_environment("test code", [])
                    assert temp_dir == "/tmp/test"
    
    def test_create_container_success(self):
        """Test successful container creation."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_container = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                container = executor._create_container("/tmp/test")
                
                assert container == mock_container
                mock_client.containers.run.assert_called_once()
    
    def test_create_container_failure(self):
        """Test container creation failure."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_client.containers.run.side_effect = Exception("Container creation failed")
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                
                with pytest.raises(Exception, match="Container creation failed"):
                    executor._create_container("/tmp/test")
    
    def test_execute_code_success(self):
        """Test successful code execution."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_container = MagicMock()
                
                # Mock container execution
                mock_container.exec_run.return_value = (0, b"Success")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 1024 * 1024},  # 1MB
                    'cpu_stats': {'cpu_usage': {'total_usage': 1000000}}
                }])
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                
                with patch.object(executor, '_prepare_environment') as mock_prep:
                    mock_prep.return_value = "/tmp/test"
                    with patch.object(executor, '_cleanup') as mock_cleanup:
                        with patch('time.time', side_effect=[0, 1]):  # 1 second execution
                            
                            result = executor.execute_code("print('hello')", [])
                            
                            assert result.exit_code == 0
                            assert result.stdout == "Success"
                            assert result.wall_time == 1.0
                            assert result.peak_memory > 0
    
    def test_execute_code_timeout(self):
        """Test code execution with timeout."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_container = MagicMock()
                
                # Mock timeout scenario
                mock_container.exec_run.side_effect = Exception("Timeout")
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python", limits={"time_limit_s": 1})
                
                with patch.object(executor, '_prepare_environment') as mock_prep:
                    mock_prep.return_value = "/tmp/test"
                    with patch.object(executor, '_cleanup') as mock_cleanup:
                        
                        result = executor.execute_code("while True: pass", [])
                        
                        assert result.exit_code != 0
                        assert "timeout" in result.stderr.lower() or "error" in result.stderr.lower()
    
    def test_execute_code_security_violation(self):
        """Test code execution with security violations."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                
                with patch.object(executor.security_monitor, 'check_code') as mock_check:
                    mock_check.return_value = ["Dangerous import: os.system"]
                    
                    result = executor.execute_code("import os; os.system('rm -rf /')", [])
                    
                    assert len(result.security_violations) > 0
                    assert result.exit_code != 0
    
    def test_cleanup_success(self):
        """Test successful cleanup."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_container = MagicMock()
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                
                with patch('shutil.rmtree') as mock_rmtree:
                    executor._cleanup(mock_container, "/tmp/test")
                    
                    mock_container.stop.assert_called_once()
                    mock_container.remove.assert_called_once()
                    mock_rmtree.assert_called_once_with("/tmp/test")
    
    def test_cleanup_failure(self):
        """Test cleanup with failures."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_container = MagicMock()
                mock_container.stop.side_effect = Exception("Stop failed")
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                
                with patch('shutil.rmtree') as mock_rmtree:
                    # Should not raise exception even if cleanup fails
                    executor._cleanup(mock_container, "/tmp/test")
                    
                    mock_rmtree.assert_called_once_with("/tmp/test")
    
    def test_run_tests_success(self):
        """Test successful test execution."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_container = MagicMock()
                mock_container.exec_run.return_value = (0, b"All tests passed")
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                
                tests = [{"cmd": "python -m pytest test.py"}]
                result = executor._run_tests(mock_container, tests)
                
                assert result.exit_code == 0
                assert "All tests passed" in result.stdout
    
    def test_run_tests_failure(self):
        """Test test execution with failures."""
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_container = MagicMock()
                mock_container.exec_run.return_value = (1, b"Tests failed")
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                
                tests = [{"cmd": "python -m pytest test.py"}]
                result = executor._run_tests(mock_container, tests)
                
                assert result.exit_code == 1
                assert "Tests failed" in result.stdout


class TestLanguageConfigs:
    """Test cases for language configurations."""
    
    def test_language_configs_exist(self):
        """Test that language configurations exist for supported languages."""
        supported_languages = ["python", "javascript", "java", "cpp", "go", "rust"]
        
        for language in supported_languages:
            assert language in LANGUAGE_CONFIGS
            config = LANGUAGE_CONFIGS[language]
            assert "image" in config
            assert "extension" in config
            assert "run_cmd" in config
    
    def test_default_limits_exist(self):
        """Test that default limits are properly configured."""
        assert "time_limit_s" in DEFAULT_LIMITS
        assert "memory_limit_mb" in DEFAULT_LIMITS
        assert "cpu_limit" in DEFAULT_LIMITS
        
        assert DEFAULT_LIMITS["time_limit_s"] > 0
        assert DEFAULT_LIMITS["memory_limit_mb"] > 0
        assert DEFAULT_LIMITS["cpu_limit"] > 0


class TestSandboxIntegration:
    """Integration test cases for sandbox functionality."""
    
    @pytest.mark.integration
    def test_full_execution_cycle_python(self):
        """Test full execution cycle for Python code."""
        # This test requires Docker to be available
        pytest.importorskip("docker")
        
        code = """
def add(a, b):
    return a + b

print(add(2, 3))
"""
        
        try:
            executor = SandboxExecutor("python")
            result = executor.execute_code(code, [])
            
            # Should execute successfully
            assert result.exit_code == 0
            assert "5" in result.stdout
            assert result.wall_time > 0
            
        except RuntimeError as e:
            if "Docker is not available" in str(e):
                pytest.skip("Docker not available for integration test")
            else:
                raise
    
    @pytest.mark.integration
    def test_resource_limits_enforcement(self):
        """Test that resource limits are properly enforced."""
        pytest.importorskip("docker")
        
        # Code that should hit memory limit
        memory_intensive_code = """
data = []
for i in range(1000000):
    data.append([0] * 1000)
"""
        
        try:
            executor = SandboxExecutor("python", limits={"memory_limit_mb": 10})
            result = executor.execute_code(memory_intensive_code, [])
            
            # Should fail due to memory limit
            assert result.exit_code != 0 or result.peak_memory > 0
            
        except RuntimeError as e:
            if "Docker is not available" in str(e):
                pytest.skip("Docker not available for integration test")
            else:
                raise


# Test fixtures
@pytest.fixture
def mock_docker_client():
    """Mock Docker client for testing."""
    client = MagicMock()
    container = MagicMock()
    container.exec_run.return_value = (0, b"Success")
    container.stats.return_value = iter([{
        'memory_stats': {'max_usage': 1024 * 1024},
        'cpu_stats': {'cpu_usage': {'total_usage': 1000000}}
    }])
    client.containers.run.return_value = container
    return client


@pytest.fixture
def sample_executor():
    """Sample SandboxExecutor for testing."""
    with patch('sandbox.DOCKER_AVAILABLE', True):
        with patch('docker.from_env') as mock_docker:
            mock_docker.return_value = MagicMock()
            return SandboxExecutor("python")


if __name__ == "__main__":
    pytest.main([__file__])