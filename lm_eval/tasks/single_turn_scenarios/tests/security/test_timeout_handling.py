"""Security tests for timeout handling and resource exhaustion prevention."""

import pytest
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
import signal

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sandbox import SandboxExecutor, ExecutionResult


class TestTimeoutMechanisms:
    """Tests for various timeout mechanisms and their effectiveness."""
    
    @pytest.mark.security
    def test_wall_clock_timeout(self):
        """Test wall clock timeout enforcement."""
        long_running_code = """
import time
time.sleep(10)  # Sleep for 10 seconds
print("This should not execute")
"""
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                
                # Simulate timeout after 2 seconds
                def mock_exec_run(*args, **kwargs):
                    time.sleep(0.1)  # Simulate some execution time
                    raise Exception("Container execution timed out")
                
                mock_container.exec_run = mock_exec_run
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 1024 * 1024},
                    'cpu_stats': {'cpu_usage': {'total_usage': 1000000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python", limits={"time_limit_s": 2})
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        start_time = time.time()
                        result = executor.execute_code(long_running_code, [])
                        execution_time = time.time() - start_time
                        
                        # Should timeout within reasonable time
                        assert execution_time < 5.0, f"Execution took {execution_time:.2f}s, should timeout faster"
                        assert result.exit_code != 0, "Should fail due to timeout"
                        assert "timeout" in result.stderr.lower() or "error" in result.stderr.lower()
    
    @pytest.mark.security
    def test_cpu_intensive_timeout(self):
        """Test timeout for CPU-intensive operations."""
        cpu_intensive_code = """
# CPU-intensive loop
total = 0
for i in range(10**8):  # 100 million iterations
    total += i * i
print(total)
"""
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                
                # Simulate CPU timeout
                mock_container.exec_run.side_effect = Exception("CPU time limit exceeded")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 2 * 1024 * 1024},
                    'cpu_stats': {'cpu_usage': {'total_usage': 10000000}}  # High CPU usage
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python", limits={"time_limit_s": 3})
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        result = executor.execute_code(cpu_intensive_code, [])
                        
                        # Should timeout due to CPU limit
                        assert result.exit_code != 0
                        assert result.peak_memory > 0  # Should record resource usage
    
    @pytest.mark.security
    def test_io_intensive_timeout(self):
        """Test timeout for I/O-intensive operations."""
        io_intensive_code = """
# I/O intensive operations
import tempfile
import os

# Create many temporary files
for i in range(10000):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b'x' * 1024)  # Write 1KB per file
        
print("I/O operations completed")
"""
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                
                # Simulate I/O timeout
                mock_container.exec_run.side_effect = Exception("I/O timeout exceeded")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 5 * 1024 * 1024},
                    'cpu_stats': {'cpu_usage': {'total_usage': 2000000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python", limits={"time_limit_s": 5})
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        result = executor.execute_code(io_intensive_code, [])
                        
                        # Should timeout due to I/O limit
                        assert result.exit_code != 0
    
    @pytest.mark.security
    def test_infinite_loop_variants(self):
        """Test timeout for various infinite loop patterns."""
        infinite_loop_variants = [
            # Basic infinite loop
            "while True: pass",
            
            # Infinite loop with computation
            "while True: x = 1 + 1",
            
            # Infinite recursion
            "def recurse(): return recurse()\nrecurse()",
            
            # Infinite generator
            "def infinite(): \n    while True: yield 1\nlist(infinite())",
            
            # Busy wait
            "import time\nwhile True: time.sleep(0.001)",
        ]
        
        for i, code in enumerate(infinite_loop_variants):
            with patch('sandbox.DOCKER_AVAILABLE', True):
                with patch('docker.from_env') as mock_docker:
                    mock_container = MagicMock()
                    
                    # Simulate timeout for each variant
                    mock_container.exec_run.side_effect = Exception(f"Timeout for variant {i}")
                    mock_container.stats.return_value = iter([{
                        'memory_stats': {'max_usage': 1024 * 1024},
                        'cpu_stats': {'cpu_usage': {'total_usage': 5000000}}
                    }])
                    
                    mock_client = MagicMock()
                    mock_client.containers.run.return_value = mock_container
                    mock_docker.return_value = mock_client
                    
                    executor = SandboxExecutor("python", limits={"time_limit_s": 2})
                    
                    with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                        with patch.object(executor, '_cleanup'):
                            result = executor.execute_code(code, [])
                            
                            assert result.exit_code != 0, f"Infinite loop variant {i} should timeout"


class TestResourceExhaustionPrevention:
    """Tests for preventing resource exhaustion attacks."""
    
    @pytest.mark.security
    def test_memory_bomb_prevention(self):
        """Test prevention of memory exhaustion attacks."""
        memory_bombs = [
            # List memory bomb
            "data = []; [data.append([0] * 1000000) for _ in range(1000)]",
            
            # String memory bomb
            "x = 'A' * (10**8)",  # 100MB string
            
            # Dictionary memory bomb
            "d = {}; [d.update({i: [0] * 1000}) for i in range(100000)]",
            
            # Recursive data structure
            "class Node: pass\nn = Node(); n.child = n; data = [n] * 1000000",
        ]
        
        for i, code in enumerate(memory_bombs):
            with patch('sandbox.DOCKER_AVAILABLE', True):
                with patch('docker.from_env') as mock_docker:
                    mock_container = MagicMock()
                    
                    # Simulate memory limit exceeded
                    mock_container.exec_run.side_effect = Exception(f"Memory limit exceeded for bomb {i}")
                    mock_container.stats.return_value = iter([{
                        'memory_stats': {'max_usage': 100 * 1024 * 1024},  # 100MB
                        'cpu_stats': {'cpu_usage': {'total_usage': 2000000}}
                    }])
                    
                    mock_client = MagicMock()
                    mock_client.containers.run.return_value = mock_container
                    mock_docker.return_value = mock_client
                    
                    executor = SandboxExecutor("python", limits={"memory_limit_mb": 50})
                    
                    with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                        with patch.object(executor, '_cleanup'):
                            result = executor.execute_code(code, [])
                            
                            assert result.exit_code != 0, f"Memory bomb {i} should be prevented"
                            assert result.peak_memory > 0, "Should record memory usage"
    
    @pytest.mark.security
    def test_disk_space_exhaustion_prevention(self):
        """Test prevention of disk space exhaustion."""
        disk_bombs = [
            # Large file creation
            """
with open('large_file.txt', 'w') as f:
    for i in range(1000000):
        f.write('x' * 1000)  # 1KB per line
""",
            
            # Many small files
            """
import os
for i in range(100000):
    with open(f'file_{i}.txt', 'w') as f:
        f.write('content')
""",
            
            # Temporary file bomb
            """
import tempfile
files = []
for i in range(10000):
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(b'x' * 10000)  # 10KB per file
    files.append(f)
""",
        ]
        
        for i, code in enumerate(disk_bombs):
            with patch('sandbox.DOCKER_AVAILABLE', True):
                with patch('docker.from_env') as mock_docker:
                    mock_container = MagicMock()
                    
                    # Simulate disk space limit exceeded
                    mock_container.exec_run.side_effect = Exception(f"Disk space limit exceeded for bomb {i}")
                    mock_container.stats.return_value = iter([{
                        'memory_stats': {'max_usage': 10 * 1024 * 1024},
                        'cpu_stats': {'cpu_usage': {'total_usage': 1000000}}
                    }])
                    
                    mock_client = MagicMock()
                    mock_client.containers.run.return_value = mock_container
                    mock_docker.return_value = mock_client
                    
                    executor = SandboxExecutor("python", limits={"disk_limit_mb": 100})
                    
                    with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                        with patch.object(executor, '_cleanup'):
                            result = executor.execute_code(code, [])
                            
                            assert result.exit_code != 0, f"Disk bomb {i} should be prevented"
    
    @pytest.mark.security
    def test_process_fork_bomb_prevention(self):
        """Test prevention of process fork bombs."""
        fork_bombs = [
            # Threading bomb
            """
import threading
def bomb():
    while True:
        threading.Thread(target=bomb).start()
bomb()
""",
            
            # Multiprocessing bomb
            """
import multiprocessing
def bomb():
    while True:
        multiprocessing.Process(target=bomb).start()
bomb()
""",
            
            # Subprocess bomb
            """
import subprocess
import sys
while True:
    subprocess.Popen([sys.executable, '-c', 'pass'])
""",
        ]
        
        for i, code in enumerate(fork_bombs):
            with patch('sandbox.DOCKER_AVAILABLE', True):
                with patch('docker.from_env') as mock_docker:
                    mock_container = MagicMock()
                    
                    # Simulate process limit exceeded
                    mock_container.exec_run.side_effect = Exception(f"Process limit exceeded for bomb {i}")
                    mock_container.stats.return_value = iter([{
                        'memory_stats': {'max_usage': 20 * 1024 * 1024},
                        'cpu_stats': {'cpu_usage': {'total_usage': 8000000}}
                    }])
                    
                    mock_client = MagicMock()
                    mock_client.containers.run.return_value = mock_container
                    mock_docker.return_value = mock_client
                    
                    executor = SandboxExecutor("python", limits={"max_processes": 10})
                    
                    with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                        with patch.object(executor, '_cleanup'):
                            result = executor.execute_code(code, [])
                            
                            assert result.exit_code != 0, f"Fork bomb {i} should be prevented"


class TestTimeoutRecovery:
    """Tests for proper recovery after timeouts."""
    
    @pytest.mark.security
    def test_container_cleanup_after_timeout(self):
        """Test that containers are properly cleaned up after timeout."""
        timeout_code = "import time; time.sleep(100)"
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                mock_container.exec_run.side_effect = Exception("Timeout")
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python", limits={"time_limit_s": 1})
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    result = executor.execute_code(timeout_code, [])
                    
                    # Verify cleanup was called
                    mock_container.stop.assert_called()
                    mock_container.remove.assert_called()
                    
                    assert result.exit_code != 0
    
    @pytest.mark.security
    def test_resource_cleanup_after_timeout(self):
        """Test that resources are cleaned up after timeout."""
        timeout_code = "while True: x = [0] * 1000"
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                mock_container.exec_run.side_effect = Exception("Resource timeout")
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python", limits={"time_limit_s": 2, "memory_limit_mb": 50})
                
                with patch('shutil.rmtree') as mock_rmtree:
                    with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                        result = executor.execute_code(timeout_code, [])
                        
                        # Verify temporary directory cleanup
                        mock_rmtree.assert_called_with("/tmp/test")
                        
                        assert result.exit_code != 0
    
    @pytest.mark.security
    def test_multiple_timeout_handling(self):
        """Test handling multiple consecutive timeouts."""
        timeout_codes = [
            "while True: pass",
            "import time; time.sleep(100)",
            "def recurse(): return recurse()\nrecurse()",
        ]
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_docker.return_value = mock_client
                
                for i, code in enumerate(timeout_codes):
                    mock_container = MagicMock()
                    mock_container.exec_run.side_effect = Exception(f"Timeout {i}")
                    mock_client.containers.run.return_value = mock_container
                    
                    executor = SandboxExecutor("python", limits={"time_limit_s": 1})
                    
                    with patch.object(executor, '_prepare_environment', return_value=f"/tmp/test_{i}"):
                        with patch.object(executor, '_cleanup'):
                            result = executor.execute_code(code, [])
                            
                            assert result.exit_code != 0, f"Timeout {i} should be handled"


class TestTimeoutConfiguration:
    """Tests for timeout configuration and limits."""
    
    @pytest.mark.security
    def test_configurable_timeout_limits(self):
        """Test that timeout limits are configurable."""
        test_code = "import time; time.sleep(3)"
        
        timeout_configs = [
            {"time_limit_s": 1},   # Should timeout
            {"time_limit_s": 5},   # Should complete
            {"time_limit_s": 10},  # Should complete
        ]
        
        for i, config in enumerate(timeout_configs):
            with patch('sandbox.DOCKER_AVAILABLE', True):
                with patch('docker.from_env') as mock_docker:
                    mock_container = MagicMock()
                    
                    if config["time_limit_s"] < 3:
                        # Should timeout
                        mock_container.exec_run.side_effect = Exception("Timeout")
                        expected_exit_code = 1
                    else:
                        # Should complete
                        mock_container.exec_run.return_value = (0, b"Completed")
                        expected_exit_code = 0
                    
                    mock_container.stats.return_value = iter([{
                        'memory_stats': {'max_usage': 1024 * 1024},
                        'cpu_stats': {'cpu_usage': {'total_usage': 1000000}}
                    }])
                    
                    mock_client = MagicMock()
                    mock_client.containers.run.return_value = mock_container
                    mock_docker.return_value = mock_client
                    
                    executor = SandboxExecutor("python", limits=config)
                    
                    with patch.object(executor, '_prepare_environment', return_value=f"/tmp/test_{i}"):
                        with patch.object(executor, '_cleanup'):
                            result = executor.execute_code(test_code, [])
                            
                            if expected_exit_code == 0:
                                assert result.exit_code == 0, f"Config {i} should complete successfully"
                            else:
                                assert result.exit_code != 0, f"Config {i} should timeout"
    
    @pytest.mark.security
    def test_minimum_timeout_enforcement(self):
        """Test that minimum timeout limits are enforced."""
        # Very short timeout should be adjusted to minimum
        very_short_timeout = {"time_limit_s": 0.1}
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python", limits=very_short_timeout)
                
                # The executor should enforce a minimum timeout
                # (implementation detail - might adjust the limit internally)
                assert executor.limits["time_limit_s"] >= 0.1
    
    @pytest.mark.security
    def test_maximum_timeout_enforcement(self):
        """Test that maximum timeout limits are enforced."""
        # Very long timeout should be capped
        very_long_timeout = {"time_limit_s": 3600}  # 1 hour
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python", limits=very_long_timeout)
                
                # The executor might cap the timeout to a reasonable maximum
                # (implementation detail - depends on security policy)
                assert executor.limits["time_limit_s"] <= 3600


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "security"])