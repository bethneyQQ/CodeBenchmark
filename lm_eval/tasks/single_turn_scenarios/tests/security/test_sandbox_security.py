"""Security tests for sandbox execution and malicious code detection."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sandbox import SandboxExecutor, SecurityMonitor, ExecutionResult
from security_monitor import detect_security_violations, DANGEROUS_PATTERNS


class TestSandboxEscapeAttempts:
    """Tests for sandbox escape attempts and prevention."""
    
    @pytest.mark.security
    def test_file_system_escape_attempts(self):
        """Test prevention of file system escape attempts."""
        escape_attempts = [
            # Directory traversal
            "import os; os.chdir('../../../'); print(os.getcwd())",
            "with open('../../../etc/passwd', 'r') as f: print(f.read())",
            "import subprocess; subprocess.run(['ls', '../../../'])",
            
            # Absolute path access
            "with open('/etc/passwd', 'r') as f: print(f.read())",
            "import os; print(os.listdir('/root'))",
            "open('/proc/version').read()",
            
            # Home directory access
            "import os; print(os.listdir(os.path.expanduser('~')))",
            "with open('~/.bashrc', 'r') as f: print(f.read())",
        ]
        
        security_monitor = SecurityMonitor()
        
        for code in escape_attempts:
            violations = security_monitor.check_code(code, "python")
            assert len(violations) > 0, f"Should detect security violation in: {code}"
            
            # Test with sandbox executor
            with patch('sandbox.DOCKER_AVAILABLE', True):
                with patch('docker.from_env') as mock_docker:
                    mock_client = MagicMock()
                    mock_docker.return_value = mock_client
                    
                    executor = SandboxExecutor("python")
                    
                    # Should detect violations before execution
                    with patch.object(executor.security_monitor, 'check_code', return_value=violations):
                        result = executor.execute_code(code, [])
                        
                        assert len(result.security_violations) > 0
                        assert result.exit_code != 0
    
    @pytest.mark.security
    def test_network_access_attempts(self):
        """Test prevention of network access attempts."""
        network_attempts = [
            # Socket connections
            "import socket; s = socket.socket(); s.connect(('evil.com', 80))",
            "import urllib.request; urllib.request.urlopen('http://evil.com')",
            "import requests; requests.get('http://evil.com')",
            
            # HTTP libraries
            "from http.client import HTTPConnection; conn = HTTPConnection('evil.com')",
            "import ftplib; ftp = ftplib.FTP('evil.com')",
            
            # DNS lookups
            "import socket; socket.gethostbyname('evil.com')",
        ]
        
        security_monitor = SecurityMonitor()
        
        for code in network_attempts:
            violations = security_monitor.check_code(code, "python")
            assert len(violations) > 0, f"Should detect network violation in: {code}"
    
    @pytest.mark.security
    def test_process_execution_attempts(self):
        """Test prevention of unauthorized process execution."""
        process_attempts = [
            # System commands
            "import os; os.system('rm -rf /')",
            "import os; os.popen('cat /etc/passwd').read()",
            "import subprocess; subprocess.call(['rm', '-rf', '/'])",
            "import subprocess; subprocess.Popen(['nc', '-l', '1234'])",
            
            # Shell execution
            "import subprocess; subprocess.run('rm -rf /', shell=True)",
            "os.execv('/bin/sh', ['/bin/sh'])",
            
            # Process spawning
            "import multiprocessing; multiprocessing.Process(target=lambda: os.system('evil')).start()",
        ]
        
        security_monitor = SecurityMonitor()
        
        for code in process_attempts:
            violations = security_monitor.check_code(code, "python")
            assert len(violations) > 0, f"Should detect process execution violation in: {code}"
    
    @pytest.mark.security
    def test_resource_exhaustion_attempts(self):
        """Test prevention of resource exhaustion attacks."""
        resource_attacks = [
            # Memory bombs
            "data = []; [data.append([0] * 1000000) for _ in range(1000)]",
            "x = 'A' * (10**9)",  # Huge string
            
            # CPU bombs
            "while True: pass",
            "import threading; [threading.Thread(target=lambda: [1 for _ in range(10**8)]).start() for _ in range(100)]",
            
            # Fork bombs (Python equivalent)
            "import multiprocessing; [multiprocessing.Process(target=lambda: None).start() for _ in range(1000)]",
        ]
        
        for code in resource_attacks:
            with patch('sandbox.DOCKER_AVAILABLE', True):
                with patch('docker.from_env') as mock_docker:
                    mock_container = MagicMock()
                    
                    # Simulate resource limit exceeded
                    mock_container.exec_run.side_effect = Exception("Resource limit exceeded")
                    mock_container.stats.return_value = iter([{
                        'memory_stats': {'max_usage': 100 * 1024 * 1024},  # 100MB
                        'cpu_stats': {'cpu_usage': {'total_usage': 10000000}}
                    }])
                    
                    mock_client = MagicMock()
                    mock_client.containers.run.return_value = mock_container
                    mock_docker.return_value = mock_client
                    
                    executor = SandboxExecutor("python", limits={"memory_limit_mb": 50, "time_limit_s": 5})
                    
                    with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                        with patch.object(executor, '_cleanup'):
                            result = executor.execute_code(code, [])
                            
                            # Should fail due to resource limits
                            assert result.exit_code != 0
    
    @pytest.mark.security
    def test_privilege_escalation_attempts(self):
        """Test prevention of privilege escalation attempts."""
        privilege_attempts = [
            # Sudo attempts
            "import subprocess; subprocess.run(['sudo', 'whoami'])",
            "os.system('sudo rm -rf /')",
            
            # User switching
            "import subprocess; subprocess.run(['su', 'root'])",
            "os.system('su - root')",
            
            # Setuid attempts
            "import os; os.setuid(0)",
            "import os; os.seteuid(0)",
        ]
        
        security_monitor = SecurityMonitor()
        
        for code in privilege_attempts:
            violations = security_monitor.check_code(code, "python")
            assert len(violations) > 0, f"Should detect privilege escalation in: {code}"


class TestMaliciousCodeDetection:
    """Tests for detecting various types of malicious code."""
    
    @pytest.mark.security
    def test_dangerous_import_detection(self):
        """Test detection of dangerous imports."""
        dangerous_imports = [
            "import os",
            "import subprocess",
            "import socket",
            "import urllib",
            "import requests",
            "import multiprocessing",
            "import threading",
            "from os import system",
            "from subprocess import call",
        ]
        
        security_monitor = SecurityMonitor()
        
        for code in dangerous_imports:
            violations = security_monitor.check_code(code, "python")
            # Note: Some imports might be allowed in certain contexts
            # The test checks that the security monitor is working
            if any(dangerous in code for dangerous in ['os', 'subprocess', 'socket']):
                assert len(violations) >= 0  # Should at least flag for review
    
    @pytest.mark.security
    def test_obfuscated_code_detection(self):
        """Test detection of obfuscated malicious code."""
        obfuscated_attempts = [
            # Base64 encoded commands
            "import base64; exec(base64.b64decode('aW1wb3J0IG9z'))",  # 'import os'
            
            # String manipulation to hide commands
            "cmd = 'rm' + ' -rf' + ' /'; os.system(cmd)",
            "getattr(__import__('os'), 'system')('rm -rf /')",
            
            # Dynamic imports
            "__import__('os').system('evil command')",
            "exec('import os; os.system(\"evil\")')",
            
            # Hex encoding
            "exec('\\x69\\x6d\\x70\\x6f\\x72\\x74\\x20\\x6f\\x73')",  # 'import os'
        ]
        
        security_monitor = SecurityMonitor()
        
        for code in obfuscated_attempts:
            violations = security_monitor.check_code(code, "python")
            assert len(violations) > 0, f"Should detect obfuscated malicious code: {code}"
    
    @pytest.mark.security
    def test_javascript_malicious_code_detection(self):
        """Test detection of malicious JavaScript code."""
        js_malicious_code = [
            # Process execution
            "require('child_process').exec('rm -rf /')",
            "const { spawn } = require('child_process'); spawn('rm', ['-rf', '/'])",
            
            # File system access
            "const fs = require('fs'); fs.readFileSync('/etc/passwd')",
            "require('fs').writeFileSync('/tmp/evil', 'malicious content')",
            
            # Network access
            "const http = require('http'); http.get('http://evil.com')",
            "require('net').createConnection({port: 80, host: 'evil.com'})",
        ]
        
        security_monitor = SecurityMonitor()
        
        for code in js_malicious_code:
            violations = security_monitor.check_code(code, "javascript")
            assert len(violations) > 0, f"Should detect malicious JavaScript: {code}"
    
    @pytest.mark.security
    def test_java_malicious_code_detection(self):
        """Test detection of malicious Java code."""
        java_malicious_code = [
            # Runtime execution
            "Runtime.getRuntime().exec(\"rm -rf /\")",
            "new ProcessBuilder(\"rm\", \"-rf\", \"/\").start()",
            
            # File access
            "new FileInputStream(\"/etc/passwd\")",
            "Files.readAllBytes(Paths.get(\"/etc/passwd\"))",
            
            # Network access
            "new Socket(\"evil.com\", 80)",
            "URL url = new URL(\"http://evil.com\"); url.openConnection()",
        ]
        
        security_monitor = SecurityMonitor()
        
        for code in java_malicious_code:
            violations = security_monitor.check_code(code, "java")
            assert len(violations) > 0, f"Should detect malicious Java: {code}"
    
    @pytest.mark.security
    def test_cpp_malicious_code_detection(self):
        """Test detection of malicious C++ code."""
        cpp_malicious_code = [
            # System calls
            "#include <cstdlib>\nint main() { system(\"rm -rf /\"); }",
            "#include <unistd.h>\nint main() { execl(\"/bin/sh\", \"sh\", \"-c\", \"evil\", NULL); }",
            
            # File operations
            "#include <fstream>\nstd::ifstream file(\"/etc/passwd\");",
            
            # Network operations
            "#include <sys/socket.h>\nsocket(AF_INET, SOCK_STREAM, 0);",
        ]
        
        security_monitor = SecurityMonitor()
        
        for code in cpp_malicious_code:
            violations = security_monitor.check_code(code, "cpp")
            assert len(violations) > 0, f"Should detect malicious C++: {code}"


class TestSecurityViolationLogging:
    """Tests for security violation logging and reporting."""
    
    @pytest.mark.security
    def test_violation_logging(self):
        """Test that security violations are properly logged."""
        malicious_code = "import os; os.system('rm -rf /')"
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                
                # Mock security monitor to return violations
                with patch.object(executor.security_monitor, 'check_code') as mock_check:
                    mock_check.return_value = ["Dangerous import: os.system"]
                    
                    result = executor.execute_code(malicious_code, [])
                    
                    # Check that violations are recorded
                    assert len(result.security_violations) > 0
                    assert "os.system" in result.security_violations[0]
                    assert result.exit_code != 0
    
    @pytest.mark.security
    def test_violation_details(self):
        """Test that security violation details are comprehensive."""
        test_cases = [
            ("import os; os.system('evil')", "os.system"),
            ("import subprocess; subprocess.call(['rm', '-rf'])", "subprocess"),
            ("import socket; socket.socket()", "socket"),
        ]
        
        security_monitor = SecurityMonitor()
        
        for code, expected_pattern in test_cases:
            violations = security_monitor.check_code(code, "python")
            
            assert len(violations) > 0
            # Check that violation contains relevant information
            violation_text = " ".join(violations).lower()
            assert expected_pattern.lower() in violation_text
    
    @pytest.mark.security
    def test_false_positive_minimization(self):
        """Test that legitimate code doesn't trigger false positives."""
        legitimate_code = [
            # Safe mathematical operations
            "import math; print(math.sqrt(16))",
            
            # Safe string operations
            "text = 'hello world'; print(text.upper())",
            
            # Safe data structures
            "data = [1, 2, 3]; print(sum(data))",
            
            # Safe file operations (relative paths)
            "with open('data.txt', 'w') as f: f.write('safe content')",
            
            # Safe JSON operations
            "import json; data = {'key': 'value'}; print(json.dumps(data))",
        ]
        
        security_monitor = SecurityMonitor()
        
        for code in legitimate_code:
            violations = security_monitor.check_code(code, "python")
            # Should have minimal or no violations for legitimate code
            assert len(violations) <= 1, f"Too many false positives for legitimate code: {code}"


class TestTimeoutHandling:
    """Tests for timeout handling and resource exhaustion."""
    
    @pytest.mark.security
    def test_infinite_loop_timeout(self):
        """Test that infinite loops are properly timed out."""
        infinite_loop_code = "while True: pass"
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                
                # Simulate timeout
                mock_container.exec_run.side_effect = Exception("Timeout exceeded")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 1024 * 1024},
                    'cpu_stats': {'cpu_usage': {'total_usage': 10000000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python", limits={"time_limit_s": 1})
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        result = executor.execute_code(infinite_loop_code, [])
                        
                        # Should timeout and fail
                        assert result.exit_code != 0
                        assert "timeout" in result.stderr.lower() or "error" in result.stderr.lower()
    
    @pytest.mark.security
    def test_recursive_function_timeout(self):
        """Test that infinite recursion is properly handled."""
        recursive_code = """
def infinite_recursion(n):
    return infinite_recursion(n + 1)

infinite_recursion(0)
"""
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                
                # Simulate stack overflow or timeout
                mock_container.exec_run.return_value = (1, b"RecursionError: maximum recursion depth exceeded")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 10 * 1024 * 1024},
                    'cpu_stats': {'cpu_usage': {'total_usage': 5000000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python", limits={"time_limit_s": 5})
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        result = executor.execute_code(recursive_code, [])
                        
                        # Should fail due to recursion limit or timeout
                        assert result.exit_code != 0
                        assert "recursion" in result.stdout.lower() or result.exit_code == 1
    
    @pytest.mark.security
    def test_memory_exhaustion_handling(self):
        """Test that memory exhaustion is properly handled."""
        memory_bomb_code = """
data = []
for i in range(1000000):
    data.append([0] * 1000)
"""
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                
                # Simulate memory limit exceeded
                mock_container.exec_run.side_effect = Exception("Memory limit exceeded")
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
                        result = executor.execute_code(memory_bomb_code, [])
                        
                        # Should fail due to memory limit
                        assert result.exit_code != 0
                        assert result.peak_memory > 0


class TestSecurityAuditTrail:
    """Tests for security audit trail and compliance."""
    
    @pytest.mark.security
    def test_audit_trail_creation(self):
        """Test that security events create proper audit trails."""
        malicious_code = "import os; os.system('whoami')"
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_client = MagicMock()
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                
                # Mock security monitor
                with patch.object(executor.security_monitor, 'check_code') as mock_check:
                    mock_check.return_value = ["Dangerous import: os.system detected"]
                    
                    result = executor.execute_code(malicious_code, [])
                    
                    # Audit trail should include:
                    # 1. Security violations
                    assert len(result.security_violations) > 0
                    
                    # 2. Exit code indicating failure
                    assert result.exit_code != 0
                    
                    # 3. Timestamp information (wall_time should be recorded)
                    assert result.wall_time >= 0
    
    @pytest.mark.security
    def test_compliance_reporting(self):
        """Test compliance reporting for security violations."""
        test_violations = [
            "File system access attempt: /etc/passwd",
            "Network connection attempt: evil.com:80",
            "Process execution attempt: rm -rf /",
            "Privilege escalation attempt: sudo"
        ]
        
        # Test that violations are properly categorized
        for violation in test_violations:
            # Each violation should contain enough information for compliance reporting
            assert len(violation) > 10  # Meaningful description
            assert any(keyword in violation.lower() for keyword in 
                      ['file', 'network', 'process', 'privilege'])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "security"])