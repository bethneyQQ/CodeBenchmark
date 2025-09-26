"""
Secure sandbox execution system for single_turn_scenarios evaluation.

This module provides isolated execution environments using Docker containers
with strict resource limits and security controls.
"""

import os
import time
import tempfile
import shutil
import subprocess
import json
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Handle Docker import gracefully
try:
    import docker
    import docker.types
    from docker.errors import ContainerError, ImageNotFound, APIError
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    # Create mock classes for when Docker is not available
    class docker:
        @staticmethod
        def from_env():
            raise RuntimeError("Docker is not available")
        
        class types:
            @staticmethod
            def Ulimit(name, soft, hard):
                return {'name': name, 'soft': soft, 'hard': hard}
    
    class ContainerError(Exception):
        pass
    
    class ImageNotFound(Exception):
        pass
    
    class APIError(Exception):
        pass


logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox environment."""
    stdout: str
    stderr: str
    exit_code: int
    wall_time: float
    peak_memory: int
    security_violations: List[str]
    success: bool = True
    error_message: Optional[str] = None


class SandboxExecutor:
    """
    Secure sandbox executor using Docker containers.
    
    Provides isolated execution environments for different programming languages
    with resource limits and security controls.
    """
    
    # Language-specific configuration
    LANGUAGE_CONFIGS = {
        'python': {
            'dockerfile': 'python.Dockerfile',
            'image_name': 'single_turn_scenarios:python',
            'file_extension': '.py',
            'run_command': ['python', '{filename}'],
            'test_command': ['python', '-m', 'pytest', '-v', '{test_file}']
        },
        'javascript': {
            'dockerfile': 'node.Dockerfile', 
            'image_name': 'single_turn_scenarios:node',
            'file_extension': '.js',
            'run_command': ['node', '{filename}'],
            'test_command': ['npm', 'test', '{test_file}']
        },
        'typescript': {
            'dockerfile': 'node.Dockerfile',
            'image_name': 'single_turn_scenarios:node', 
            'file_extension': '.ts',
            'run_command': ['npx', 'ts-node', '{filename}'],
            'test_command': ['npm', 'test', '{test_file}']
        },
        'java': {
            'dockerfile': 'java.Dockerfile',
            'image_name': 'single_turn_scenarios:java',
            'file_extension': '.java',
            'run_command': ['javac', '{filename}', '&&', 'java', '{classname}'],
            'test_command': ['mvn', 'test', '-Dtest={test_class}']
        },
        'cpp': {
            'dockerfile': 'gcc.Dockerfile',
            'image_name': 'single_turn_scenarios:gcc',
            'file_extension': '.cpp',
            'run_command': ['g++', '-o', '{output}', '{filename}', '&&', './{output}'],
            'test_command': ['g++', '-o', 'test', '{test_file}', '&&', './test']
        },
        'c': {
            'dockerfile': 'gcc.Dockerfile', 
            'image_name': 'single_turn_scenarios:gcc',
            'file_extension': '.c',
            'run_command': ['gcc', '-o', '{output}', '{filename}', '&&', './{output}'],
            'test_command': ['gcc', '-o', 'test', '{test_file}', '&&', './test']
        },
        'go': {
            'dockerfile': 'go.Dockerfile',
            'image_name': 'single_turn_scenarios:go',
            'file_extension': '.go',
            'run_command': ['go', 'run', '{filename}'],
            'test_command': ['go', 'test', '{test_file}']
        },
        'rust': {
            'dockerfile': 'rust.Dockerfile',
            'image_name': 'single_turn_scenarios:rust', 
            'file_extension': '.rs',
            'run_command': ['rustc', '{filename}', '&&', './{output}'],
            'test_command': ['cargo', 'test', '{test_name}']
        }
    }
    
    # Default resource limits
    DEFAULT_LIMITS = {
        'cpu_count': 1,
        'memory_mb': 512,
        'timeout_s': 30,
        'disk_mb': 100,
        'max_processes': 10
    }
    
    # Security violation patterns by language
    SECURITY_PATTERNS = {
        'python': [
            r'import\s+os',
            r'import\s+subprocess', 
            r'import\s+sys',
            r'import\s+socket',
            r'import\s+urllib',
            r'import\s+requests',
            r'import\s+http',
            r'import\s+ftplib',
            r'import\s+telnetlib',
            r'import\s+smtplib',
            r'import\s+poplib',
            r'import\s+imaplib',
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\(',
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'delattr\s*\(',
            r'hasattr\s*\(',
        ],
        'javascript': [
            r'require\s*\(\s*[\'"]fs[\'"]',
            r'require\s*\(\s*[\'"]child_process[\'"]',
            r'require\s*\(\s*[\'"]net[\'"]',
            r'require\s*\(\s*[\'"]http[\'"]',
            r'require\s*\(\s*[\'"]https[\'"]',
            r'require\s*\(\s*[\'"]url[\'"]',
            r'require\s*\(\s*[\'"]os[\'"]',
            r'require\s*\(\s*[\'"]path[\'"]',
            r'eval\s*\(',
            r'Function\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\(',
            r'process\.',
            r'global\.',
            r'window\.',
            r'document\.',
        ],
        'java': [
            r'import\s+java\.io\.',
            r'import\s+java\.net\.',
            r'import\s+java\.nio\.',
            r'import\s+java\.lang\.reflect\.',
            r'import\s+java\.lang\.Runtime',
            r'import\s+java\.lang\.ProcessBuilder',
            r'Runtime\.getRuntime',
            r'ProcessBuilder',
            r'System\.exit',
            r'System\.getProperty',
            r'System\.setProperty',
            r'Class\.forName',
            r'Method\.invoke',
            r'Field\.set',
        ],
        'cpp': [
            r'#include\s*<fstream>',
            r'#include\s*<filesystem>',
            r'#include\s*<cstdlib>',
            r'#include\s*<unistd\.h>',
            r'#include\s*<sys/',
            r'system\s*\(',
            r'exec\w*\s*\(',
            r'fork\s*\(',
            r'popen\s*\(',
            r'fopen\s*\(',
            r'freopen\s*\(',
        ],
        'c': [
            r'#include\s*<stdlib\.h>',
            r'#include\s*<unistd\.h>',
            r'#include\s*<sys/',
            r'system\s*\(',
            r'exec\w*\s*\(',
            r'fork\s*\(',
            r'popen\s*\(',
            r'fopen\s*\(',
            r'freopen\s*\(',
        ],
        'go': [
            r'import\s+"os"',
            r'import\s+"os/exec"',
            r'import\s+"net"',
            r'import\s+"net/http"',
            r'import\s+"syscall"',
            r'os\.Exit',
            r'os\.Getenv',
            r'os\.Setenv',
            r'exec\.Command',
            r'syscall\.',
        ],
        'rust': [
            r'use\s+std::process',
            r'use\s+std::fs',
            r'use\s+std::net',
            r'use\s+std::env',
            r'Command::new',
            r'File::open',
            r'File::create',
            r'env::var',
            r'env::set_var',
        ]
    }
    
    def __init__(self, language: str, limits: Optional[Dict[str, Any]] = None):
        """
        Initialize sandbox executor for specified language.
        
        Args:
            language: Programming language (python, javascript, java, etc.)
            limits: Resource limits override (cpu_count, memory_mb, timeout_s, etc.)
        """
        if language not in self.LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {language}")
            
        self.language = language
        self.config = self.LANGUAGE_CONFIGS[language]
        self.limits = {**self.DEFAULT_LIMITS, **(limits or {})}
        
        # Initialize Docker client
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available. Please install Docker to use sandbox execution.")
            
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Docker client: {e}")
            
        # Ensure Docker image exists
        self._ensure_image()
        
        # Track active containers for cleanup
        self.active_containers = []
        
    def _ensure_image(self) -> None:
        """Ensure Docker image exists, build if necessary."""
        image_name = self.config['image_name']
        
        try:
            self.docker_client.images.get(image_name)
            logger.debug(f"Docker image {image_name} already exists")
        except ImageNotFound:
            logger.info(f"Building Docker image {image_name}")
            self._build_image()
            
    def _build_image(self) -> None:
        """Build Docker image from Dockerfile."""
        dockerfile_path = Path(__file__).parent / 'docker' / self.config['dockerfile']
        
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
            
        try:
            # Build image
            image, build_logs = self.docker_client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile=self.config['dockerfile'],
                tag=self.config['image_name'],
                rm=True,
                forcerm=True
            )
            
            logger.info(f"Successfully built image {self.config['image_name']}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to build Docker image: {e}")
            
    def _create_temp_workspace(self) -> str:
        """Create temporary workspace directory."""
        workspace = tempfile.mkdtemp(prefix='sandbox_')
        logger.debug(f"Created temporary workspace: {workspace}")
        return workspace
        
    def _cleanup_workspace(self, workspace: str) -> None:
        """Clean up temporary workspace."""
        try:
            shutil.rmtree(workspace)
            logger.debug(f"Cleaned up workspace: {workspace}")
        except Exception as e:
            logger.warning(f"Failed to cleanup workspace {workspace}: {e}")
            
    def _detect_security_violations(self, code: str) -> List[str]:
        """
        Detect potential security violations in code.
        
        Args:
            code: Source code to analyze
            
        Returns:
            List of detected security violations
        """
        import re
        violations = []
        
        # Get patterns for current language
        patterns = self.SECURITY_PATTERNS.get(self.language, [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                violations.append(
                    f"Line {line_num}: Detected potentially dangerous pattern: {pattern} -> '{match.group()}'"
                )
                
        # Additional runtime security checks
        violations.extend(self._check_runtime_security_patterns(code))
                
        return violations
        
    def _check_runtime_security_patterns(self, code: str) -> List[str]:
        """Check for runtime security patterns that might be obfuscated."""
        import re
        violations = []
        
        # Check for string concatenation that might hide dangerous calls
        suspicious_strings = [
            r'[\'"]os[\'"]',
            r'[\'"]sys[\'"]', 
            r'[\'"]subprocess[\'"]',
            r'[\'"]socket[\'"]',
            r'[\'"]eval[\'"]',
            r'[\'"]exec[\'"]',
            r'[\'"]system[\'"]',
        ]
        
        for pattern in suspicious_strings:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Suspicious string literal detected: {pattern}")
                
        # Check for base64 or hex encoded content (potential obfuscation)
        if re.search(r'base64|b64decode|fromhex|unhexlify', code, re.IGNORECASE):
            violations.append("Potential code obfuscation detected (base64/hex)")
            
        # Check for dynamic imports or function calls
        dynamic_patterns = [
            r'getattr\s*\([^)]*[\'"]__import__[\'"]',
            r'globals\s*\(\s*\)\s*\[',
            r'locals\s*\(\s*\)\s*\[',
        ]
        
        for pattern in dynamic_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Dynamic code execution pattern detected: {pattern}")
                
        return violations
        
    def _prepare_execution_environment(self, code: str, tests: List[Dict]) -> Tuple[str, str, List[str]]:
        """
        Prepare execution environment with code and test files.
        
        Args:
            code: Source code to execute
            tests: List of test definitions
            
        Returns:
            Tuple of (workspace_path, main_file_path, test_file_paths)
        """
        workspace = self._create_temp_workspace()
        
        # Write main code file
        main_filename = f"main{self.config['file_extension']}"
        main_file_path = os.path.join(workspace, main_filename)
        
        with open(main_file_path, 'w', encoding='utf-8') as f:
            f.write(code)
            
        # Write test files
        test_file_paths = []
        for i, test in enumerate(tests):
            if 'content' in test:
                test_filename = f"test_{i}{self.config['file_extension']}"
                test_file_path = os.path.join(workspace, test_filename)
                
                with open(test_file_path, 'w', encoding='utf-8') as f:
                    f.write(test['content'])
                    
                test_file_paths.append(test_file_path)
                
        return workspace, main_file_path, test_file_paths
        
    def _run_container(self, workspace: str, command: List[str]) -> ExecutionResult:
        """
        Run command in Docker container with resource limits and monitoring.
        
        Args:
            workspace: Host workspace directory
            command: Command to execute in container
            
        Returns:
            ExecutionResult with execution details
        """
        container = None
        start_time = time.time()
        monitoring_thread = None
        
        try:
            # Container configuration with enhanced security
            container_config = {
                'image': self.config['image_name'],
                'command': command,
                'working_dir': '/workspace',
                'volumes': {workspace: {'bind': '/workspace', 'mode': 'rw'}},
                'mem_limit': f"{self.limits['memory_mb']}m",
                'memswap_limit': f"{self.limits['memory_mb']}m",
                'cpu_count': self.limits['cpu_count'],
                'cpu_quota': int(self.limits['cpu_count'] * 100000),  # CPU quota in microseconds
                'cpu_period': 100000,  # 100ms period
                'pids_limit': self.limits['max_processes'],
                'network_disabled': True,  # Disable network access
                'read_only': False,  # Allow writes to workspace only
                'security_opt': [
                    'no-new-privileges:true',
                    'seccomp=unconfined',  # We'll rely on other security measures
                ],
                'cap_drop': ['ALL'],  # Drop all capabilities
                'cap_add': [],  # Don't add any capabilities back
                'user': '1000:1000',  # Run as non-root user
                'detach': True,
                'stdout': True,
                'stderr': True,
                'remove': False,  # We'll remove manually after getting stats
                'tmpfs': {
                    '/tmp': f'size={self.limits["disk_mb"]}m,noexec,nosuid,nodev',
                    '/var/tmp': f'size={self.limits["disk_mb"]}m,noexec,nosuid,nodev'
                },
                'ulimits': [
                    docker.types.Ulimit(name='nproc', soft=self.limits['max_processes'], hard=self.limits['max_processes']),
                    docker.types.Ulimit(name='nofile', soft=1024, hard=1024),
                    docker.types.Ulimit(name='fsize', soft=self.limits['disk_mb'] * 1024 * 1024, hard=self.limits['disk_mb'] * 1024 * 1024)
                ]
            }
            
            # Create and start container
            container = self.docker_client.containers.run(**container_config)
            self.active_containers.append(container)
            
            # Start monitoring thread for resource usage and security violations
            monitoring_data = {'peak_memory': 0, 'violations': []}
            monitoring_thread = self._start_monitoring(container, monitoring_data)
            
            # Wait for completion with timeout and monitoring
            exit_code = None
            try:
                result = container.wait(timeout=self.limits['timeout_s'])
                exit_code = result['StatusCode']
            except Exception as e:
                # Timeout or other error occurred
                logger.warning(f"Container execution interrupted: {e}")
                self._terminate_container(container, "Execution timeout or error")
                raise TimeoutError(f"Execution timed out after {self.limits['timeout_s']} seconds")
                
            # Stop monitoring
            if monitoring_thread and monitoring_thread.is_alive():
                monitoring_thread.join(timeout=1.0)
                
            # Get output
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')
            
            wall_time = time.time() - start_time
            
            # Check for runtime security violations in output
            runtime_violations = self._check_runtime_violations(stdout, stderr)
            all_violations = monitoring_data['violations'] + runtime_violations
            
            return ExecutionResult(
                stdout=stdout,
                stderr=stderr, 
                exit_code=exit_code,
                wall_time=wall_time,
                peak_memory=monitoring_data['peak_memory'],
                security_violations=all_violations,
                success=(exit_code == 0 and not all_violations)
            )
            
        except TimeoutError as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                wall_time=time.time() - start_time,
                peak_memory=monitoring_data.get('peak_memory', 0) if 'monitoring_data' in locals() else 0,
                security_violations=monitoring_data.get('violations', []) if 'monitoring_data' in locals() else [],
                success=False,
                error_message=str(e)
            )
            
        except Exception as e:
            logger.error(f"Container execution failed: {e}")
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1, 
                wall_time=time.time() - start_time,
                peak_memory=0,
                security_violations=[],
                success=False,
                error_message=str(e)
            )
            
        finally:
            # Stop monitoring thread
            if monitoring_thread and monitoring_thread.is_alive():
                monitoring_thread.join(timeout=1.0)
                
            # Clean up container
            if container:
                self._cleanup_container(container)
                
    def _start_monitoring(self, container, monitoring_data: Dict) -> 'threading.Thread':
        """Start monitoring thread for container resource usage and security."""
        import threading
        
        def monitor():
            try:
                while True:
                    try:
                        # Check if container is still running
                        container.reload()
                        if container.status != 'running':
                            break
                            
                        # Get resource stats
                        stats = container.stats(stream=False)
                        
                        # Update peak memory usage
                        if 'memory' in stats and 'usage' in stats['memory']:
                            current_memory = stats['memory']['usage'] // (1024 * 1024)  # Convert to MB
                            monitoring_data['peak_memory'] = max(monitoring_data['peak_memory'], current_memory)
                            
                        # Check for memory limit violations
                        if monitoring_data['peak_memory'] > self.limits['memory_mb']:
                            monitoring_data['violations'].append(f"Memory limit exceeded: {monitoring_data['peak_memory']}MB > {self.limits['memory_mb']}MB")
                            self._terminate_container(container, "Memory limit exceeded")
                            break
                            
                        # Check CPU usage (if available)
                        if 'cpu_stats' in stats and 'precpu_stats' in stats:
                            cpu_percent = self._calculate_cpu_percent(stats)
                            if cpu_percent > 95.0:  # High CPU usage threshold
                                monitoring_data['violations'].append(f"High CPU usage detected: {cpu_percent:.1f}%")
                                
                        time.sleep(0.1)  # Monitor every 100ms
                        
                    except Exception as e:
                        logger.debug(f"Monitoring error: {e}")
                        break
                        
            except Exception as e:
                logger.warning(f"Monitoring thread failed: {e}")
                
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        return thread
        
    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
                return cpu_percent
        except (KeyError, ZeroDivisionError):
            pass
        return 0.0
        
    def _check_runtime_violations(self, stdout: str, stderr: str) -> List[str]:
        """Check for security violations in runtime output."""
        violations = []
        
        # Check for suspicious output patterns
        suspicious_outputs = [
            r'Permission denied',
            r'Access denied', 
            r'Segmentation fault',
            r'Bus error',
            r'Killed',
            r'Out of memory',
            r'Resource temporarily unavailable',
            r'Too many open files',
            r'Network is unreachable',
            r'Connection refused',
        ]
        
        combined_output = f"{stdout}\n{stderr}"
        
        for pattern in suspicious_outputs:
            import re
            if re.search(pattern, combined_output, re.IGNORECASE):
                violations.append(f"Suspicious runtime output: {pattern}")
                
        return violations
        
    def _terminate_container(self, container, reason: str) -> None:
        """Forcefully terminate container due to security violation or resource limit."""
        logger.warning(f"Terminating container {container.id[:12]}: {reason}")
        
        try:
            # Try graceful stop first
            container.stop(timeout=2)
        except Exception:
            # Force kill if graceful stop fails
            try:
                container.kill()
            except Exception as e:
                logger.error(f"Failed to kill container: {e}")
                
    def _cleanup_container(self, container) -> None:
        """Clean up container resources."""
        try:
            if container in self.active_containers:
                self.active_containers.remove(container)
                
            # Stop container if still running
            container.reload()
            if container.status == 'running':
                container.stop(timeout=5)
                
            # Remove container
            container.remove(force=True)
            logger.debug(f"Cleaned up container {container.id[:12]}")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup container: {e}")
                    
    def execute_code(self, code: str, tests: Optional[List[Dict]] = None) -> ExecutionResult:
        """
        Execute code in secure sandbox environment.
        
        Args:
            code: Source code to execute
            tests: Optional list of test definitions
            
        Returns:
            ExecutionResult with execution details and security analysis
        """
        # Detect security violations before execution
        security_violations = self._detect_security_violations(code)
        
        if security_violations:
            logger.warning(f"Security violations detected: {security_violations}")
            return ExecutionResult(
                stdout="",
                stderr="Security violations detected - execution blocked",
                exit_code=-1,
                wall_time=0.0,
                peak_memory=0,
                security_violations=security_violations,
                success=False,
                error_message="Code contains security violations"
            )
            
        workspace = None
        
        try:
            # Prepare execution environment
            workspace, main_file, test_files = self._prepare_execution_environment(
                code, tests or []
            )
            
            # Prepare execution command
            main_filename = os.path.basename(main_file)
            command = self._prepare_command(main_filename)
            
            # Execute code in container
            result = self._run_container(workspace, command)
            
            # Run tests if provided
            if tests and result.success:
                test_results = self._run_tests(workspace, test_files, tests)
                result = self._merge_results(result, test_results)
                
            return result
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                wall_time=0.0,
                peak_memory=0,
                security_violations=security_violations,
                success=False,
                error_message=str(e)
            )
            
        finally:
            # Clean up workspace
            if workspace:
                self._cleanup_workspace(workspace)
                
    def _prepare_command(self, filename: str) -> List[str]:
        """Prepare execution command for the language."""
        command_template = self.config['run_command']
        
        # Handle different command formats
        if self.language in ['cpp', 'c', 'rust']:
            output_name = filename.rsplit('.', 1)[0]
            command = [cmd.format(filename=filename, output=output_name) 
                      for cmd in command_template]
        elif self.language == 'java':
            classname = filename.rsplit('.', 1)[0]
            command = [cmd.format(filename=filename, classname=classname)
                      for cmd in command_template]
        else:
            command = [cmd.format(filename=filename) for cmd in command_template]
            
        # Handle shell commands (&&)
        if '&&' in command:
            # Convert to shell execution
            shell_command = ' '.join(command)
            return ['/bin/bash', '-c', shell_command]
        else:
            return command
            
    def _run_tests(self, workspace: str, test_files: List[str], tests: List[Dict]) -> ExecutionResult:
        """Run test suite in sandbox."""
        if not test_files:
            return ExecutionResult(
                stdout="No tests to run",
                stderr="",
                exit_code=0,
                wall_time=0.0,
                peak_memory=0,
                security_violations=[],
                success=True
            )
            
        # Run each test file
        all_stdout = []
        all_stderr = []
        total_time = 0.0
        max_memory = 0
        overall_success = True
        
        for test_file, test_def in zip(test_files, tests):
            test_filename = os.path.basename(test_file)
            
            # Prepare test command
            if 'cmd' in test_def:
                # Use custom command
                command = test_def['cmd'].split()
            else:
                # Use default test command
                command_template = self.config['test_command']
                command = [cmd.format(test_file=test_filename) for cmd in command_template]
                
            # Execute test
            result = self._run_container(workspace, command)
            
            all_stdout.append(f"=== Test {test_filename} ===\n{result.stdout}")
            all_stderr.append(result.stderr)
            total_time += result.wall_time
            max_memory = max(max_memory, result.peak_memory)
            
            if not result.success:
                overall_success = False
                
        return ExecutionResult(
            stdout='\n'.join(all_stdout),
            stderr='\n'.join(all_stderr),
            exit_code=0 if overall_success else 1,
            wall_time=total_time,
            peak_memory=max_memory,
            security_violations=[],
            success=overall_success
        )
        
    def _merge_results(self, main_result: ExecutionResult, test_result: ExecutionResult) -> ExecutionResult:
        """Merge main execution and test results."""
        return ExecutionResult(
            stdout=f"{main_result.stdout}\n\n=== TESTS ===\n{test_result.stdout}",
            stderr=f"{main_result.stderr}\n{test_result.stderr}",
            exit_code=main_result.exit_code if main_result.exit_code != 0 else test_result.exit_code,
            wall_time=main_result.wall_time + test_result.wall_time,
            peak_memory=max(main_result.peak_memory, test_result.peak_memory),
            security_violations=main_result.security_violations,
            success=main_result.success and test_result.success,
            error_message=main_result.error_message or test_result.error_message
        )
        
    def cleanup(self) -> None:
        """Clean up all active containers and resources."""
        logger.info("Cleaning up sandbox executor")
        
        # Stop and remove all active containers
        for container in self.active_containers[:]:  # Copy list to avoid modification during iteration
            try:
                container.kill()
                container.remove(force=True)
                self.active_containers.remove(container)
                logger.debug(f"Cleaned up container {container.id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup container {container.id}: {e}")
                
        # Close Docker client
        try:
            self.docker_client.close()
        except Exception as e:
            logger.warning(f"Failed to close Docker client: {e}")
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


def create_sandbox_executor(language: str, limits: Optional[Dict[str, Any]] = None) -> SandboxExecutor:
    """
    Factory function to create sandbox executor.
    
    Args:
        language: Programming language
        limits: Optional resource limits
        
    Returns:
        Configured SandboxExecutor instance
    """
    return SandboxExecutor(language, limits)


# Utility functions for common operations
def execute_code_safely(language: str, code: str, tests: Optional[List[Dict]] = None, 
                       limits: Optional[Dict[str, Any]] = None) -> ExecutionResult:
    """
    Convenience function to execute code safely in sandbox.
    
    Args:
        language: Programming language
        code: Source code to execute
        tests: Optional test definitions
        limits: Optional resource limits
        
    Returns:
        ExecutionResult with execution details
    """
    with create_sandbox_executor(language, limits) as executor:
        return executor.execute_code(code, tests)


def validate_sandbox_environment() -> Dict[str, bool]:
    """
    Validate that sandbox environment is properly configured.
    
    Returns:
        Dictionary mapping language to availability status
    """
    results = {}
    
    # Check Docker availability
    try:
        client = docker.from_env()
        client.ping()
        docker_available = True
        client.close()
    except Exception:
        docker_available = False
        
    if not docker_available:
        return {lang: False for lang in SandboxExecutor.LANGUAGE_CONFIGS.keys()}
        
    # Check each language environment
    for language in SandboxExecutor.LANGUAGE_CONFIGS.keys():
        try:
            # Try to create executor (this will build image if needed)
            with create_sandbox_executor(language) as executor:
                # Run simple test
                simple_tests = {
                    'python': 'print("Hello, World!")',
                    'javascript': 'console.log("Hello, World!");',
                    'typescript': 'console.log("Hello, World!");',
                    'java': 'public class Main { public static void main(String[] args) { System.out.println("Hello, World!"); } }',
                    'cpp': '#include <iostream>\nint main() { std::cout << "Hello, World!" << std::endl; return 0; }',
                    'c': '#include <stdio.h>\nint main() { printf("Hello, World!\\n"); return 0; }',
                    'go': 'package main\nimport "fmt"\nfunc main() { fmt.Println("Hello, World!") }',
                    'rust': 'fn main() { println!("Hello, World!"); }'
                }
                
                if language in simple_tests:
                    result = executor.execute_code(simple_tests[language])
                    results[language] = result.success and "Hello, World!" in result.stdout
                else:
                    results[language] = True
                    
        except Exception as e:
            logger.warning(f"Failed to validate {language} environment: {e}")
            results[language] = False
            
    return results


class ExecutionLogger:
    """
    Execution logging and audit trail functionality.
    
    Provides comprehensive logging of all sandbox executions for security
    auditing and debugging purposes.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize execution logger.
        
        Args:
            log_dir: Directory to store execution logs (default: temp directory)
        """
        if log_dir is None:
            log_dir = os.path.join(tempfile.gettempdir(), 'sandbox_logs')
            
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('sandbox_execution')
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_dir / 'execution.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def log_execution_start(self, execution_id: str, language: str, code_hash: str, 
                           limits: Dict[str, Any]) -> None:
        """Log the start of code execution."""
        self.logger.info(
            f"EXECUTION_START - ID: {execution_id}, Language: {language}, "
            f"CodeHash: {code_hash}, Limits: {limits}"
        )
        
    def log_execution_end(self, execution_id: str, result: ExecutionResult) -> None:
        """Log the end of code execution."""
        self.logger.info(
            f"EXECUTION_END - ID: {execution_id}, Success: {result.success}, "
            f"ExitCode: {result.exit_code}, WallTime: {result.wall_time:.3f}s, "
            f"PeakMemory: {result.peak_memory}MB, Violations: {len(result.security_violations)}"
        )
        
    def log_security_violation(self, execution_id: str, violation: str) -> None:
        """Log security violation."""
        self.logger.warning(
            f"SECURITY_VIOLATION - ID: {execution_id}, Violation: {violation}"
        )
        
    def log_resource_limit_exceeded(self, execution_id: str, resource: str, 
                                   limit: Any, actual: Any) -> None:
        """Log resource limit exceeded."""
        self.logger.warning(
            f"RESOURCE_LIMIT_EXCEEDED - ID: {execution_id}, Resource: {resource}, "
            f"Limit: {limit}, Actual: {actual}"
        )
        
    def log_container_event(self, execution_id: str, event: str, details: str = "") -> None:
        """Log container lifecycle events."""
        self.logger.info(
            f"CONTAINER_EVENT - ID: {execution_id}, Event: {event}, Details: {details}"
        )
        
    def create_audit_report(self, start_time: Optional[str] = None, 
                           end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Create audit report for executions within time range.
        
        Args:
            start_time: Start time in ISO format (optional)
            end_time: End time in ISO format (optional)
            
        Returns:
            Dictionary containing audit report data
        """
        log_file = self.log_dir / 'execution.log'
        
        if not log_file.exists():
            return {'error': 'No execution log found'}
            
        report = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'security_violations': 0,
            'resource_violations': 0,
            'languages_used': set(),
            'violations_by_type': {},
            'execution_times': [],
            'memory_usage': []
        }
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if 'EXECUTION_END' in line:
                        report['total_executions'] += 1
                        if 'Success: True' in line:
                            report['successful_executions'] += 1
                        else:
                            report['failed_executions'] += 1
                            
                        # Extract execution time and memory
                        import re
                        time_match = re.search(r'WallTime: ([\d.]+)s', line)
                        if time_match:
                            report['execution_times'].append(float(time_match.group(1)))
                            
                        memory_match = re.search(r'PeakMemory: (\d+)MB', line)
                        if memory_match:
                            report['memory_usage'].append(int(memory_match.group(1)))
                            
                    elif 'EXECUTION_START' in line:
                        # Extract language
                        lang_match = re.search(r'Language: (\w+)', line)
                        if lang_match:
                            report['languages_used'].add(lang_match.group(1))
                            
                    elif 'SECURITY_VIOLATION' in line:
                        report['security_violations'] += 1
                        
                    elif 'RESOURCE_LIMIT_EXCEEDED' in line:
                        report['resource_violations'] += 1
                        
        except Exception as e:
            report['error'] = f"Failed to parse log file: {e}"
            
        # Convert set to list for JSON serialization
        report['languages_used'] = list(report['languages_used'])
        
        return report


class ExecutionResultSerializer:
    """
    Serialization and deserialization for ExecutionResult objects.
    
    Provides JSON serialization with proper error handling and validation.
    """
    
    @staticmethod
    def to_dict(result: ExecutionResult) -> Dict[str, Any]:
        """
        Convert ExecutionResult to dictionary.
        
        Args:
            result: ExecutionResult instance
            
        Returns:
            Dictionary representation
        """
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.exit_code,
            'wall_time': result.wall_time,
            'peak_memory': result.peak_memory,
            'security_violations': result.security_violations,
            'success': result.success,
            'error_message': result.error_message,
            'timestamp': time.time(),
            'version': '1.0'
        }
        
    @staticmethod
    def to_json(result: ExecutionResult, indent: Optional[int] = None) -> str:
        """
        Convert ExecutionResult to JSON string.
        
        Args:
            result: ExecutionResult instance
            indent: JSON indentation (optional)
            
        Returns:
            JSON string representation
        """
        try:
            return json.dumps(
                ExecutionResultSerializer.to_dict(result), 
                indent=indent,
                ensure_ascii=False
            )
        except Exception as e:
            # Fallback for serialization errors
            error_result = ExecutionResult(
                stdout="",
                stderr=f"Serialization error: {e}",
                exit_code=-1,
                wall_time=0.0,
                peak_memory=0,
                security_violations=[],
                success=False,
                error_message=f"Failed to serialize result: {e}"
            )
            return json.dumps(ExecutionResultSerializer.to_dict(error_result), indent=indent)
            
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ExecutionResult:
        """
        Create ExecutionResult from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ExecutionResult instance
        """
        try:
            return ExecutionResult(
                stdout=data.get('stdout', ''),
                stderr=data.get('stderr', ''),
                exit_code=data.get('exit_code', -1),
                wall_time=data.get('wall_time', 0.0),
                peak_memory=data.get('peak_memory', 0),
                security_violations=data.get('security_violations', []),
                success=data.get('success', False),
                error_message=data.get('error_message')
            )
        except Exception as e:
            # Return error result if deserialization fails
            return ExecutionResult(
                stdout="",
                stderr=f"Deserialization error: {e}",
                exit_code=-1,
                wall_time=0.0,
                peak_memory=0,
                security_violations=[],
                success=False,
                error_message=f"Failed to deserialize result: {e}"
            )
            
    @staticmethod
    def from_json(json_str: str) -> ExecutionResult:
        """
        Create ExecutionResult from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            ExecutionResult instance
        """
        try:
            data = json.loads(json_str)
            return ExecutionResultSerializer.from_dict(data)
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=f"JSON parsing error: {e}",
                exit_code=-1,
                wall_time=0.0,
                peak_memory=0,
                security_violations=[],
                success=False,
                error_message=f"Failed to parse JSON: {e}"
            )


class ExecutionResultHandler:
    """
    Comprehensive execution result handling with error states and logging.
    
    Provides centralized handling of execution results with proper error
    categorization and audit trail management.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize result handler.
        
        Args:
            log_dir: Directory for execution logs
        """
        self.logger = ExecutionLogger(log_dir)
        self.serializer = ExecutionResultSerializer()
        
    def handle_execution_result(self, execution_id: str, result: ExecutionResult,
                               language: str, code: str) -> Dict[str, Any]:
        """
        Handle execution result with comprehensive logging and error categorization.
        
        Args:
            execution_id: Unique execution identifier
            result: ExecutionResult from sandbox execution
            language: Programming language used
            code: Source code that was executed
            
        Returns:
            Processed result dictionary with additional metadata
        """
        # Log execution completion
        self.logger.log_execution_end(execution_id, result)
        
        # Log security violations
        for violation in result.security_violations:
            self.logger.log_security_violation(execution_id, violation)
            
        # Categorize error state
        error_category = self._categorize_error(result)
        
        # Create processed result
        processed_result = {
            'execution_id': execution_id,
            'language': language,
            'result': self.serializer.to_dict(result),
            'error_category': error_category,
            'code_hash': self._hash_code(code),
            'processed_at': time.time()
        }
        
        # Add recommendations based on error category
        if error_category != 'success':
            processed_result['recommendations'] = self._get_error_recommendations(error_category, result)
            
        return processed_result
        
    def _categorize_error(self, result: ExecutionResult) -> str:
        """Categorize execution error for better handling."""
        if result.success:
            return 'success'
            
        if result.security_violations:
            return 'security_violation'
            
        if result.exit_code == -1:
            if 'timeout' in (result.error_message or '').lower():
                return 'timeout'
            elif 'memory' in (result.error_message or '').lower():
                return 'memory_limit'
            else:
                return 'system_error'
                
        if result.exit_code != 0:
            if 'compilation' in result.stderr.lower() or 'syntax' in result.stderr.lower():
                return 'compilation_error'
            elif 'runtime' in result.stderr.lower() or 'exception' in result.stderr.lower():
                return 'runtime_error'
            else:
                return 'execution_error'
                
        return 'unknown_error'
        
    def _get_error_recommendations(self, error_category: str, result: ExecutionResult) -> List[str]:
        """Get recommendations based on error category."""
        recommendations = []
        
        if error_category == 'security_violation':
            recommendations.extend([
                "Remove potentially dangerous imports or function calls",
                "Avoid file system operations, network access, or system calls",
                "Use only safe, computational operations"
            ])
            
        elif error_category == 'timeout':
            recommendations.extend([
                "Optimize algorithm complexity",
                "Avoid infinite loops or recursive calls without base cases",
                "Consider iterative solutions instead of recursive ones"
            ])
            
        elif error_category == 'memory_limit':
            recommendations.extend([
                "Reduce memory usage by avoiding large data structures",
                "Use generators or streaming for large datasets",
                "Free unused variables and objects"
            ])
            
        elif error_category == 'compilation_error':
            recommendations.extend([
                "Check syntax and fix compilation errors",
                "Ensure all imports and dependencies are available",
                "Verify language-specific syntax requirements"
            ])
            
        elif error_category == 'runtime_error':
            recommendations.extend([
                "Add proper error handling and input validation",
                "Check for null/undefined values before use",
                "Ensure array/list bounds are respected"
            ])
            
        return recommendations
        
    def _hash_code(self, code: str) -> str:
        """Create hash of source code for tracking."""
        import hashlib
        return hashlib.sha256(code.encode('utf-8')).hexdigest()[:16]
        
    def save_result(self, processed_result: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """
        Save processed result to file.
        
        Args:
            processed_result: Processed execution result
            output_dir: Output directory (optional)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = self.logger.log_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        # Create filename with timestamp and execution ID
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        execution_id = processed_result['execution_id']
        filename = f"result_{timestamp}_{execution_id}.json"
        
        filepath = output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(processed_result, f, indent=2, ensure_ascii=False)
            return str(filepath)
        except Exception as e:
            self.logger.logger.error(f"Failed to save result: {e}")
            raise
            
    def load_result(self, filepath: str) -> Dict[str, Any]:
        """
        Load processed result from file.
        
        Args:
            filepath: Path to result file
            
        Returns:
            Processed execution result
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.logger.error(f"Failed to load result from {filepath}: {e}")
            raise


# Enhanced SandboxExecutor with result handling
class EnhancedSandboxExecutor(SandboxExecutor):
    """
    Enhanced sandbox executor with comprehensive result handling and logging.
    """
    
    def __init__(self, language: str, limits: Optional[Dict[str, Any]] = None, 
                 log_dir: Optional[str] = None):
        """
        Initialize enhanced sandbox executor.
        
        Args:
            language: Programming language
            limits: Resource limits
            log_dir: Directory for execution logs
        """
        super().__init__(language, limits)
        self.result_handler = ExecutionResultHandler(log_dir)
        
    def execute_code_with_logging(self, code: str, tests: Optional[List[Dict]] = None,
                                 execution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute code with comprehensive logging and result handling.
        
        Args:
            code: Source code to execute
            tests: Optional test definitions
            execution_id: Optional execution identifier
            
        Returns:
            Processed execution result with metadata
        """
        if execution_id is None:
            execution_id = f"{self.language}_{int(time.time())}_{os.getpid()}"
            
        # Log execution start
        code_hash = self.result_handler._hash_code(code)
        self.result_handler.logger.log_execution_start(
            execution_id, self.language, code_hash, self.limits
        )
        
        try:
            # Execute code
            result = self.execute_code(code, tests)
            
            # Handle and process result
            processed_result = self.result_handler.handle_execution_result(
                execution_id, result, self.language, code
            )
            
            return processed_result
            
        except Exception as e:
            # Handle execution failure
            error_result = ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                wall_time=0.0,
                peak_memory=0,
                security_violations=[],
                success=False,
                error_message=str(e)
            )
            
            return self.result_handler.handle_execution_result(
                execution_id, error_result, self.language, code
            )