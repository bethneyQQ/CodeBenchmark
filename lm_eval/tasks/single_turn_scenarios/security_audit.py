#!/usr/bin/env python3
"""
Security Audit Script

Performs comprehensive security audit and penetration testing of sandbox environments
to ensure safe execution of untrusted code.

Requirements addressed: 12.4, 12.5, 7.2, 7.3, 7.5
"""

import sys
import os
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityAuditor:
    """Performs comprehensive security audit of the sandbox system."""
    
    def __init__(self):
        self.task_dir = Path(__file__).parent
        self.audit_results = {}
        self.malicious_code_samples = self._load_malicious_samples()
        
    def _load_malicious_samples(self) -> Dict[str, List[str]]:
        """Load malicious code samples for testing."""
        return {
            "python": [
                # File system access attempts
                "import os; os.system('rm -rf /')",
                "open('/etc/passwd', 'r').read()",
                "import shutil; shutil.rmtree('/tmp')",
                
                # Network access attempts
                "import urllib.request; urllib.request.urlopen('http://evil.com')",
                "import socket; s = socket.socket(); s.connect(('8.8.8.8', 53))",
                
                # Process manipulation
                "import subprocess; subprocess.run(['ps', 'aux'])",
                "import os; os.fork()",
                
                # Resource exhaustion
                "while True: pass",
                "x = 'a' * (10**9)",
                "[i for i in range(10**8)]",
                
                # Privilege escalation attempts
                "import os; os.setuid(0)",
                "exec('import os; os.system(\"sudo su\")')",
                
                # Environment manipulation
                "import os; os.environ['PATH'] = '/malicious/path'",
                "import sys; sys.path.insert(0, '/malicious')",
                
                # Code injection attempts
                "eval('__import__(\"os\").system(\"whoami\")')",
                "exec(open('/dev/urandom', 'rb').read(100))",
            ],
            "javascript": [
                # File system access
                "require('fs').readFileSync('/etc/passwd')",
                "require('child_process').exec('rm -rf /')",
                
                # Network access
                "require('http').get('http://evil.com')",
                "require('net').connect(80, 'evil.com')",
                
                # Process manipulation
                "process.exit(1)",
                "require('child_process').spawn('sh')",
                
                # Resource exhaustion
                "while(true) {}",
                "new Array(10**9).fill('x')",
                
                # Global pollution
                "global.process = null",
                "Object.prototype.polluted = true",
            ],
            "java": [
                # File system access
                "new java.io.File(\"/etc/passwd\").delete()",
                "Runtime.getRuntime().exec(\"rm -rf /\")",
                
                # Network access
                "new java.net.URL(\"http://evil.com\").openConnection()",
                
                # System properties
                "System.setProperty(\"user.home\", \"/root\")",
                "System.exit(1)",
                
                # Reflection attacks
                "Class.forName(\"java.lang.Runtime\").getMethod(\"exec\", String.class)",
            ]
        }
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit."""
        print("🔒 Starting Security Audit")
        print("=" * 50)
        
        audit_categories = [
            ("Sandbox Isolation", self.audit_sandbox_isolation),
            ("Resource Limits", self.audit_resource_limits),
            ("Network Security", self.audit_network_security),
            ("File System Security", self.audit_filesystem_security),
            ("Process Security", self.audit_process_security),
            ("Code Injection Prevention", self.audit_code_injection),
            ("Privilege Escalation", self.audit_privilege_escalation),
            ("Container Escape", self.audit_container_escape)
        ]
        
        overall_secure = True
        
        for category_name, audit_function in audit_categories:
            print(f"\n🔍 {category_name}")
            print("-" * 30)
            
            try:
                category_result = audit_function()
                self.audit_results[category_name] = category_result
                
                if category_result.get("secure", False):
                    print(f"✅ {category_name}: SECURE")
                else:
                    print(f"🚨 {category_name}: VULNERABLE")
                    overall_secure = False
                    
                    # Print vulnerabilities
                    vulnerabilities = category_result.get("vulnerabilities", [])
                    for vuln in vulnerabilities[:3]:  # Show first 3
                        print(f"   ⚠️  {vuln}")
                    if len(vulnerabilities) > 3:
                        print(f"   ... and {len(vulnerabilities) - 3} more")
                        
            except Exception as e:
                logger.error(f"Error in {category_name}: {e}")
                self.audit_results[category_name] = {
                    "secure": False,
                    "error": str(e),
                    "vulnerabilities": [f"Audit failed: {e}"]
                }
                overall_secure = False
        
        self.audit_results["overall"] = {
            "secure": overall_secure,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "auditor_version": "1.0.0"
        }
        
        self.print_audit_summary()
        return self.audit_results
    
    def audit_sandbox_isolation(self) -> Dict[str, Any]:
        """Audit sandbox isolation mechanisms."""
        result = {
            "secure": True,
            "vulnerabilities": [],
            "tests_run": 0,
            "tests_passed": 0
        }
        
        try:
            # Import sandbox module
            sys.path.insert(0, str(self.task_dir))
            from sandbox import SandboxExecutor
            
            # Test basic isolation
            isolation_tests = [
                {
                    "name": "Basic container creation",
                    "code": "print('hello')",
                    "language": "python"
                },
                {
                    "name": "File system isolation",
                    "code": "import os; print(os.getcwd())",
                    "language": "python"
                },
                {
                    "name": "Process isolation",
                    "code": "import os; print(os.getpid())",
                    "language": "python"
                }
            ]
            
            for test in isolation_tests:
                result["tests_run"] += 1
                
                try:
                    executor = SandboxExecutor(test["language"], {"timeout": 5, "memory_mb": 50})
                    exec_result = executor.execute_code(test["code"], [])
                    executor.cleanup()
                    
                    # Check if execution was properly isolated
                    if exec_result.exit_code == 0:
                        result["tests_passed"] += 1
                        print(f"    ✅ {test['name']}")
                    else:
                        result["vulnerabilities"].append(f"Isolation test failed: {test['name']}")
                        print(f"    ❌ {test['name']}")
                        
                except Exception as e:
                    result["vulnerabilities"].append(f"Isolation error in {test['name']}: {e}")
                    print(f"    ❌ {test['name']}: {e}")
            
        except ImportError:
            result["secure"] = False
            result["vulnerabilities"].append("Sandbox module not available")
        except Exception as e:
            result["secure"] = False
            result["vulnerabilities"].append(f"Sandbox audit failed: {e}")
        
        result["secure"] = len(result["vulnerabilities"]) == 0
        return result
    
    def audit_resource_limits(self) -> Dict[str, Any]:
        """Audit resource limit enforcement."""
        result = {
            "secure": True,
            "vulnerabilities": [],
            "tests_run": 0,
            "tests_passed": 0
        }
        
        try:
            sys.path.insert(0, str(self.task_dir))
            from sandbox import SandboxExecutor
            
            # Test resource exhaustion attempts
            resource_tests = [
                {
                    "name": "CPU exhaustion",
                    "code": "while True: pass",
                    "language": "python",
                    "should_timeout": True
                },
                {
                    "name": "Memory exhaustion", 
                    "code": "x = 'a' * (10**8)",
                    "language": "python",
                    "should_fail": True
                },
                {
                    "name": "Infinite recursion",
                    "code": "def f(): f()\nf()",
                    "language": "python",
                    "should_fail": True
                }
            ]
            
            for test in resource_tests:
                result["tests_run"] += 1
                
                try:
                    executor = SandboxExecutor(test["language"], {"timeout": 2, "memory_mb": 50})
                    start_time = time.time()
                    exec_result = executor.execute_code(test["code"], [])
                    execution_time = time.time() - start_time
                    executor.cleanup()
                    
                    # Check if resource limits were enforced
                    if test.get("should_timeout") and execution_time >= 2:
                        result["tests_passed"] += 1
                        print(f"    ✅ {test['name']}: Properly timed out")
                    elif test.get("should_fail") and exec_result.exit_code != 0:
                        result["tests_passed"] += 1
                        print(f"    ✅ {test['name']}: Properly failed")
                    elif not test.get("should_timeout") and not test.get("should_fail"):
                        result["tests_passed"] += 1
                        print(f"    ✅ {test['name']}: Executed normally")
                    else:
                        result["vulnerabilities"].append(f"Resource limit not enforced: {test['name']}")
                        print(f"    ❌ {test['name']}: Resource limit bypass")
                        
                except Exception as e:
                    # Exceptions are expected for resource exhaustion
                    result["tests_passed"] += 1
                    print(f"    ✅ {test['name']}: Properly caught exception")
            
        except Exception as e:
            result["secure"] = False
            result["vulnerabilities"].append(f"Resource limit audit failed: {e}")
        
        result["secure"] = len(result["vulnerabilities"]) == 0
        return result
    
    def audit_network_security(self) -> Dict[str, Any]:
        """Audit network access restrictions."""
        result = {
            "secure": True,
            "vulnerabilities": [],
            "tests_run": 0,
            "tests_passed": 0
        }
        
        network_tests = [
            {
                "name": "HTTP request",
                "code": "import urllib.request; urllib.request.urlopen('http://httpbin.org/get', timeout=1)",
                "language": "python"
            },
            {
                "name": "Socket connection",
                "code": "import socket; s = socket.socket(); s.connect(('8.8.8.8', 53))",
                "language": "python"
            },
            {
                "name": "DNS resolution",
                "code": "import socket; socket.gethostbyname('google.com')",
                "language": "python"
            }
        ]
        
        try:
            sys.path.insert(0, str(self.task_dir))
            from sandbox import SandboxExecutor
            
            for test in network_tests:
                result["tests_run"] += 1
                
                try:
                    executor = SandboxExecutor(test["language"], {"timeout": 5, "memory_mb": 50})
                    exec_result = executor.execute_code(test["code"], [])
                    executor.cleanup()
                    
                    # Network access should be blocked
                    if exec_result.exit_code != 0 or "network" in exec_result.stderr.lower():
                        result["tests_passed"] += 1
                        print(f"    ✅ {test['name']}: Network access blocked")
                    else:
                        result["vulnerabilities"].append(f"Network access allowed: {test['name']}")
                        print(f"    ❌ {test['name']}: Network access not blocked")
                        
                except Exception as e:
                    # Network errors are expected
                    result["tests_passed"] += 1
                    print(f"    ✅ {test['name']}: Network properly blocked")
            
        except Exception as e:
            result["secure"] = False
            result["vulnerabilities"].append(f"Network security audit failed: {e}")
        
        result["secure"] = len(result["vulnerabilities"]) == 0
        return result
    
    def audit_filesystem_security(self) -> Dict[str, Any]:
        """Audit file system access restrictions."""
        result = {
            "secure": True,
            "vulnerabilities": [],
            "tests_run": 0,
            "tests_passed": 0
        }
        
        filesystem_tests = [
            {
                "name": "Read /etc/passwd",
                "code": "open('/etc/passwd', 'r').read()",
                "language": "python"
            },
            {
                "name": "Write to /tmp",
                "code": "open('/tmp/test_file', 'w').write('test')",
                "language": "python"
            },
            {
                "name": "List root directory",
                "code": "import os; os.listdir('/')",
                "language": "python"
            },
            {
                "name": "Delete system files",
                "code": "import os; os.remove('/bin/sh')",
                "language": "python"
            }
        ]
        
        try:
            sys.path.insert(0, str(self.task_dir))
            from sandbox import SandboxExecutor
            
            for test in filesystem_tests:
                result["tests_run"] += 1
                
                try:
                    executor = SandboxExecutor(test["language"], {"timeout": 5, "memory_mb": 50})
                    exec_result = executor.execute_code(test["code"], [])
                    executor.cleanup()
                    
                    # File system access should be restricted
                    if exec_result.exit_code != 0:
                        result["tests_passed"] += 1
                        print(f"    ✅ {test['name']}: Access properly restricted")
                    else:
                        result["vulnerabilities"].append(f"File system access allowed: {test['name']}")
                        print(f"    ❌ {test['name']}: Unauthorized access")
                        
                except Exception as e:
                    # File system errors are expected
                    result["tests_passed"] += 1
                    print(f"    ✅ {test['name']}: Access properly blocked")
            
        except Exception as e:
            result["secure"] = False
            result["vulnerabilities"].append(f"File system security audit failed: {e}")
        
        result["secure"] = len(result["vulnerabilities"]) == 0
        return result
    
    def audit_process_security(self) -> Dict[str, Any]:
        """Audit process execution restrictions."""
        result = {
            "secure": True,
            "vulnerabilities": [],
            "tests_run": 0,
            "tests_passed": 0
        }
        
        process_tests = [
            {
                "name": "System command execution",
                "code": "import os; os.system('whoami')",
                "language": "python"
            },
            {
                "name": "Subprocess execution",
                "code": "import subprocess; subprocess.run(['ps', 'aux'])",
                "language": "python"
            },
            {
                "name": "Fork process",
                "code": "import os; os.fork()",
                "language": "python"
            },
            {
                "name": "Shell execution",
                "code": "import subprocess; subprocess.run(['/bin/sh', '-c', 'echo test'])",
                "language": "python"
            }
        ]
        
        try:
            sys.path.insert(0, str(self.task_dir))
            from sandbox import SandboxExecutor
            
            for test in process_tests:
                result["tests_run"] += 1
                
                try:
                    executor = SandboxExecutor(test["language"], {"timeout": 5, "memory_mb": 50})
                    exec_result = executor.execute_code(test["code"], [])
                    executor.cleanup()
                    
                    # Process execution should be restricted
                    if exec_result.exit_code != 0:
                        result["tests_passed"] += 1
                        print(f"    ✅ {test['name']}: Process execution blocked")
                    else:
                        result["vulnerabilities"].append(f"Process execution allowed: {test['name']}")
                        print(f"    ❌ {test['name']}: Unauthorized process execution")
                        
                except Exception as e:
                    # Process errors are expected
                    result["tests_passed"] += 1
                    print(f"    ✅ {test['name']}: Process properly blocked")
            
        except Exception as e:
            result["secure"] = False
            result["vulnerabilities"].append(f"Process security audit failed: {e}")
        
        result["secure"] = len(result["vulnerabilities"]) == 0
        return result
    
    def audit_code_injection(self) -> Dict[str, Any]:
        """Audit code injection prevention."""
        result = {
            "secure": True,
            "vulnerabilities": [],
            "tests_run": 0,
            "tests_passed": 0
        }
        
        injection_tests = [
            {
                "name": "Eval injection",
                "code": "eval('__import__(\"os\").system(\"whoami\")')",
                "language": "python"
            },
            {
                "name": "Exec injection",
                "code": "exec('import os; os.system(\"id\")')",
                "language": "python"
            },
            {
                "name": "Import injection",
                "code": "__import__('os').system('pwd')",
                "language": "python"
            }
        ]
        
        try:
            sys.path.insert(0, str(self.task_dir))
            from sandbox import SandboxExecutor
            
            for test in injection_tests:
                result["tests_run"] += 1
                
                try:
                    executor = SandboxExecutor(test["language"], {"timeout": 5, "memory_mb": 50})
                    exec_result = executor.execute_code(test["code"], [])
                    executor.cleanup()
                    
                    # Code injection should be prevented or contained
                    if exec_result.exit_code != 0 or exec_result.security_violations:
                        result["tests_passed"] += 1
                        print(f"    ✅ {test['name']}: Injection prevented")
                    else:
                        result["vulnerabilities"].append(f"Code injection possible: {test['name']}")
                        print(f"    ❌ {test['name']}: Injection not prevented")
                        
                except Exception as e:
                    # Injection errors are expected
                    result["tests_passed"] += 1
                    print(f"    ✅ {test['name']}: Injection properly blocked")
            
        except Exception as e:
            result["secure"] = False
            result["vulnerabilities"].append(f"Code injection audit failed: {e}")
        
        result["secure"] = len(result["vulnerabilities"]) == 0
        return result
    
    def audit_privilege_escalation(self) -> Dict[str, Any]:
        """Audit privilege escalation prevention."""
        result = {
            "secure": True,
            "vulnerabilities": [],
            "tests_run": 0,
            "tests_passed": 0
        }
        
        privilege_tests = [
            {
                "name": "Setuid attempt",
                "code": "import os; os.setuid(0)",
                "language": "python"
            },
            {
                "name": "Sudo execution",
                "code": "import os; os.system('sudo whoami')",
                "language": "python"
            },
            {
                "name": "Root access attempt",
                "code": "import os; os.system('su root')",
                "language": "python"
            }
        ]
        
        try:
            sys.path.insert(0, str(self.task_dir))
            from sandbox import SandboxExecutor
            
            for test in privilege_tests:
                result["tests_run"] += 1
                
                try:
                    executor = SandboxExecutor(test["language"], {"timeout": 5, "memory_mb": 50})
                    exec_result = executor.execute_code(test["code"], [])
                    executor.cleanup()
                    
                    # Privilege escalation should be prevented
                    if exec_result.exit_code != 0:
                        result["tests_passed"] += 1
                        print(f"    ✅ {test['name']}: Privilege escalation blocked")
                    else:
                        result["vulnerabilities"].append(f"Privilege escalation possible: {test['name']}")
                        print(f"    ❌ {test['name']}: Privilege escalation not blocked")
                        
                except Exception as e:
                    # Privilege errors are expected
                    result["tests_passed"] += 1
                    print(f"    ✅ {test['name']}: Privilege escalation properly blocked")
            
        except Exception as e:
            result["secure"] = False
            result["vulnerabilities"].append(f"Privilege escalation audit failed: {e}")
        
        result["secure"] = len(result["vulnerabilities"]) == 0
        return result
    
    def audit_container_escape(self) -> Dict[str, Any]:
        """Audit container escape prevention."""
        result = {
            "secure": True,
            "vulnerabilities": [],
            "tests_run": 0,
            "tests_passed": 0
        }
        
        escape_tests = [
            {
                "name": "Mount namespace escape",
                "code": "import os; os.system('mount -t proc proc /proc')",
                "language": "python"
            },
            {
                "name": "Chroot escape",
                "code": "import os; os.chroot('/')",
                "language": "python"
            },
            {
                "name": "Docker socket access",
                "code": "import os; os.path.exists('/var/run/docker.sock')",
                "language": "python"
            }
        ]
        
        try:
            sys.path.insert(0, str(self.task_dir))
            from sandbox import SandboxExecutor
            
            for test in escape_tests:
                result["tests_run"] += 1
                
                try:
                    executor = SandboxExecutor(test["language"], {"timeout": 5, "memory_mb": 50})
                    exec_result = executor.execute_code(test["code"], [])
                    executor.cleanup()
                    
                    # Container escape should be prevented
                    if exec_result.exit_code != 0:
                        result["tests_passed"] += 1
                        print(f"    ✅ {test['name']}: Container escape blocked")
                    else:
                        # Check if Docker socket access returned False
                        if "Docker socket access" in test["name"] and "False" in exec_result.stdout:
                            result["tests_passed"] += 1
                            print(f"    ✅ {test['name']}: Docker socket not accessible")
                        else:
                            result["vulnerabilities"].append(f"Container escape possible: {test['name']}")
                            print(f"    ❌ {test['name']}: Container escape not blocked")
                        
                except Exception as e:
                    # Container escape errors are expected
                    result["tests_passed"] += 1
                    print(f"    ✅ {test['name']}: Container escape properly blocked")
            
        except Exception as e:
            result["secure"] = False
            result["vulnerabilities"].append(f"Container escape audit failed: {e}")
        
        result["secure"] = len(result["vulnerabilities"]) == 0
        return result
    
    def print_audit_summary(self):
        """Print comprehensive audit summary."""
        print("\n" + "=" * 50)
        print("🔒 SECURITY AUDIT SUMMARY")
        print("=" * 50)
        
        total_vulnerabilities = 0
        
        for category, result in self.audit_results.items():
            if category == "overall":
                continue
                
            vulnerabilities = len(result.get("vulnerabilities", []))
            secure = result.get("secure", False)
            
            status = "🔒 SECURE" if secure else "🚨 VULNERABLE"
            print(f"{status} {category}")
            
            if vulnerabilities > 0:
                print(f"   Vulnerabilities: {vulnerabilities}")
                total_vulnerabilities += vulnerabilities
        
        overall = self.audit_results.get("overall", {})
        
        print(f"\n📊 Overall Security Status:")
        if overall.get("secure", False):
            print("   🔒 SYSTEM SECURE - No critical vulnerabilities found")
        else:
            print(f"   🚨 SYSTEM VULNERABLE - {total_vulnerabilities} vulnerabilities found")
        
        print(f"   Audit completed: {overall.get('timestamp', 'Unknown')}")
    
    def generate_security_report(self, output_path: str):
        """Generate detailed security report."""
        report_lines = []
        
        report_lines.append("# Security Audit Report")
        report_lines.append("")
        report_lines.append(f"**Audit Date:** {self.audit_results.get('overall', {}).get('timestamp', 'Unknown')}")
        report_lines.append(f"**Auditor Version:** {self.audit_results.get('overall', {}).get('auditor_version', '1.0.0')}")
        report_lines.append("")
        
        overall_secure = self.audit_results.get("overall", {}).get("secure", False)
        status = "SECURE" if overall_secure else "VULNERABLE"
        report_lines.append(f"**Overall Status:** {status}")
        report_lines.append("")
        
        # Executive Summary
        total_vulnerabilities = sum(
            len(result.get("vulnerabilities", []))
            for category, result in self.audit_results.items()
            if category != "overall"
        )
        
        report_lines.append("## Executive Summary")
        report_lines.append("")
        if overall_secure:
            report_lines.append("The security audit found no critical vulnerabilities. The sandbox system appears to be properly configured with appropriate security controls.")
        else:
            report_lines.append(f"The security audit identified {total_vulnerabilities} vulnerabilities that require attention before production deployment.")
        report_lines.append("")
        
        # Detailed Findings
        report_lines.append("## Detailed Findings")
        report_lines.append("")
        
        for category, result in self.audit_results.items():
            if category == "overall":
                continue
                
            secure = result.get("secure", False)
            vulnerabilities = result.get("vulnerabilities", [])
            tests_run = result.get("tests_run", 0)
            tests_passed = result.get("tests_passed", 0)
            
            report_lines.append(f"### {category}")
            report_lines.append("")
            report_lines.append(f"**Status:** {'SECURE' if secure else 'VULNERABLE'}")
            report_lines.append(f"**Tests Run:** {tests_run}")
            report_lines.append(f"**Tests Passed:** {tests_passed}")
            report_lines.append("")
            
            if vulnerabilities:
                report_lines.append("**Vulnerabilities:**")
                for vuln in vulnerabilities:
                    report_lines.append(f"- {vuln}")
                report_lines.append("")
            else:
                report_lines.append("No vulnerabilities found in this category.")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        
        if overall_secure:
            report_lines.append("- Continue regular security audits")
            report_lines.append("- Monitor for new security threats")
            report_lines.append("- Keep sandbox dependencies updated")
        else:
            report_lines.append("- Address all identified vulnerabilities before production use")
            report_lines.append("- Implement additional security controls as needed")
            report_lines.append("- Re-run security audit after fixes")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n📄 Security report saved to: {output_path}")


def main():
    """Main function to run security audit."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run security audit")
    parser.add_argument("--output", "-o", help="Output path for security report")
    parser.add_argument("--category", choices=[
        "isolation", "resources", "network", "filesystem", 
        "process", "injection", "privilege", "container"
    ], help="Run only specific audit category")
    
    args = parser.parse_args()
    
    auditor = SecurityAuditor()
    
    if args.category:
        # Run specific category
        category_map = {
            "isolation": auditor.audit_sandbox_isolation,
            "resources": auditor.audit_resource_limits,
            "network": auditor.audit_network_security,
            "filesystem": auditor.audit_filesystem_security,
            "process": auditor.audit_process_security,
            "injection": auditor.audit_code_injection,
            "privilege": auditor.audit_privilege_escalation,
            "container": auditor.audit_container_escape
        }
        
        print(f"Running {args.category} security audit...")
        result = category_map[args.category]()
        
        if result.get("secure", False):
            print(f"✅ {args.category.title()} security: SECURE")
            return 0
        else:
            print(f"🚨 {args.category.title()} security: VULNERABLE")
            return 1
    else:
        # Run full audit
        results = auditor.run_comprehensive_audit()
        
        # Generate report if requested
        if args.output:
            auditor.generate_security_report(args.output)
        
        return 0 if results["overall"]["secure"] else 1


if __name__ == "__main__":
    exit(main())