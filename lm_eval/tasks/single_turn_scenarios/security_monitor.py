#!/usr/bin/env python3
"""
Security monitoring and violation reporting system for single_turn_scenarios.
Implements security violation logging, detection, and reporting mechanisms.

Requirements: 12.4, 12.5
"""

import os
import re
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security violation severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ViolationType(Enum):
    """Types of security violations."""
    DANGEROUS_FUNCTION = "dangerous_function"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM_ACCESS = "file_system_access"
    PROCESS_CONTROL = "process_control"
    RESOURCE_ABUSE = "resource_abuse"
    CODE_INJECTION = "code_injection"
    SYSTEM_COMMAND = "system_command"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_PATTERN = "malicious_pattern"

@dataclass
class SecurityViolation:
    """Record of a security violation."""
    violation_id: str
    timestamp: str
    violation_type: ViolationType
    severity: SecurityLevel
    code_snippet: str
    language: str
    problem_id: Optional[str]
    model_name: Optional[str]
    detection_method: str  # 'static', 'dynamic', 'manual'
    description: str
    mitigation_action: str
    resolved: bool = False
    resolution_notes: Optional[str] = None
    false_positive: bool = False

@dataclass
class SecurityReport:
    """Security monitoring report."""
    report_id: str
    timestamp: str
    period_start: str
    period_end: str
    total_evaluations: int
    total_violations: int
    violations_by_severity: Dict[str, int]
    violations_by_type: Dict[str, int]
    violations_by_language: Dict[str, int]
    top_violations: List[Dict[str, Any]]
    recommendations: List[str]

class SecurityMonitor:
    """Monitors code execution for security violations."""
    
    # Dangerous function patterns by language
    DANGEROUS_PATTERNS = {
        'python': {
            SecurityLevel.CRITICAL: [
                r'\beval\s*\(',
                r'\bexec\s*\(',
                r'\b__import__\s*\(',
                r'os\.system\s*\(',
                r'subprocess\.call\s*\(',
                r'subprocess\.run\s*\(',
                r'subprocess\.Popen\s*\(',
                r'pickle\.loads\s*\(',
                r'marshal\.loads\s*\(',
            ],
            SecurityLevel.HIGH: [
                r'open\s*\(\s*[\'"][^\'"]*/etc/',
                r'open\s*\(\s*[\'"][^\'"]*/proc/',
                r'open\s*\(\s*[\'"][^\'"]*/sys/',
                r'socket\.socket\s*\(',
                r'urllib\.request\.',
                r'requests\.get\s*\(',
                r'requests\.post\s*\(',
                r'os\.remove\s*\(',
                r'os\.rmdir\s*\(',
                r'shutil\.rmtree\s*\(',
            ],
            SecurityLevel.MEDIUM: [
                r'compile\s*\(',
                r'importlib\.',
                r'getattr\s*\(',
                r'setattr\s*\(',
                r'hasattr\s*\(',
                r'globals\s*\(\)',
                r'locals\s*\(\)',
                r'vars\s*\(',
            ],
            SecurityLevel.LOW: [
                r'input\s*\(',
                r'raw_input\s*\(',
                r'os\.environ',
                r'sys\.exit\s*\(',
                r'exit\s*\(',
                r'quit\s*\(',
            ]
        },
        'javascript': {
            SecurityLevel.CRITICAL: [
                r'\beval\s*\(',
                r'Function\s*\(',
                r'require\s*\(\s*[\'"]child_process[\'"]',
                r'require\s*\(\s*[\'"]fs[\'"]',
                r'require\s*\(\s*[\'"]net[\'"]',
                r'require\s*\(\s*[\'"]http[\'"]',
                r'require\s*\(\s*[\'"]https[\'"]',
            ],
            SecurityLevel.HIGH: [
                r'document\.write\s*\(',
                r'innerHTML\s*=',
                r'outerHTML\s*=',
                r'document\.createElement\s*\(',
                r'XMLHttpRequest\s*\(',
                r'fetch\s*\(',
            ],
            SecurityLevel.MEDIUM: [
                r'setTimeout\s*\(',
                r'setInterval\s*\(',
                r'console\.log\s*\(',
                r'alert\s*\(',
                r'confirm\s*\(',
                r'prompt\s*\(',
            ]
        },
        'java': {
            SecurityLevel.CRITICAL: [
                r'Runtime\.getRuntime\(\)\.exec\s*\(',
                r'ProcessBuilder\s*\(',
                r'Class\.forName\s*\(',
                r'System\.exit\s*\(',
                r'ObjectInputStream\.readObject\s*\(',
            ],
            SecurityLevel.HIGH: [
                r'FileInputStream\s*\(',
                r'FileOutputStream\s*\(',
                r'Socket\s*\(',
                r'ServerSocket\s*\(',
                r'URL\s*\(',
                r'URLConnection\s*\(',
            ],
            SecurityLevel.MEDIUM: [
                r'System\.getProperty\s*\(',
                r'System\.setProperty\s*\(',
                r'Thread\s*\(',
                r'Runnable\s*\(',
            ]
        },
        'cpp': {
            SecurityLevel.CRITICAL: [
                r'\bsystem\s*\(',
                r'\bpopen\s*\(',
                r'\bexec[lv]p?\s*\(',
                r'\bfork\s*\(',
                r'#include\s*<cstdlib>',
                r'#include\s*<stdlib\.h>',
            ],
            SecurityLevel.HIGH: [
                r'\bfopen\s*\(',
                r'\bfwrite\s*\(',
                r'\bfread\s*\(',
                r'\bmalloc\s*\(',
                r'\bcalloc\s*\(',
                r'\brealloc\s*\(',
                r'\bfree\s*\(',
            ],
            SecurityLevel.MEDIUM: [
                r'\bprintf\s*\(',
                r'\bsprintf\s*\(',
                r'\bstrcpy\s*\(',
                r'\bstrcat\s*\(',
                r'\bgets\s*\(',
            ]
        },
        'go': {
            SecurityLevel.CRITICAL: [
                r'os/exec',
                r'exec\.Command\s*\(',
                r'syscall\.',
                r'unsafe\.',
            ],
            SecurityLevel.HIGH: [
                r'os\.Open\s*\(',
                r'os\.Create\s*\(',
                r'net\.',
                r'http\.',
            ],
            SecurityLevel.MEDIUM: [
                r'os\.Getenv\s*\(',
                r'os\.Setenv\s*\(',
                r'log\.',
                r'fmt\.Print',
            ]
        },
        'rust': {
            SecurityLevel.CRITICAL: [
                r'std::process::Command',
                r'unsafe\s*\{',
                r'libc::', 
                r'std::ffi::',
            ],
            SecurityLevel.HIGH: [
                r'std::fs::',
                r'std::net::',
                r'std::io::',
                r'std::env::',
            ],
            SecurityLevel.MEDIUM: [
                r'println!\s*\(',
                r'print!\s*\(',
                r'panic!\s*\(',
                r'unreachable!\s*\(',
            ]
        }
    }
    
    # Network-related patterns
    NETWORK_PATTERNS = [
        r'socket\.',
        r'urllib\.',
        r'requests\.',
        r'http\.',
        r'https\.',
        r'ftp\.',
        r'smtp\.',
        r'telnet\.',
        r'ssh\.',
        r'ping\s+',
        r'curl\s+',
        r'wget\s+',
        r'nc\s+',
        r'netcat\s+',
    ]
    
    # File system escape patterns
    FILE_ESCAPE_PATTERNS = [
        r'\.\./\.\.',
        r'/etc/',
        r'/proc/',
        r'/sys/',
        r'/dev/',
        r'/root/',
        r'/home/',
        r'C:\\Windows',
        r'C:\\Users',
        r'C:\\Program Files',
    ]
    
    # Malicious code patterns
    MALICIOUS_PATTERNS = [
        r'rm\s+-rf\s+/',
        r'del\s+/[qsf]',
        r'format\s+c:',
        r'dd\s+if=/dev/zero',
        r':(){ :|:& };:',  # Fork bomb
        r'while\s*\(\s*true\s*\)',  # Infinite loop
        r'for\s*\(\s*;\s*;\s*\)',  # Infinite loop
    ]
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize security monitor.
        
        Args:
            log_dir: Directory for security logs (defaults to security_logs/).
        """
        if log_dir is None:
            self.log_dir = Path(__file__).parent / "security_logs"
        else:
            self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(exist_ok=True)
        
        self.violations_file = self.log_dir / "violations.json"
        self.reports_file = self.log_dir / "reports.json"
        
        self.violations: List[SecurityViolation] = []
        self.reports: List[SecurityReport] = []
        
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing violations and reports."""
        # Load violations
        if self.violations_file.exists():
            try:
                with open(self.violations_file, 'r') as f:
                    data = json.load(f)
                    self.violations = [
                        SecurityViolation(
                            violation_type=ViolationType(v['violation_type']),
                            severity=SecurityLevel(v['severity']),
                            **{k: v for k, v in v.items() if k not in ['violation_type', 'severity']}
                        )
                        for v in data
                    ]
            except Exception as e:
                logger.warning(f"Failed to load violations: {e}")
        
        # Load reports
        if self.reports_file.exists():
            try:
                with open(self.reports_file, 'r') as f:
                    data = json.load(f)
                    self.reports = [SecurityReport(**r) for r in data]
            except Exception as e:
                logger.warning(f"Failed to load reports: {e}")
    
    def _save_data(self):
        """Save violations and reports to files."""
        try:
            # Save violations
            violations_data = []
            for v in self.violations:
                v_dict = asdict(v)
                v_dict['violation_type'] = v.violation_type.value
                v_dict['severity'] = v.severity.value
                violations_data.append(v_dict)
            
            with open(self.violations_file, 'w') as f:
                json.dump(violations_data, f, indent=2)
            
            # Save reports
            reports_data = [asdict(r) for r in self.reports]
            with open(self.reports_file, 'w') as f:
                json.dump(reports_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save security data: {e}")
    
    def _generate_violation_id(self, code_snippet: str) -> str:
        """Generate unique violation ID based on code content."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        code_hash = hashlib.md5(code_snippet.encode()).hexdigest()[:8]
        return f"SEC_{timestamp}_{code_hash}"
    
    def scan_code_static(self, 
                        code: str, 
                        language: str,
                        problem_id: Optional[str] = None,
                        model_name: Optional[str] = None) -> List[SecurityViolation]:
        """Perform static analysis of code for security violations.
        
        Args:
            code: Code to analyze.
            language: Programming language.
            problem_id: Problem identifier (if applicable).
            model_name: Model name (if applicable).
        
        Returns:
            List of detected security violations.
        """
        violations = []
        language = language.lower()
        
        if language not in self.DANGEROUS_PATTERNS:
            logger.warning(f"No security patterns defined for language: {language}")
            return violations
        
        # Check dangerous function patterns
        for severity, patterns in self.DANGEROUS_PATTERNS[language].items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    violation = SecurityViolation(
                        violation_id=self._generate_violation_id(code),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        violation_type=ViolationType.DANGEROUS_FUNCTION,
                        severity=severity,
                        code_snippet=self._extract_code_context(code, match.start(), match.end()),
                        language=language,
                        problem_id=problem_id,
                        model_name=model_name,
                        detection_method='static',
                        description=f"Dangerous function pattern detected: {pattern}",
                        mitigation_action="Code execution blocked"
                    )
                    violations.append(violation)
        
        # Check network access patterns
        for pattern in self.NETWORK_PATTERNS:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                violation = SecurityViolation(
                    violation_id=self._generate_violation_id(code),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    violation_type=ViolationType.NETWORK_ACCESS,
                    severity=SecurityLevel.HIGH,
                    code_snippet=self._extract_code_context(code, match.start(), match.end()),
                    language=language,
                    problem_id=problem_id,
                    model_name=model_name,
                    detection_method='static',
                    description=f"Network access pattern detected: {pattern}",
                    mitigation_action="Network access blocked"
                )
                violations.append(violation)
        
        # Check file system escape patterns
        for pattern in self.FILE_ESCAPE_PATTERNS:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                violation = SecurityViolation(
                    violation_id=self._generate_violation_id(code),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    violation_type=ViolationType.FILE_SYSTEM_ACCESS,
                    severity=SecurityLevel.CRITICAL,
                    code_snippet=self._extract_code_context(code, match.start(), match.end()),
                    language=language,
                    problem_id=problem_id,
                    model_name=model_name,
                    detection_method='static',
                    description=f"File system escape pattern detected: {pattern}",
                    mitigation_action="File access blocked"
                )
                violations.append(violation)
        
        # Check malicious patterns
        for pattern in self.MALICIOUS_PATTERNS:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                violation = SecurityViolation(
                    violation_id=self._generate_violation_id(code),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    violation_type=ViolationType.MALICIOUS_PATTERN,
                    severity=SecurityLevel.CRITICAL,
                    code_snippet=self._extract_code_context(code, match.start(), match.end()),
                    language=language,
                    problem_id=problem_id,
                    model_name=model_name,
                    detection_method='static',
                    description=f"Malicious code pattern detected: {pattern}",
                    mitigation_action="Code execution blocked"
                )
                violations.append(violation)
        
        # Store violations
        self.violations.extend(violations)
        self._save_data()
        
        return violations
    
    def _extract_code_context(self, code: str, start: int, end: int, context_lines: int = 2) -> str:
        """Extract code context around a match.
        
        Args:
            code: Full code string.
            start: Start position of match.
            end: End position of match.
            context_lines: Number of context lines to include.
        
        Returns:
            Code snippet with context.
        """
        lines = code.split('\n')
        
        # Find line numbers
        char_count = 0
        start_line = 0
        end_line = 0
        
        for i, line in enumerate(lines):
            if char_count <= start <= char_count + len(line):
                start_line = i
            if char_count <= end <= char_count + len(line):
                end_line = i
                break
            char_count += len(line) + 1  # +1 for newline
        
        # Extract context
        context_start = max(0, start_line - context_lines)
        context_end = min(len(lines), end_line + context_lines + 1)
        
        context_lines_list = lines[context_start:context_end]
        
        # Mark the problematic lines
        for i in range(len(context_lines_list)):
            line_num = context_start + i
            if start_line <= line_num <= end_line:
                context_lines_list[i] = f">>> {context_lines_list[i]}"
            else:
                context_lines_list[i] = f"    {context_lines_list[i]}"
        
        return '\n'.join(context_lines_list)
    
    def report_runtime_violation(self,
                                violation_type: ViolationType,
                                severity: SecurityLevel,
                                description: str,
                                code_snippet: str,
                                language: str,
                                problem_id: Optional[str] = None,
                                model_name: Optional[str] = None) -> SecurityViolation:
        """Report a runtime security violation.
        
        Args:
            violation_type: Type of violation.
            severity: Severity level.
            description: Description of the violation.
            code_snippet: Relevant code snippet.
            language: Programming language.
            problem_id: Problem identifier (if applicable).
            model_name: Model name (if applicable).
        
        Returns:
            SecurityViolation record.
        """
        violation = SecurityViolation(
            violation_id=self._generate_violation_id(code_snippet),
            timestamp=datetime.now(timezone.utc).isoformat(),
            violation_type=violation_type,
            severity=severity,
            code_snippet=code_snippet,
            language=language,
            problem_id=problem_id,
            model_name=model_name,
            detection_method='dynamic',
            description=description,
            mitigation_action="Execution terminated"
        )
        
        self.violations.append(violation)
        self._save_data()
        
        logger.warning(f"Runtime security violation: {violation.violation_id}")
        return violation
    
    def mark_false_positive(self, violation_id: str, reviewer: str, notes: str):
        """Mark a violation as a false positive.
        
        Args:
            violation_id: Violation identifier.
            reviewer: Person marking as false positive.
            notes: Explanation notes.
        """
        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.false_positive = True
                violation.resolved = True
                violation.resolution_notes = f"False positive marked by {reviewer}: {notes}"
                self._save_data()
                logger.info(f"Violation {violation_id} marked as false positive")
                return
        
        logger.warning(f"Violation {violation_id} not found")
    
    def resolve_violation(self, violation_id: str, resolver: str, notes: str):
        """Mark a violation as resolved.
        
        Args:
            violation_id: Violation identifier.
            resolver: Person resolving the violation.
            notes: Resolution notes.
        """
        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.resolved = True
                violation.resolution_notes = f"Resolved by {resolver}: {notes}"
                self._save_data()
                logger.info(f"Violation {violation_id} resolved")
                return
        
        logger.warning(f"Violation {violation_id} not found")
    
    def generate_security_report(self, 
                                period_days: int = 30,
                                include_resolved: bool = False) -> SecurityReport:
        """Generate security monitoring report.
        
        Args:
            period_days: Number of days to include in report.
            include_resolved: Whether to include resolved violations.
        
        Returns:
            SecurityReport with analysis and recommendations.
        """
        # Filter violations by time period
        cutoff_date = datetime.now(timezone.utc).timestamp() - (period_days * 24 * 3600)
        
        relevant_violations = [
            v for v in self.violations
            if datetime.fromisoformat(v.timestamp.replace('Z', '+00:00')).timestamp() >= cutoff_date
            and (include_resolved or not v.resolved)
            and not v.false_positive
        ]
        
        # Count violations by severity
        violations_by_severity = {level.value: 0 for level in SecurityLevel}
        for violation in relevant_violations:
            violations_by_severity[violation.severity.value] += 1
        
        # Count violations by type
        violations_by_type = {vtype.value: 0 for vtype in ViolationType}
        for violation in relevant_violations:
            violations_by_type[violation.violation_type.value] += 1
        
        # Count violations by language
        violations_by_language = {}
        for violation in relevant_violations:
            lang = violation.language
            violations_by_language[lang] = violations_by_language.get(lang, 0) + 1
        
        # Top violations (most frequent patterns)
        violation_patterns = {}
        for violation in relevant_violations:
            pattern = f"{violation.violation_type.value}_{violation.severity.value}"
            if pattern not in violation_patterns:
                violation_patterns[pattern] = {
                    'pattern': pattern,
                    'count': 0,
                    'severity': violation.severity.value,
                    'type': violation.violation_type.value,
                    'description': violation.description
                }
            violation_patterns[pattern]['count'] += 1
        
        top_violations = sorted(
            violation_patterns.values(),
            key=lambda x: x['count'],
            reverse=True
        )[:10]
        
        # Generate recommendations
        recommendations = []
        
        if violations_by_severity['critical'] > 0:
            recommendations.append("Critical security violations detected - immediate review required")
        
        if violations_by_severity['high'] > 5:
            recommendations.append("High number of high-severity violations - review security policies")
        
        if violations_by_type['network_access'] > 0:
            recommendations.append("Network access attempts detected - verify sandbox isolation")
        
        if violations_by_type['file_system_access'] > 0:
            recommendations.append("File system access violations - check sandbox file permissions")
        
        if len(violations_by_language) > 3:
            recommendations.append("Violations across multiple languages - review language-specific security patterns")
        
        # Create report
        report = SecurityReport(
            report_id=f"SEC_RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            period_start=(datetime.now(timezone.utc) - datetime.timedelta(days=period_days)).isoformat(),
            period_end=datetime.now(timezone.utc).isoformat(),
            total_evaluations=0,  # This would need to be tracked separately
            total_violations=len(relevant_violations),
            violations_by_severity=violations_by_severity,
            violations_by_type=violations_by_type,
            violations_by_language=violations_by_language,
            top_violations=top_violations,
            recommendations=recommendations
        )
        
        self.reports.append(report)
        self._save_data()
        
        return report
    
    def export_violations_csv(self, output_file: Path, include_resolved: bool = False):
        """Export violations to CSV file.
        
        Args:
            output_file: Output CSV file path.
            include_resolved: Whether to include resolved violations.
        """
        import csv
        
        violations_to_export = [
            v for v in self.violations
            if (include_resolved or not v.resolved) and not v.false_positive
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Violation ID', 'Timestamp', 'Type', 'Severity', 'Language',
                'Problem ID', 'Model', 'Detection Method', 'Description',
                'Resolved', 'False Positive'
            ])
            
            # Data
            for violation in violations_to_export:
                writer.writerow([
                    violation.violation_id,
                    violation.timestamp,
                    violation.violation_type.value,
                    violation.severity.value,
                    violation.language,
                    violation.problem_id or '',
                    violation.model_name or '',
                    violation.detection_method,
                    violation.description,
                    violation.resolved,
                    violation.false_positive
                ])

def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Security monitoring and reporting tool')
    parser.add_argument('--scan', help='Scan code file for security violations')
    parser.add_argument('--language', help='Programming language for scanning')
    parser.add_argument('--report', action='store_true', help='Generate security report')
    parser.add_argument('--days', type=int, default=30, help='Report period in days')
    parser.add_argument('--export-csv', help='Export violations to CSV file')
    parser.add_argument('--output', help='Output file for reports')
    
    args = parser.parse_args()
    
    monitor = SecurityMonitor()
    
    if args.scan and args.language:
        code_file = Path(args.scan)
        if code_file.exists():
            code = code_file.read_text()
            violations = monitor.scan_code_static(code, args.language)
            print(f"Found {len(violations)} security violations")
            for violation in violations:
                print(f"- {violation.severity.value.upper()}: {violation.description}")
        else:
            print(f"Code file not found: {args.scan}")
    
    if args.report:
        report = monitor.generate_security_report(args.days)
        print(f"Security Report: {report.report_id}")
        print(f"Period: {report.period_start[:10]} to {report.period_end[:10]}")
        print(f"Total Violations: {report.total_violations}")
        print(f"By Severity: {report.violations_by_severity}")
        print(f"Recommendations: {len(report.recommendations)}")
        
        if args.output:
            report_text = json.dumps(asdict(report), indent=2)
            Path(args.output).write_text(report_text)
            print(f"Report saved to {args.output}")
    
    if args.export_csv:
        monitor.export_violations_csv(Path(args.export_csv))
        print(f"Violations exported to {args.export_csv}")

if __name__ == '__main__':
    main()