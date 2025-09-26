# Security Best Practices Guide

## Overview

This document provides comprehensive security best practices for the single_turn_scenarios evaluation task. It covers secure deployment, configuration, monitoring, and incident response procedures.

**Requirements**: 12.4, 12.5

## Table of Contents

1. [Deployment Security](#deployment-security)
2. [Configuration Security](#configuration-security)
3. [Code Execution Security](#code-execution-security)
4. [Monitoring and Logging](#monitoring-and-logging)
5. [Incident Response](#incident-response)
6. [Regular Maintenance](#regular-maintenance)
7. [Compliance and Auditing](#compliance-and-auditing)

## Deployment Security

### Environment Isolation

**Principle**: Run evaluations in completely isolated environments to prevent any potential security breaches from affecting the host system.

#### Docker Configuration
```bash
# Use minimal base images
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 evaluser
USER evaluser

# Disable network access (except for whitelisted connections)
docker run --network=none --read-only --tmpfs /tmp evaluation_container

# Set resource limits
docker run --memory=512m --cpus=1.0 --ulimit nofile=1024:1024 evaluation_container
```

#### Network Security
- **Default Deny**: Block all network access by default
- **Whitelist Only**: Allow only explicitly approved connections
- **DNS Filtering**: Block or filter DNS resolution
- **Firewall Rules**: Implement strict firewall rules for evaluation hosts

```bash
# Example iptables rules for evaluation host
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A INPUT -i lo -j ACCEPT
# Add specific rules for required services only
```

### File System Security

#### Directory Structure
```
/opt/evaluation/
├── sandbox/          # Temporary execution directories (tmpfs)
├── problems/         # Read-only problem data
├── configs/          # Read-only configuration files
├── logs/            # Write-only log directory
└── results/         # Write-only results directory
```

#### Permissions
```bash
# Set strict permissions
chmod 755 /opt/evaluation
chmod 644 /opt/evaluation/problems/*
chmod 600 /opt/evaluation/configs/*
chmod 700 /opt/evaluation/sandbox
chmod 755 /opt/evaluation/logs
chmod 755 /opt/evaluation/results

# Use tmpfs for temporary directories
mount -t tmpfs -o size=100M,noexec,nosuid,nodev tmpfs /opt/evaluation/sandbox
```

## Configuration Security

### API Key Management

#### Environment Variables
```bash
# Use .env file (never commit to repository)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...

# Set proper file permissions
chmod 600 .env
chown evaluser:evaluser .env
```

#### Key Rotation
- **Regular Rotation**: Rotate API keys every 90 days
- **Automated Alerts**: Set up alerts for key expiration
- **Backup Keys**: Maintain backup keys for continuity
- **Audit Trail**: Log all key usage and rotation events

#### Secure Storage
```python
# Use secure key management services when available
import keyring
import os

def get_api_key(service_name):
    """Get API key from secure storage."""
    # Try keyring first
    key = keyring.get_password("lm_eval", service_name)
    if key:
        return key
    
    # Fallback to environment variable
    env_var = f"{service_name.upper()}_API_KEY"
    return os.getenv(env_var)
```

### Configuration Validation

#### Schema Validation
```python
# Validate all configuration files
def validate_config(config_data):
    """Validate configuration against schema."""
    required_fields = ['model_name', 'endpoint_config', 'generation_params']
    
    for field in required_fields:
        if field not in config_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate sensitive data is not hardcoded
    for key, value in config_data.items():
        if is_sensitive_key(key) and not value.startswith('${'):
            raise ValueError(f"Sensitive data should use environment variables: {key}")
```

#### Security Checks
- **No Hardcoded Secrets**: Ensure no API keys or passwords in config files
- **Proper Permissions**: Verify file permissions are restrictive
- **Input Validation**: Validate all configuration inputs
- **Default Security**: Use secure defaults for all settings

## Code Execution Security

### Sandbox Configuration

#### Container Security
```dockerfile
# Secure Dockerfile example
FROM python:3.11-slim

# Install security updates
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Create non-root user
RUN groupadd -r evalgroup && useradd -r -g evalgroup evaluser

# Remove unnecessary packages
RUN apt-get remove -y wget curl && apt-get autoremove -y

# Set security limits
RUN echo "evaluser soft nproc 100" >> /etc/security/limits.conf
RUN echo "evaluser hard nproc 100" >> /etc/security/limits.conf
RUN echo "evaluser soft nofile 1024" >> /etc/security/limits.conf
RUN echo "evaluser hard nofile 1024" >> /etc/security/limits.conf

USER evaluser
WORKDIR /tmp/evaluation
```

#### Runtime Security
```python
import subprocess
import signal
import os
import resource

def secure_execute(code, language, timeout=30):
    """Execute code with security restrictions."""
    
    # Set resource limits
    def set_limits():
        # CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        # Memory limit (100MB)
        resource.setrlimit(resource.RLIMIT_AS, (100*1024*1024, 100*1024*1024))
        # File size limit (10MB)
        resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))
        # Number of processes
        resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
    
    try:
        # Execute with restrictions
        process = subprocess.Popen(
            [get_interpreter(language), '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=set_limits,
            cwd='/tmp/evaluation',
            env={'PATH': '/usr/bin:/bin'}  # Minimal environment
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        return stdout, stderr, process.returncode
        
    except subprocess.TimeoutExpired:
        process.kill()
        raise SecurityViolation("Execution timeout exceeded")
    except Exception as e:
        raise SecurityViolation(f"Execution failed: {e}")
```

### Static Analysis

#### Pre-execution Scanning
```python
import re
from typing import List, Tuple

class SecurityScanner:
    """Static security analysis for code."""
    
    DANGEROUS_PATTERNS = {
        'python': [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'os\.system\s*\(',
            r'subprocess\.',
            r'__import__\s*\(',
            r'open\s*\([^)]*[\'"][^\'"]*/etc/',
        ],
        'javascript': [
            r'\beval\s*\(',
            r'Function\s*\(',
            r'require\s*\([^)]*[\'"]fs[\'"]',
            r'require\s*\([^)]*[\'"]child_process[\'"]',
        ]
    }
    
    def scan_code(self, code: str, language: str) -> List[Tuple[str, str]]:
        """Scan code for security violations."""
        violations = []
        
        if language not in self.DANGEROUS_PATTERNS:
            return violations
        
        for pattern in self.DANGEROUS_PATTERNS[language]:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                violations.append((pattern, match.group()))
        
        return violations
```

### Dynamic Monitoring

#### Runtime Monitoring
```python
import psutil
import threading
import time

class RuntimeMonitor:
    """Monitor code execution for security violations."""
    
    def __init__(self, process_id, limits):
        self.process_id = process_id
        self.limits = limits
        self.violations = []
        self.monitoring = True
    
    def monitor(self):
        """Monitor process resource usage."""
        try:
            process = psutil.Process(self.process_id)
            
            while self.monitoring:
                # Check memory usage
                memory_mb = process.memory_info().rss / 1024 / 1024
                if memory_mb > self.limits['memory_mb']:
                    self.violations.append(f"Memory limit exceeded: {memory_mb}MB")
                    process.terminate()
                    break
                
                # Check CPU usage
                cpu_percent = process.cpu_percent()
                if cpu_percent > self.limits['cpu_percent']:
                    self.violations.append(f"CPU limit exceeded: {cpu_percent}%")
                
                # Check file descriptors
                num_fds = process.num_fds()
                if num_fds > self.limits['max_fds']:
                    self.violations.append(f"File descriptor limit exceeded: {num_fds}")
                
                time.sleep(0.1)
                
        except psutil.NoSuchProcess:
            pass  # Process ended normally
        except Exception as e:
            self.violations.append(f"Monitoring error: {e}")
```

## Monitoring and Logging

### Security Event Logging

#### Log Configuration
```python
import logging
import json
from datetime import datetime

# Configure security logger
security_logger = logging.getLogger('security')
security_handler = logging.FileHandler('/var/log/evaluation/security.log')
security_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
security_handler.setFormatter(security_formatter)
security_logger.addHandler(security_handler)
security_logger.setLevel(logging.INFO)

def log_security_event(event_type, severity, details):
    """Log security event in structured format."""
    event = {
        'timestamp': datetime.utcnow().isoformat(),
        'event_type': event_type,
        'severity': severity,
        'details': details,
        'source': 'evaluation_system'
    }
    
    security_logger.info(json.dumps(event))
```

#### Event Types to Log
- **Authentication Events**: API key usage, validation failures
- **Authorization Events**: Access attempts, permission denials
- **Execution Events**: Code execution start/stop, resource usage
- **Security Violations**: Pattern matches, runtime violations
- **System Events**: Container creation/destruction, file access
- **Configuration Changes**: Config updates, key rotations

### Alerting

#### Real-time Alerts
```python
import smtplib
from email.mime.text import MIMEText

class SecurityAlerter:
    """Send security alerts for critical events."""
    
    def __init__(self, smtp_config):
        self.smtp_config = smtp_config
    
    def send_alert(self, severity, message, details=None):
        """Send security alert email."""
        if severity in ['HIGH', 'CRITICAL']:
            subject = f"[SECURITY ALERT] {severity}: {message}"
            body = f"Security Alert\n\nSeverity: {severity}\nMessage: {message}\n"
            
            if details:
                body += f"\nDetails:\n{json.dumps(details, indent=2)}"
            
            self._send_email(subject, body)
    
    def _send_email(self, subject, body):
        """Send email notification."""
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.smtp_config['from']
        msg['To'] = self.smtp_config['to']
        
        with smtplib.SMTP(self.smtp_config['server']) as server:
            server.send_message(msg)
```

### Metrics Collection

#### Security Metrics
```python
class SecurityMetrics:
    """Collect and track security metrics."""
    
    def __init__(self):
        self.metrics = {
            'total_evaluations': 0,
            'security_violations': 0,
            'blocked_executions': 0,
            'false_positives': 0,
            'response_times': [],
            'resource_usage': []
        }
    
    def record_evaluation(self, duration, violations, resources):
        """Record evaluation metrics."""
        self.metrics['total_evaluations'] += 1
        self.metrics['security_violations'] += len(violations)
        self.metrics['response_times'].append(duration)
        self.metrics['resource_usage'].append(resources)
        
        if violations:
            self.metrics['blocked_executions'] += 1
    
    def get_security_score(self):
        """Calculate security score (0-100)."""
        if self.metrics['total_evaluations'] == 0:
            return 100
        
        violation_rate = self.metrics['security_violations'] / self.metrics['total_evaluations']
        return max(0, 100 - (violation_rate * 100))
```

## Incident Response

### Incident Classification

#### Severity Levels
- **CRITICAL**: System compromise, data breach, service disruption
- **HIGH**: Security violation, unauthorized access attempt
- **MEDIUM**: Policy violation, suspicious activity
- **LOW**: Configuration issue, minor security concern

#### Response Procedures

**CRITICAL Incidents**:
1. **Immediate**: Stop all evaluations, isolate affected systems
2. **Within 15 minutes**: Notify security team, preserve evidence
3. **Within 1 hour**: Begin investigation, implement containment
4. **Within 4 hours**: Complete initial assessment, notify stakeholders
5. **Within 24 hours**: Implement fixes, restore service if safe

**HIGH Incidents**:
1. **Immediate**: Log incident, block violating code/model
2. **Within 1 hour**: Investigate violation, assess impact
3. **Within 4 hours**: Implement additional controls if needed
4. **Within 24 hours**: Complete investigation, update procedures

### Incident Response Team

#### Roles and Responsibilities
- **Incident Commander**: Overall response coordination
- **Security Analyst**: Technical investigation and analysis
- **System Administrator**: System isolation and recovery
- **Communications Lead**: Stakeholder notifications
- **Legal/Compliance**: Regulatory requirements and notifications

#### Contact Information
```yaml
incident_response:
  commander:
    name: "Security Team Lead"
    phone: "+1-555-0123"
    email: "security-lead@organization.com"
  
  analyst:
    name: "Security Analyst"
    phone: "+1-555-0124"
    email: "security-analyst@organization.com"
  
  escalation:
    - "CISO: +1-555-0100"
    - "CTO: +1-555-0101"
    - "Legal: +1-555-0102"
```

## Regular Maintenance

### Security Updates

#### Automated Updates
```bash
#!/bin/bash
# Security update script

# Update base system
apt-get update && apt-get upgrade -y

# Update Docker images
docker pull python:3.11-slim
docker pull node:18-slim
docker pull openjdk:17-slim

# Update Python packages
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Rebuild evaluation containers
docker build -t evaluation-python ./docker/python/
docker build -t evaluation-node ./docker/node/
docker build -t evaluation-java ./docker/java/

# Test security configurations
python security_monitor.py --test
```

#### Update Schedule
- **Daily**: Security patches for critical vulnerabilities
- **Weekly**: Regular system updates and dependency updates
- **Monthly**: Container image updates and security reviews
- **Quarterly**: Comprehensive security audit and penetration testing

### Backup and Recovery

#### Backup Strategy
```bash
#!/bin/bash
# Backup script for evaluation system

BACKUP_DIR="/backup/evaluation/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup configurations
tar -czf "$BACKUP_DIR/configs.tar.gz" /opt/evaluation/configs/

# Backup security logs
tar -czf "$BACKUP_DIR/security_logs.tar.gz" /var/log/evaluation/

# Backup compliance records
tar -czf "$BACKUP_DIR/compliance.tar.gz" /opt/evaluation/compliance/

# Encrypt backups
gpg --cipher-algo AES256 --compress-algo 1 --symmetric \
    --output "$BACKUP_DIR/backup.gpg" "$BACKUP_DIR"/*.tar.gz

# Clean up unencrypted files
rm "$BACKUP_DIR"/*.tar.gz
```

#### Recovery Procedures
1. **System Recovery**: Restore from clean system image
2. **Configuration Recovery**: Restore configurations from encrypted backup
3. **Data Recovery**: Restore evaluation data and logs
4. **Security Validation**: Verify all security controls are functioning
5. **Service Restoration**: Gradually restore evaluation services

## Compliance and Auditing

### Compliance Requirements

#### Data Protection
- **GDPR**: If processing EU personal data
- **CCPA**: If processing California resident data
- **SOC 2**: For service organization controls
- **ISO 27001**: For information security management

#### Security Standards
- **NIST Cybersecurity Framework**: Risk management approach
- **OWASP Top 10**: Web application security risks
- **CIS Controls**: Critical security controls implementation

### Audit Procedures

#### Internal Audits
```python
# Automated compliance checking
def run_compliance_audit():
    """Run automated compliance checks."""
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'checks': []
    }
    
    # Check API key security
    api_key_check = check_api_key_security()
    results['checks'].append({
        'name': 'API Key Security',
        'status': 'PASS' if api_key_check else 'FAIL',
        'details': api_key_check
    })
    
    # Check file permissions
    permission_check = check_file_permissions()
    results['checks'].append({
        'name': 'File Permissions',
        'status': 'PASS' if permission_check else 'FAIL',
        'details': permission_check
    })
    
    # Check security configurations
    config_check = check_security_configurations()
    results['checks'].append({
        'name': 'Security Configurations',
        'status': 'PASS' if config_check else 'FAIL',
        'details': config_check
    })
    
    return results
```

#### External Audits
- **Annual**: Third-party security assessment
- **Bi-annual**: Penetration testing
- **As needed**: Compliance audits for certifications

### Documentation Requirements

#### Security Documentation
- **Security Policies**: Comprehensive security policies and procedures
- **Risk Assessments**: Regular risk assessments and mitigation plans
- **Incident Reports**: Detailed incident response documentation
- **Training Records**: Security awareness training completion
- **Audit Reports**: Internal and external audit findings and remediation

#### Change Management
- **Change Requests**: All security-related changes must be documented
- **Approval Process**: Security changes require appropriate approvals
- **Testing Requirements**: Security testing before deployment
- **Rollback Procedures**: Documented rollback procedures for changes

## Conclusion

Security is a continuous process that requires ongoing attention and improvement. This guide provides a foundation for secure operation of the single_turn_scenarios evaluation task, but should be regularly updated based on:

- **Threat Landscape Changes**: New security threats and vulnerabilities
- **Technology Updates**: Changes in underlying technologies and dependencies
- **Regulatory Changes**: Updates to compliance requirements
- **Lessons Learned**: Insights from incidents and security assessments

Regular review and updates of these security practices ensure the evaluation system remains secure and compliant with evolving requirements.

---

**Document Version**: 1.0  
**Last Updated**: 2025-09-25  
**Next Review**: 2025-12-25  
**Owner**: Security Team  
**Approved By**: CISO