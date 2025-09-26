#!/usr/bin/env python3
"""
API Key Validation and Security Checking Tool

This script validates API keys for model evaluation and performs security checks
to ensure keys are properly configured and not exposed in the repository.

Requirements: 12.1, 12.4
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class APIKeyCheck:
    """Result of an API key validation check"""
    provider: str
    key_name: str
    is_present: bool
    is_valid_format: bool
    is_secure: bool
    issues: List[str]
    
class APIKeyValidator:
    """Validates and checks security of API keys"""
    
    # Known API key patterns for validation
    KEY_PATTERNS = {
        'OPENAI_API_KEY': r'^sk-[A-Za-z0-9]{48}$',
        'ANTHROPIC_API_KEY': r'^sk-ant-[A-Za-z0-9\-_]{95,}$',
        'DEEPSEEK_API_KEY': r'^sk-[A-Za-z0-9]{32,}$',
        'COHERE_API_KEY': r'^[A-Za-z0-9\-_]{40,}$',
        'HUGGINGFACE_API_KEY': r'^hf_[A-Za-z0-9]{37}$'
    }
    
    # Insecure patterns that should never be used
    INSECURE_PATTERNS = [
        r'test',
        r'demo',
        r'example',
        r'placeholder',
        r'your_.*_key_here',
        r'sk-1234',
        r'abc123'
    ]
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize validator with optional custom .env file path"""
        self.env_file = env_file or '.env'
        self.base_dir = Path(__file__).parent
        
    def load_environment(self) -> bool:
        """Load environment variables from .env file"""
        env_path = self.base_dir / self.env_file
        
        if not env_path.exists():
            logger.warning(f"No .env file found at {env_path}")
            logger.info("Using environment variables from system")
            return False
            
        try:
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load .env file: {e}")
            return False
    
    def check_repository_security(self) -> List[str]:
        """Check for API keys accidentally committed to repository"""
        issues = []
        
        # Check if .env file is in .gitignore
        gitignore_path = self.base_dir.parent.parent.parent / '.gitignore'
        if gitignore_path.exists():
            try:
                gitignore_content = gitignore_path.read_text()
                if '.env' not in gitignore_content:
                    issues.append(".env file not found in .gitignore - risk of committing API keys")
            except Exception as e:
                issues.append(f"Could not read .gitignore: {e}")
        else:
            issues.append("No .gitignore found - API keys may be committed to repository")
        
        # Check for hardcoded API keys in Python files
        python_files = list(self.base_dir.rglob('*.py'))
        for py_file in python_files:
            if py_file.name == 'check_api_keys.py':  # Skip this file
                continue
                
            try:
                content = py_file.read_text()
                for pattern_name, pattern in self.KEY_PATTERNS.items():
                    if re.search(pattern, content):
                        issues.append(f"Potential API key found in {py_file.relative_to(self.base_dir)}")
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
        
        return issues
    
    def validate_key_format(self, key_name: str, key_value: str) -> Tuple[bool, List[str]]:
        """Validate API key format and security"""
        issues = []
        is_valid = True
        
        if not key_value:
            issues.append("Key is empty")
            return False, issues
        
        # Check against known patterns
        if key_name in self.KEY_PATTERNS:
            pattern = self.KEY_PATTERNS[key_name]
            if not re.match(pattern, key_value):
                issues.append(f"Key format does not match expected pattern for {key_name}")
                is_valid = False
        
        # Check for insecure patterns
        for insecure_pattern in self.INSECURE_PATTERNS:
            if re.search(insecure_pattern, key_value, re.IGNORECASE):
                issues.append(f"Key contains insecure pattern: {insecure_pattern}")
                is_valid = False
        
        # Basic security checks
        if len(key_value) < 20:
            issues.append("Key is too short (less than 20 characters)")
            is_valid = False
        
        if key_value.isalnum() and len(set(key_value)) < 10:
            issues.append("Key has low entropy (too few unique characters)")
            is_valid = False
        
        return is_valid, issues
    
    def check_key_permissions(self, key_name: str, key_value: str) -> Tuple[bool, List[str]]:
        """Check if API key has appropriate permissions (basic validation)"""
        issues = []
        
        # This is a placeholder for actual API validation
        # In a real implementation, you would make test API calls
        # to verify the key works and has appropriate permissions
        
        if key_name == 'OPENAI_API_KEY':
            # Could test with a simple completion request
            pass
        elif key_name == 'ANTHROPIC_API_KEY':
            # Could test with a simple message request
            pass
        elif key_name == 'DEEPSEEK_API_KEY':
            # Could test with a simple request
            pass
        
        # For now, just check if it's not obviously invalid
        if 'invalid' in key_value.lower() or 'expired' in key_value.lower():
            issues.append("Key appears to be invalid or expired")
            return False, issues
        
        return True, issues
    
    def validate_all_keys(self) -> List[APIKeyCheck]:
        """Validate all configured API keys"""
        results = []
        
        # Load environment
        self.load_environment()
        
        # Check each known API key
        for key_name in self.KEY_PATTERNS.keys():
            key_value = os.getenv(key_name, '')
            
            # Basic presence check
            is_present = bool(key_value)
            
            # Format validation
            is_valid_format = True
            format_issues = []
            if is_present:
                is_valid_format, format_issues = self.validate_key_format(key_name, key_value)
            
            # Security validation
            is_secure = True
            security_issues = []
            if is_present:
                is_secure, security_issues = self.check_key_permissions(key_name, key_value)
            
            # Combine all issues
            all_issues = format_issues + security_issues
            if not is_present:
                all_issues.append("API key not configured")
            
            # Determine provider name
            provider = key_name.replace('_API_KEY', '').lower()
            
            result = APIKeyCheck(
                provider=provider,
                key_name=key_name,
                is_present=is_present,
                is_valid_format=is_valid_format,
                is_secure=is_secure,
                issues=all_issues
            )
            results.append(result)
        
        return results
    
    def generate_security_report(self) -> Dict:
        """Generate comprehensive security report"""
        # Validate API keys
        key_results = self.validate_all_keys()
        
        # Check repository security
        repo_issues = self.check_repository_security()
        
        # Count statistics
        total_keys = len(key_results)
        configured_keys = sum(1 for r in key_results if r.is_present)
        valid_keys = sum(1 for r in key_results if r.is_present and r.is_valid_format and r.is_secure)
        
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'summary': {
                'total_keys_checked': total_keys,
                'configured_keys': configured_keys,
                'valid_keys': valid_keys,
                'security_score': (valid_keys / total_keys * 100) if total_keys > 0 else 0
            },
            'key_results': [
                {
                    'provider': r.provider,
                    'key_name': r.key_name,
                    'status': 'valid' if r.is_present and r.is_valid_format and r.is_secure else 'invalid',
                    'is_present': r.is_present,
                    'is_valid_format': r.is_valid_format,
                    'is_secure': r.is_secure,
                    'issues': r.issues
                }
                for r in key_results
            ],
            'repository_security': {
                'issues': repo_issues,
                'secure': len(repo_issues) == 0
            }
        }
        
        return report

def print_security_report(report: Dict):
    """Print formatted security report to console"""
    print("\n" + "="*60)
    print("API KEY SECURITY REPORT")
    print("="*60)
    
    # Summary
    summary = report['summary']
    print(f"\nSUMMARY:")
    print(f"  Total Keys Checked: {summary['total_keys_checked']}")
    print(f"  Configured Keys: {summary['configured_keys']}")
    print(f"  Valid Keys: {summary['valid_keys']}")
    print(f"  Security Score: {summary['security_score']:.1f}%")
    
    # Key details
    print(f"\nAPI KEY DETAILS:")
    for key_result in report['key_results']:
        status_icon = "✓" if key_result['status'] == 'valid' else "✗"
        print(f"  {status_icon} {key_result['provider'].upper()}: {key_result['status']}")
        
        if key_result['issues']:
            for issue in key_result['issues']:
                print(f"    - {issue}")
    
    # Repository security
    print(f"\nREPOSITORY SECURITY:")
    repo_security = report['repository_security']
    if repo_security['secure']:
        print("  ✓ No security issues detected")
    else:
        print("  ✗ Security issues found:")
        for issue in repo_security['issues']:
            print(f"    - {issue}")
    
    print("\n" + "="*60)

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate API keys and check security')
    parser.add_argument('--env-file', help='Path to .env file (default: .env)')
    parser.add_argument('--output', help='Output report to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output')
    
    args = parser.parse_args()
    
    # Create validator
    validator = APIKeyValidator(args.env_file)
    
    # Generate report
    try:
        report = validator.generate_security_report()
        
        # Output to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(report, indent=2))
            if not args.quiet:
                print(f"Security report saved to {output_path}")
        
        # Print to console unless quiet
        if not args.quiet:
            print_security_report(report)
        
        # Exit with error code if security issues found
        if not report['repository_security']['secure'] or report['summary']['security_score'] < 100:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Failed to generate security report: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()