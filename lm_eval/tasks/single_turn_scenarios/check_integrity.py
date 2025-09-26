#!/usr/bin/env python3
"""
Integrity checker for single_turn_scenarios problems and tests.
Performs comprehensive validation of the entire task suite.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return "ERROR"

def check_problems_integrity() -> Dict:
    """Check integrity of problems.jsonl file."""
    problems_file = Path(__file__).parent / "problems.jsonl"
    
    if not problems_file.exists():
        return {"status": "ERROR", "message": "problems.jsonl not found"}
    
    try:
        problems = []
        with open(problems_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
        
        # Check for required fields and consistency
        required_fields = {'id', 'title', 'language', 'scenario', 'difficulty', 
                          'context_mode', 'prompt', 'reference', 'tests', 'metadata'}
        
        issues = []
        for i, problem in enumerate(problems):
            missing_fields = required_fields - set(problem.keys())
            if missing_fields:
                issues.append(f"Problem {i+1}: Missing fields {missing_fields}")
            
            # Check metadata completeness
            if 'metadata' in problem:
                metadata = problem['metadata']
                required_metadata = {'time_limit_s', 'memory_limit_mb', 'seed', 'author', 'license'}
                missing_metadata = required_metadata - set(metadata.keys())
                if missing_metadata:
                    issues.append(f"Problem {i+1}: Missing metadata {missing_metadata}")
        
        return {
            "status": "OK" if not issues else "WARNING",
            "problems_count": len(problems),
            "file_hash": calculate_file_hash(problems_file),
            "issues": issues
        }
        
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def check_tests_integrity() -> Dict:
    """Check integrity of test files."""
    tests_dir = Path(__file__).parent / "tests"
    
    if not tests_dir.exists():
        return {"status": "ERROR", "message": "tests directory not found"}
    
    test_files = list(tests_dir.glob("test_*.py")) + \
                 list(tests_dir.glob("test_*.js")) + \
                 list(tests_dir.glob("test_*.java")) + \
                 list(tests_dir.glob("test_*.cpp")) + \
                 list(tests_dir.glob("test_*.go")) + \
                 list(tests_dir.glob("test_*.rs")) + \
                 list(tests_dir.glob("test_*.sql"))
    
    test_info = []
    for test_file in test_files:
        test_info.append({
            "name": test_file.name,
            "size": test_file.stat().st_size,
            "hash": calculate_file_hash(test_file)
        })
    
    return {
        "status": "OK",
        "test_files_count": len(test_files),
        "test_files": test_info
    }

def check_documentation_integrity() -> Dict:
    """Check integrity of documentation files."""
    base_dir = Path(__file__).parent
    
    required_docs = [
        "README.md",
        "LICENSING_COMPLIANCE.md",
        "validation_report.md"
    ]
    
    doc_status = {}
    for doc in required_docs:
        doc_path = base_dir / doc
        if doc_path.exists():
            doc_status[doc] = {
                "exists": True,
                "size": doc_path.stat().st_size,
                "hash": calculate_file_hash(doc_path)
            }
        else:
            doc_status[doc] = {"exists": False}
    
    missing_docs = [doc for doc, status in doc_status.items() if not status["exists"]]
    
    return {
        "status": "OK" if not missing_docs else "WARNING",
        "missing_docs": missing_docs,
        "documentation": doc_status
    }

def check_configuration_integrity() -> Dict:
    """Check integrity of configuration files."""
    base_dir = Path(__file__).parent
    
    config_files = [
        "context_configs.json",
        "single_turn_scenarios_suite.yaml"
    ]
    
    config_status = {}
    for config in config_files:
        config_path = base_dir / config
        if config_path.exists():
            config_status[config] = {
                "exists": True,
                "size": config_path.stat().st_size,
                "hash": calculate_file_hash(config_path)
            }
        else:
            config_status[config] = {"exists": False}
    
    missing_configs = [cfg for cfg, status in config_status.items() if not status["exists"]]
    
    return {
        "status": "OK" if not missing_configs else "WARNING",
        "missing_configs": missing_configs,
        "configurations": config_status
    }

def generate_integrity_report() -> str:
    """Generate comprehensive integrity report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Run all integrity checks
    problems_check = check_problems_integrity()
    tests_check = check_tests_integrity()
    docs_check = check_documentation_integrity()
    config_check = check_configuration_integrity()
    
    # Determine overall status
    all_checks = [problems_check, tests_check, docs_check, config_check]
    has_errors = any(check["status"] == "ERROR" for check in all_checks)
    has_warnings = any(check["status"] == "WARNING" for check in all_checks)
    
    overall_status = "ERROR" if has_errors else ("WARNING" if has_warnings else "OK")
    
    report = f"""# Single Turn Scenarios Integrity Report

**Generated**: {timestamp}  
**Overall Status**: {overall_status}

## Problems Dataset Integrity

**Status**: {problems_check["status"]}
"""
    
    if problems_check["status"] != "ERROR":
        report += f"""
- **Problems Count**: {problems_check.get("problems_count", 0)}
- **File Hash**: {problems_check.get("file_hash", "N/A")}
"""
        if problems_check.get("issues"):
            report += "\n**Issues Found**:\n"
            for issue in problems_check["issues"]:
                report += f"- {issue}\n"
    else:
        report += f"\n**Error**: {problems_check.get('message', 'Unknown error')}\n"
    
    report += f"""
## Test Suite Integrity

**Status**: {tests_check["status"]}
"""
    
    if tests_check["status"] != "ERROR":
        report += f"""
- **Test Files Count**: {tests_check.get("test_files_count", 0)}

### Test Files
"""
        for test_file in tests_check.get("test_files", []):
            size_kb = test_file["size"] / 1024
            report += f"- **{test_file['name']}**: {size_kb:.1f} KB (hash: {test_file['hash'][:8]}...)\n"
    
    report += f"""
## Documentation Integrity

**Status**: {docs_check["status"]}
"""
    
    if docs_check.get("missing_docs"):
        report += "\n**Missing Documentation**:\n"
        for doc in docs_check["missing_docs"]:
            report += f"- {doc}\n"
    
    report += "\n**Documentation Files**:\n"
    for doc, status in docs_check.get("documentation", {}).items():
        if status["exists"]:
            size_kb = status["size"] / 1024
            report += f"- **{doc}**: {size_kb:.1f} KB (hash: {status['hash'][:8]}...)\n"
        else:
            report += f"- **{doc}**: MISSING\n"
    
    report += f"""
## Configuration Integrity

**Status**: {config_check["status"]}
"""
    
    if config_check.get("missing_configs"):
        report += "\n**Missing Configurations**:\n"
        for config in config_check["missing_configs"]:
            report += f"- {config}\n"
    
    report += "\n**Configuration Files**:\n"
    for config, status in config_check.get("configurations", {}).items():
        if status["exists"]:
            size_kb = status["size"] / 1024
            report += f"- **{config}**: {size_kb:.1f} KB (hash: {status['hash'][:8]}...)\n"
        else:
            report += f"- **{config}**: MISSING\n"
    
    # Add recommendations
    report += """
## Recommendations

"""
    
    if overall_status == "OK":
        report += "✅ All integrity checks passed. The task suite is ready for use.\n"
    elif overall_status == "WARNING":
        report += "⚠️ Some issues found but task is functional. Consider addressing warnings.\n"
    else:
        report += "❌ Critical errors found. Task suite requires fixes before use.\n"
    
    report += """
## Verification Commands

To verify integrity manually:

```bash
# Validate problems schema and metadata
python validate_problems.py

# Check test coverage and syntax
python tests/validate_tests.py

# Run comprehensive test suite
python tests/run_all_tests.py

# Generate fresh integrity report
python check_integrity.py
```

---
*This report was generated automatically by the integrity checker.*
"""
    
    return report

def main():
    """Main integrity check function."""
    print("Single Turn Scenarios Integrity Check")
    print("=" * 50)
    
    report = generate_integrity_report()
    print(report)
    
    # Save report
    report_file = Path(__file__).parent / "integrity_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nIntegrity report saved to: {report_file}")
    
    # Determine exit code based on overall status
    problems_check = check_problems_integrity()
    tests_check = check_tests_integrity()
    docs_check = check_documentation_integrity()
    config_check = check_configuration_integrity()
    
    has_errors = any(check["status"] == "ERROR" for check in [problems_check, tests_check, docs_check, config_check])
    
    return 0 if not has_errors else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)