#!/usr/bin/env python3
"""
Test validation script for single_turn_scenarios.
Validates that all problems have corresponding test files and that tests are executable.
"""

import json
import os
from pathlib import Path
import subprocess

def load_problems():
    """Load problems from problems.jsonl file."""
    problems_file = Path(__file__).parent.parent / "problems.jsonl"
    problems = []
    
    try:
        with open(problems_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        problems.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: Problems file not found at {problems_file}")
        return []
    
    return problems

def validate_test_coverage():
    """Validate that all problems have corresponding test files."""
    problems = load_problems()
    tests_dir = Path(__file__).parent
    
    print("Validating test coverage...")
    print("=" * 50)
    
    missing_tests = []
    existing_tests = []
    
    for problem in problems:
        test_id = problem['id']
        language = problem['language']
        
        # Look for test files with various extensions
        test_patterns = [
            f"test_{test_id}.py",
            f"test_{test_id}.js", 
            f"test_{test_id}.java",
            f"test_{test_id}.cpp",
            f"test_{test_id}.go",
            f"test_{test_id}.rs",
            f"test_{test_id}.sql"
        ]
        
        found_test = False
        for pattern in test_patterns:
            test_file = tests_dir / pattern
            if test_file.exists():
                existing_tests.append({
                    'problem_id': test_id,
                    'language': language,
                    'scenario': problem['scenario'],
                    'test_file': pattern,
                    'file_size': test_file.stat().st_size
                })
                found_test = True
                break
        
        if not found_test:
            missing_tests.append({
                'problem_id': test_id,
                'language': language,
                'scenario': problem['scenario']
            })
    
    # Print results
    print(f"Total problems: {len(problems)}")
    print(f"Tests found: {len(existing_tests)}")
    print(f"Missing tests: {len(missing_tests)}")
    print(f"Coverage: {len(existing_tests)/len(problems)*100:.1f}%")
    
    if missing_tests:
        print("\nMissing test files:")
        for missing in missing_tests:
            print(f"  - {missing['problem_id']} ({missing['language']}, {missing['scenario']})")
    
    if existing_tests:
        print("\nExisting test files:")
        for test in existing_tests:
            size_kb = test['file_size'] / 1024
            print(f"  ✓ {test['test_file']} ({size_kb:.1f} KB) - {test['scenario']}")
    
    return len(missing_tests) == 0

def validate_test_syntax():
    """Validate that test files have correct syntax."""
    tests_dir = Path(__file__).parent
    
    print("\nValidating test file syntax...")
    print("=" * 50)
    
    python_tests = list(tests_dir.glob("test_*.py"))
    js_tests = list(tests_dir.glob("test_*.js"))
    java_tests = list(tests_dir.glob("test_*.java"))
    
    syntax_errors = []
    
    # Validate Python tests
    for test_file in python_tests:
        try:
            result = subprocess.run(
                ['python', '-m', 'py_compile', str(test_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                syntax_errors.append({
                    'file': test_file.name,
                    'error': result.stderr
                })
            else:
                print(f"  ✓ {test_file.name} - Python syntax OK")
        except Exception as e:
            syntax_errors.append({
                'file': test_file.name,
                'error': str(e)
            })
    
    # Validate JavaScript tests
    for test_file in js_tests:
        try:
            result = subprocess.run(
                ['node', '--check', str(test_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                syntax_errors.append({
                    'file': test_file.name,
                    'error': result.stderr
                })
            else:
                print(f"  ✓ {test_file.name} - JavaScript syntax OK")
        except Exception as e:
            syntax_errors.append({
                'file': test_file.name,
                'error': str(e)
            })
    
    # Validate Java tests (compilation check)
    for test_file in java_tests:
        try:
            result = subprocess.run(
                ['javac', '-cp', '.', str(test_file)],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=tests_dir
            )
            if result.returncode != 0:
                syntax_errors.append({
                    'file': test_file.name,
                    'error': result.stderr
                })
            else:
                print(f"  ✓ {test_file.name} - Java compilation OK")
                # Clean up compiled class files
                class_file = tests_dir / (test_file.stem + ".class")
                if class_file.exists():
                    class_file.unlink()
        except Exception as e:
            syntax_errors.append({
                'file': test_file.name,
                'error': str(e)
            })
    
    if syntax_errors:
        print("\nSyntax errors found:")
        for error in syntax_errors:
            print(f"  ✗ {error['file']}: {error['error']}")
        return False
    else:
        print("\nAll test files have valid syntax!")
        return True

def validate_test_structure():
    """Validate that test files have proper structure."""
    tests_dir = Path(__file__).parent
    
    print("\nValidating test file structure...")
    print("=" * 50)
    
    python_tests = list(tests_dir.glob("test_*.py"))
    structure_issues = []
    
    for test_file in python_tests:
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            issues = []
            
            # Check for basic test structure
            if 'def test_' not in content:
                issues.append("No test functions found (should have 'def test_')")
            
            if 'import pytest' not in content and 'import unittest' not in content:
                issues.append("No test framework import found")
            
            if '__name__ == "__main__"' not in content:
                issues.append("No main execution block found")
            
            # Check for docstring
            if '"""' not in content and "'''" not in content:
                issues.append("No docstring found")
            
            if issues:
                structure_issues.append({
                    'file': test_file.name,
                    'issues': issues
                })
            else:
                print(f"  ✓ {test_file.name} - Structure OK")
                
        except Exception as e:
            structure_issues.append({
                'file': test_file.name,
                'issues': [f"Error reading file: {str(e)}"]
            })
    
    if structure_issues:
        print("\nStructure issues found:")
        for issue in structure_issues:
            print(f"  ✗ {issue['file']}:")
            for problem in issue['issues']:
                print(f"    - {problem}")
        return False
    else:
        print("\nAll test files have proper structure!")
        return True

def main():
    """Main validation function."""
    print("Single Turn Scenarios Test Validation")
    print("=" * 60)
    
    # Run all validations
    coverage_ok = validate_test_coverage()
    syntax_ok = validate_test_syntax()
    structure_ok = validate_test_structure()
    
    print("\n" + "=" * 60)
    print("Validation Summary:")
    print(f"  Test Coverage: {'✓ PASS' if coverage_ok else '✗ FAIL'}")
    print(f"  Syntax Check: {'✓ PASS' if syntax_ok else '✗ FAIL'}")
    print(f"  Structure Check: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    
    if coverage_ok and syntax_ok and structure_ok:
        print("\n🎉 All validations passed! Test suite is ready.")
        return True
    else:
        print("\n❌ Some validations failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)