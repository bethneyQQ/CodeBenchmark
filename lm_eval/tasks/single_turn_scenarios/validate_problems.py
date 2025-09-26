#!/usr/bin/env python3
"""
Problem validation and metadata management for single_turn_scenarios.
Validates problem schema, metadata completeness, and licensing compliance.
Includes compliance tracking integration.

Requirements: 12.2, 12.3
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import compliance tracker for integration
try:
    from .compliance_tracker import ComplianceTracker
except ImportError:
    ComplianceTracker = None

class ProblemValidator:
    """Validates problems.jsonl schema and metadata."""
    
    REQUIRED_FIELDS = {
        'id', 'title', 'language', 'scenario', 'difficulty', 
        'context_mode', 'prompt', 'reference', 'tests', 'metadata'
    }
    
    VALID_LANGUAGES = {
        'python', 'javascript', 'java', 'cpp', 'go', 'rust', 'sql'
    }
    
    VALID_SCENARIOS = {
        'code_completion', 'bug_fix', 'code_translation', 'documentation', 
        'function_generation', 'system_design', 'algorithm_implementation',
        'api_design', 'database_design', 'performance_optimization',
        'full_stack', 'testing_strategy', 'security'
    }
    
    VALID_DIFFICULTIES = {'simple', 'intermediate', 'complex'}
    
    VALID_CONTEXT_MODES = {
        'no_context', 'minimal_context', 'full_context', 'domain_context'
    }
    
    REQUIRED_METADATA_FIELDS = {
        'time_limit_s', 'memory_limit_mb', 'seed', 'author', 'license'
    }
    
    VALID_LICENSES = {
        'MIT', 'Apache-2.0', 'GPL-3.0', 'BSD-3-Clause', 'CC0-1.0', 'Unlicense'
    }
    
    def __init__(self, problems_file: Optional[Path] = None, enable_compliance_tracking: bool = True):
        if problems_file is None:
            problems_file = Path(__file__).parent / "problems.jsonl"
        self.problems_file = problems_file
        self.validation_errors = []
        self.validation_warnings = []
        
        # Initialize compliance tracker if available
        self.compliance_tracker = None
        if enable_compliance_tracking and ComplianceTracker:
            try:
                self.compliance_tracker = ComplianceTracker()
            except Exception as e:
                self.validation_warnings.append(f"Failed to initialize compliance tracker: {e}")
    
    def load_problems(self) -> List[Dict]:
        """Load and parse problems from JSONL file."""
        problems = []
        
        if not self.problems_file.exists():
            self.validation_errors.append(f"Problems file not found: {self.problems_file}")
            return []
        
        try:
            with open(self.problems_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        problem = json.loads(line)
                        problem['_line_number'] = line_num
                        problems.append(problem)
                    except json.JSONDecodeError as e:
                        self.validation_errors.append(
                            f"Line {line_num}: Invalid JSON - {str(e)}"
                        )
        except Exception as e:
            self.validation_errors.append(f"Error reading file: {str(e)}")
        
        return problems
    
    def validate_problem_schema(self, problem: Dict) -> List[str]:
        """Validate individual problem schema."""
        errors = []
        line_num = problem.get('_line_number', 'unknown')
        
        # Check required fields
        missing_fields = self.REQUIRED_FIELDS - set(problem.keys())
        if missing_fields:
            errors.append(f"Line {line_num}: Missing required fields: {missing_fields}")
        
        # Validate field values
        if 'id' in problem:
            if not re.match(r'^st_\d{4}$', problem['id']):
                errors.append(f"Line {line_num}: Invalid ID format: {problem['id']}")
        
        if 'language' in problem:
            if problem['language'] not in self.VALID_LANGUAGES:
                errors.append(f"Line {line_num}: Invalid language: {problem['language']}")
        
        if 'scenario' in problem:
            if problem['scenario'] not in self.VALID_SCENARIOS:
                errors.append(f"Line {line_num}: Invalid scenario: {problem['scenario']}")
        
        if 'difficulty' in problem:
            if problem['difficulty'] not in self.VALID_DIFFICULTIES:
                errors.append(f"Line {line_num}: Invalid difficulty: {problem['difficulty']}")
        
        if 'context_mode' in problem:
            if problem['context_mode'] not in self.VALID_CONTEXT_MODES:
                errors.append(f"Line {line_num}: Invalid context_mode: {problem['context_mode']}")
        
        # Validate prompt
        if 'prompt' in problem:
            if not isinstance(problem['prompt'], str) or len(problem['prompt'].strip()) == 0:
                errors.append(f"Line {line_num}: Prompt must be a non-empty string")
        
        # Validate reference
        if 'reference' in problem:
            if not isinstance(problem['reference'], list) or len(problem['reference']) == 0:
                errors.append(f"Line {line_num}: Reference must be a non-empty list")
        
        # Validate tests
        if 'tests' in problem:
            if not isinstance(problem['tests'], list) or len(problem['tests']) == 0:
                errors.append(f"Line {line_num}: Tests must be a non-empty list")
            else:
                for i, test in enumerate(problem['tests']):
                    if not isinstance(test, dict):
                        errors.append(f"Line {line_num}: Test {i} must be a dictionary")
                        continue
                    
                    if 'type' not in test:
                        errors.append(f"Line {line_num}: Test {i} missing 'type' field")
                    
                    if 'cmd' not in test:
                        errors.append(f"Line {line_num}: Test {i} missing 'cmd' field")
        
        return errors
    
    def validate_metadata(self, problem: Dict) -> List[str]:
        """Validate problem metadata."""
        errors = []
        line_num = problem.get('_line_number', 'unknown')
        
        if 'metadata' not in problem:
            errors.append(f"Line {line_num}: Missing metadata section")
            return errors
        
        metadata = problem['metadata']
        if not isinstance(metadata, dict):
            errors.append(f"Line {line_num}: Metadata must be a dictionary")
            return errors
        
        # Check required metadata fields
        missing_fields = self.REQUIRED_METADATA_FIELDS - set(metadata.keys())
        if missing_fields:
            errors.append(f"Line {line_num}: Missing metadata fields: {missing_fields}")
        
        # Validate specific metadata fields
        if 'time_limit_s' in metadata:
            if not isinstance(metadata['time_limit_s'], (int, float)) or metadata['time_limit_s'] <= 0:
                errors.append(f"Line {line_num}: time_limit_s must be a positive number")
        
        if 'memory_limit_mb' in metadata:
            if not isinstance(metadata['memory_limit_mb'], (int, float)) or metadata['memory_limit_mb'] <= 0:
                errors.append(f"Line {line_num}: memory_limit_mb must be a positive number")
        
        if 'seed' in metadata:
            if not isinstance(metadata['seed'], int):
                errors.append(f"Line {line_num}: seed must be an integer")
        
        if 'author' in metadata:
            if not isinstance(metadata['author'], str) or len(metadata['author'].strip()) == 0:
                errors.append(f"Line {line_num}: author must be a non-empty string")
        
        if 'license' in metadata:
            if metadata['license'] not in self.VALID_LICENSES:
                errors.append(f"Line {line_num}: Invalid license: {metadata['license']}")
        
        return errors
    
    def validate_licensing_compliance(self, problems: List[Dict]) -> List[str]:
        """Validate licensing compliance across all problems and register with compliance tracker."""
        errors = []
        
        # Check for consistent licensing
        licenses = set()
        for problem in problems:
            if 'metadata' in problem and 'license' in problem['metadata']:
                licenses.add(problem['metadata']['license'])
        
        if len(licenses) > 1:
            errors.append(f"Multiple licenses found: {licenses}. Consider using consistent licensing.")
        
        # Check for proper attribution
        authors = set()
        for problem in problems:
            if 'metadata' in problem and 'author' in problem['metadata']:
                authors.add(problem['metadata']['author'])
        
        if len(authors) == 0:
            errors.append("No author information found in any problems")
        
        # Register problems with compliance tracker
        if self.compliance_tracker:
            try:
                for problem in problems:
                    if 'id' in problem and 'metadata' in problem:
                        metadata = problem['metadata']
                        license_id = metadata.get('license', 'Unknown')
                        author = metadata.get('author', 'Unknown')
                        
                        # Create content string for checksum
                        content = json.dumps(problem, sort_keys=True)
                        
                        # Register with compliance tracker
                        self.compliance_tracker.register_component(
                            component_id=problem['id'],
                            component_type='problem',
                            license_id=license_id,
                            author=author,
                            content=content
                        )
                        
                        # Register reference implementations if present
                        if 'reference' in problem:
                            for i, ref in enumerate(problem['reference']):
                                ref_id = f"{problem['id']}_ref_{i}"
                                self.compliance_tracker.register_component(
                                    component_id=ref_id,
                                    component_type='reference',
                                    license_id=license_id,
                                    author=author,
                                    content=ref
                                )
                        
                        # Register test files if present
                        if 'tests' in problem:
                            for i, test in enumerate(problem['tests']):
                                test_id = f"{problem['id']}_test_{i}"
                                test_content = json.dumps(test, sort_keys=True)
                                self.compliance_tracker.register_component(
                                    component_id=test_id,
                                    component_type='test',
                                    license_id=license_id,
                                    author=author,
                                    content=test_content
                                )
            except Exception as e:
                errors.append(f"Failed to register components with compliance tracker: {e}")
        
        return errors
    
    def validate_problem_integrity(self, problems: List[Dict]) -> List[str]:
        """Validate problem integrity and consistency."""
        errors = []
        
        # Check for duplicate IDs
        ids = [p.get('id') for p in problems if 'id' in p]
        duplicate_ids = set([x for x in ids if ids.count(x) > 1])
        if duplicate_ids:
            errors.append(f"Duplicate problem IDs found: {duplicate_ids}")
        
        # Check for missing sequential IDs
        expected_ids = set(f"st_{i:04d}" for i in range(1, len(problems) + 1))
        actual_ids = set(ids)
        missing_ids = expected_ids - actual_ids
        if missing_ids:
            errors.append(f"Missing sequential IDs: {sorted(missing_ids)}")
        
        # Check scenario coverage
        scenarios = [p.get('scenario') for p in problems if 'scenario' in p]
        missing_scenarios = self.VALID_SCENARIOS - set(scenarios)
        if missing_scenarios:
            errors.append(f"Missing scenario coverage: {missing_scenarios}")
        
        # Check difficulty distribution
        difficulties = [p.get('difficulty') for p in problems if 'difficulty' in p]
        difficulty_counts = {d: difficulties.count(d) for d in self.VALID_DIFFICULTIES}
        
        for difficulty, count in difficulty_counts.items():
            if count == 0:
                errors.append(f"No problems with difficulty '{difficulty}'")
        
        # Check language coverage
        languages = [p.get('language') for p in problems if 'language' in p]
        language_counts = {l: languages.count(l) for l in self.VALID_LANGUAGES}
        
        primary_languages = {'python', 'javascript', 'java'}
        for lang in primary_languages:
            if language_counts.get(lang, 0) == 0:
                errors.append(f"No problems for primary language '{lang}'")
        
        return errors
    
    def generate_metadata_report(self, problems: List[Dict]) -> str:
        """Generate a comprehensive metadata report."""
        if not problems:
            return "No problems found to analyze."
        
        # Basic statistics
        total_problems = len(problems)
        
        # Language distribution
        languages = [p.get('language') for p in problems if 'language' in p]
        language_counts = {l: languages.count(l) for l in set(languages)}
        
        # Scenario distribution
        scenarios = [p.get('scenario') for p in problems if 'scenario' in p]
        scenario_counts = {s: scenarios.count(s) for s in set(scenarios)}
        
        # Difficulty distribution
        difficulties = [p.get('difficulty') for p in problems if 'difficulty' in p]
        difficulty_counts = {d: difficulties.count(d) for d in set(difficulties)}
        
        # Context mode distribution
        context_modes = [p.get('context_mode') for p in problems if 'context_mode' in p]
        context_counts = {c: context_modes.count(c) for c in set(context_modes)}
        
        # License information
        licenses = [p.get('metadata', {}).get('license') for p in problems if 'metadata' in p and 'license' in p['metadata']]
        license_counts = {l: licenses.count(l) for l in set(licenses) if l}
        
        # Authors
        authors = [p.get('metadata', {}).get('author') for p in problems if 'metadata' in p and 'author' in p['metadata']]
        author_counts = {a: authors.count(a) for a in set(authors) if a}
        
        # Time and memory limits
        time_limits = [p.get('metadata', {}).get('time_limit_s') for p in problems if 'metadata' in p and 'time_limit_s' in p['metadata']]
        memory_limits = [p.get('metadata', {}).get('memory_limit_mb') for p in problems if 'metadata' in p and 'memory_limit_mb' in p['metadata']]
        
        report = f"""# Single Turn Scenarios Metadata Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Problems**: {total_problems}
- **File**: {self.problems_file.name}

## Language Distribution
"""
        
        for lang, count in sorted(language_counts.items()):
            percentage = (count / total_problems) * 100
            report += f"- **{lang}**: {count} ({percentage:.1f}%)\n"
        
        report += "\n## Scenario Distribution\n"
        for scenario, count in sorted(scenario_counts.items()):
            percentage = (count / total_problems) * 100
            report += f"- **{scenario.replace('_', ' ').title()}**: {count} ({percentage:.1f}%)\n"
        
        report += "\n## Difficulty Distribution\n"
        for difficulty, count in sorted(difficulty_counts.items()):
            percentage = (count / total_problems) * 100
            report += f"- **{difficulty.title()}**: {count} ({percentage:.1f}%)\n"
        
        report += "\n## Context Mode Distribution\n"
        for context, count in sorted(context_counts.items()):
            percentage = (count / total_problems) * 100
            report += f"- **{context.replace('_', ' ').title()}**: {count} ({percentage:.1f}%)\n"
        
        if license_counts:
            report += "\n## Licensing Information\n"
            for license_type, count in sorted(license_counts.items()):
                percentage = (count / total_problems) * 100
                report += f"- **{license_type}**: {count} ({percentage:.1f}%)\n"
        
        if author_counts:
            report += "\n## Author Information\n"
            for author, count in sorted(author_counts.items()):
                percentage = (count / total_problems) * 100
                report += f"- **{author}**: {count} ({percentage:.1f}%)\n"
        
        if time_limits:
            avg_time = sum(time_limits) / len(time_limits)
            min_time = min(time_limits)
            max_time = max(time_limits)
            report += f"\n## Resource Limits\n"
            report += f"- **Time Limits**: {min_time}s - {max_time}s (avg: {avg_time:.1f}s)\n"
        
        if memory_limits:
            avg_memory = sum(memory_limits) / len(memory_limits)
            min_memory = min(memory_limits)
            max_memory = max(memory_limits)
            report += f"- **Memory Limits**: {min_memory}MB - {max_memory}MB (avg: {avg_memory:.1f}MB)\n"
        
        return report
    
    def validate_all(self) -> Tuple[bool, str]:
        """Run comprehensive validation and return results."""
        self.validation_errors = []
        self.validation_warnings = []
        
        print("Loading problems...")
        problems = self.load_problems()
        
        if not problems:
            return False, "No problems loaded for validation"
        
        print(f"Validating {len(problems)} problems...")
        
        # Validate each problem
        for problem in problems:
            schema_errors = self.validate_problem_schema(problem)
            self.validation_errors.extend(schema_errors)
            
            metadata_errors = self.validate_metadata(problem)
            self.validation_errors.extend(metadata_errors)
        
        # Validate licensing compliance
        licensing_errors = self.validate_licensing_compliance(problems)
        self.validation_warnings.extend(licensing_errors)
        
        # Validate problem integrity
        integrity_errors = self.validate_problem_integrity(problems)
        self.validation_warnings.extend(integrity_errors)
        
        # Generate compliance audit if tracker is available
        compliance_report = ""
        if self.compliance_tracker:
            try:
                audit_trail = self.compliance_tracker.perform_compliance_audit("validation_system")
                compliance_report = f"""

## Compliance Audit Results

- **Audit ID**: {audit_trail.audit_id}
- **Total Components**: {audit_trail.total_components}
- **Compliant**: {audit_trail.compliant_components}
- **Non-Compliant**: {audit_trail.non_compliant_components}
- **Needs Review**: {audit_trail.needs_review_components}

### Issues Found
"""
                if audit_trail.issues_found:
                    for issue in audit_trail.issues_found:
                        compliance_report += f"- {issue}\n"
                else:
                    compliance_report += "- No compliance issues found\n"
                
                compliance_report += "\n### Recommendations\n"
                if audit_trail.recommendations:
                    for rec in audit_trail.recommendations:
                        compliance_report += f"- {rec}\n"
                else:
                    compliance_report += "- No recommendations\n"
                    
            except Exception as e:
                compliance_report = f"\n## Compliance Audit\n\n- ⚠️ Failed to perform compliance audit: {e}\n"
        
        # Generate report
        report = self.generate_metadata_report(problems)
        
        # Compile results
        result_text = f"""# Validation Results

## Errors ({len(self.validation_errors)})
"""
        
        if self.validation_errors:
            for error in self.validation_errors:
                result_text += f"- ❌ {error}\n"
        else:
            result_text += "- ✅ No errors found\n"
        
        result_text += f"\n## Warnings ({len(self.validation_warnings)})\n"
        
        if self.validation_warnings:
            for warning in self.validation_warnings:
                result_text += f"- ⚠️ {warning}\n"
        else:
            result_text += "- ✅ No warnings\n"
        
        result_text += compliance_report
        result_text += f"\n{report}"
        
        success = len(self.validation_errors) == 0
        return success, result_text

def main():
    """Main validation function."""
    print("Single Turn Scenarios Problem Validation")
    print("=" * 60)
    
    validator = ProblemValidator()
    success, report = validator.validate_all()
    
    print(report)
    
    # Save report to file
    report_file = Path(__file__).parent / "validation_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nValidation report saved to: {report_file}")
    
    if success:
        print("\n🎉 Validation passed! All problems have valid metadata and licensing.")
    else:
        print(f"\n❌ Validation failed with {len(validator.validation_errors)} errors.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)