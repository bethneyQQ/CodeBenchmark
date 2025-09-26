#!/usr/bin/env python3
"""
Configuration Validation Script

This script validates all configuration files for the single_turn_scenarios task,
including YAML files, JSON files, and dataset schema validation.
"""

import os
import sys
import json
import yaml
import jsonschema
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of a validation check."""
    file_path: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]

class ConfigValidator:
    """Configuration file validator."""
    
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.results: List[ValidationResult] = []
    
    def validate_json_file(self, file_path: Path, schema: Dict[str, Any] = None) -> ValidationResult:
        """Validate a JSON file against an optional schema."""
        errors = []
        warnings = []
        details = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            details['keys'] = list(data.keys()) if isinstance(data, dict) else None
            details['type'] = type(data).__name__
            
            # Schema validation if provided
            if schema:
                try:
                    jsonschema.validate(data, schema)
                    details['schema_valid'] = True
                except jsonschema.ValidationError as e:
                    errors.append(f"Schema validation error: {e.message}")
                    details['schema_valid'] = False
                except jsonschema.SchemaError as e:
                    warnings.append(f"Schema error: {e.message}")
            
            return ValidationResult(
                file_path=str(file_path),
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
        except json.JSONDecodeError as e:
            return ValidationResult(
                file_path=str(file_path),
                is_valid=False,
                errors=[f"JSON decode error: {e.msg} at line {e.lineno}"],
                warnings=warnings,
                details=details
            )
        except Exception as e:
            return ValidationResult(
                file_path=str(file_path),
                is_valid=False,
                errors=[f"Unexpected error: {str(e)}"],
                warnings=warnings,
                details=details
            )
    
    def validate_yaml_file(self, file_path: Path, schema: Dict[str, Any] = None) -> ValidationResult:
        """Validate a YAML file against an optional schema."""
        errors = []
        warnings = []
        details = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            details['keys'] = list(data.keys()) if isinstance(data, dict) else None
            details['type'] = type(data).__name__
            
            # Schema validation if provided
            if schema:
                try:
                    jsonschema.validate(data, schema)
                    details['schema_valid'] = True
                except jsonschema.ValidationError as e:
                    errors.append(f"Schema validation error: {e.message}")
                    details['schema_valid'] = False
                except jsonschema.SchemaError as e:
                    warnings.append(f"Schema error: {e.message}")
            
            return ValidationResult(
                file_path=str(file_path),
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
        except yaml.YAMLError as e:
            return ValidationResult(
                file_path=str(file_path),
                is_valid=False,
                errors=[f"YAML parse error: {str(e)}"],
                warnings=warnings,
                details=details
            )
        except Exception as e:
            return ValidationResult(
                file_path=str(file_path),
                is_valid=False,
                errors=[f"Unexpected error: {str(e)}"],
                warnings=warnings,
                details=details
            )
    
    def validate_context_configs(self) -> ValidationResult:
        """Validate context_configs.json file."""
        file_path = self.current_dir / "context_configs.json"
        
        # Define expected schema for context configs
        schema = {
            "type": "object",
            "patternProperties": {
                "^[a-z_]+$": {
                    "type": "object",
                    "properties": {
                        "template": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["template", "description"]
                }
            },
            "additionalProperties": False
        }
        
        if not file_path.exists():
            return ValidationResult(
                file_path=str(file_path),
                is_valid=False,
                errors=["File not found"],
                warnings=[],
                details={}
            )
        
        result = self.validate_json_file(file_path, schema)
        
        # Additional validation for context configs
        if result.is_valid:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                expected_contexts = ["no_context", "minimal_context", "full_context", "domain_context"]
                missing_contexts = [ctx for ctx in expected_contexts if ctx not in data]
                
                if missing_contexts:
                    result.warnings.append(f"Missing expected contexts: {missing_contexts}")
                
                # Check template placeholders
                for context_name, context_data in data.items():
                    template = context_data.get("template", "")
                    if "{{prompt}}" not in template:
                        result.warnings.append(f"Context '{context_name}' template missing {{{{prompt}}}} placeholder")
                
                result.details['contexts'] = list(data.keys())
                result.details['expected_contexts'] = expected_contexts
                result.details['missing_contexts'] = missing_contexts
                
            except Exception as e:
                result.warnings.append(f"Additional validation error: {str(e)}")
        
        return result
    
    def validate_model_configs(self) -> List[ValidationResult]:
        """Validate all model configuration files."""
        model_configs_dir = self.current_dir / "model_configs"
        results = []
        
        if not model_configs_dir.exists():
            results.append(ValidationResult(
                file_path=str(model_configs_dir),
                is_valid=False,
                errors=["Model configs directory not found"],
                warnings=[],
                details={}
            ))
            return results
        
        # Define schema for model configs
        schema = {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "endpoint_config": {
                    "type": "object",
                    "properties": {
                        "base_url": {"type": "string"},
                        "timeout": {"type": "number"},
                        "rate_limit": {"type": "number"}
                    }
                },
                "generation_params": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number"},
                        "max_tokens": {"type": "integer"},
                        "top_p": {"type": "number"}
                    }
                },
                "batch_config": {"type": "object"},
                "tokenizer_config": {"type": "object"}
            },
            "required": ["model_name"]
        }
        
        yaml_files = list(model_configs_dir.glob("*.yaml"))
        
        if not yaml_files:
            results.append(ValidationResult(
                file_path=str(model_configs_dir),
                is_valid=False,
                errors=["No YAML files found in model_configs directory"],
                warnings=[],
                details={}
            ))
            return results
        
        for yaml_file in yaml_files:
            result = self.validate_yaml_file(yaml_file, schema)
            results.append(result)
        
        return results
    
    def validate_task_configs(self) -> List[ValidationResult]:
        """Validate task configuration YAML files."""
        results = []
        
        # Find all YAML files in the current directory (task configs)
        yaml_files = [f for f in self.current_dir.glob("*.yaml") 
                     if f.name != "single_turn_scenarios_suite.yaml"]
        
        # Basic schema for task configs
        schema = {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "dataset_path": {"type": "string"},
                "description": {"type": "string"},
                "output_type": {"type": "string"},
                "doc_to_text": {"type": "string"},
                "doc_to_target": {"type": "string"},
                "metric": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array"},
                        {"type": "object"}
                    ]
                }
            }
        }
        
        for yaml_file in yaml_files:
            result = self.validate_yaml_file(yaml_file, schema)
            results.append(result)
        
        return results
    
    def validate_suite_config(self) -> ValidationResult:
        """Validate the suite configuration file."""
        file_path = self.current_dir / "single_turn_scenarios_suite.yaml"
        
        if not file_path.exists():
            return ValidationResult(
                file_path=str(file_path),
                is_valid=False,
                errors=["Suite configuration file not found"],
                warnings=[],
                details={}
            )
        
        # Schema for suite config
        schema = {
            "type": "object",
            "properties": {
                "group": {"type": "string"},
                "task": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ]
                }
            }
        }
        
        return self.validate_yaml_file(file_path, schema)
    
    def validate_problems_dataset(self) -> ValidationResult:
        """Validate the problems.jsonl dataset file."""
        file_path = self.current_dir / "problems.jsonl"
        errors = []
        warnings = []
        details = {}
        
        if not file_path.exists():
            return ValidationResult(
                file_path=str(file_path),
                is_valid=False,
                errors=["Problems dataset file not found"],
                warnings=[],
                details={}
            )
        
        # Define schema for problem entries
        problem_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "language": {"type": "string", "enum": ["python", "javascript", "java", "cpp", "go", "rust"]},
                "scenario": {"type": "string"},
                "difficulty": {"type": "string", "enum": ["simple", "intermediate", "complex"]},
                "context_mode": {"type": "string"},
                "prompt": {"type": "string"},
                "reference": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ]
                },
                "tests": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "file": {"type": "string"},
                            "cmd": {"type": "string"}
                        },
                        "required": ["type"]
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "time_limit_s": {"type": "number"},
                        "memory_limit_mb": {"type": "number"},
                        "seed": {"type": "integer"},
                        "author": {"type": "string"},
                        "license": {"type": "string"}
                    }
                }
            },
            "required": ["id", "title", "language", "scenario", "difficulty", "prompt"]
        }
        
        try:
            problems = []
            line_count = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    if line.strip():
                        try:
                            problem = json.loads(line)
                            problems.append(problem)
                            
                            # Validate against schema
                            try:
                                jsonschema.validate(problem, problem_schema)
                            except jsonschema.ValidationError as e:
                                errors.append(f"Problem {problem.get('id', line_num)} schema error: {e.message}")
                                
                        except json.JSONDecodeError as e:
                            errors.append(f"JSON decode error at line {line_num}: {e.msg}")
            
            # Collect statistics
            if problems:
                scenarios = [p.get("scenario") for p in problems]
                languages = [p.get("language") for p in problems]
                difficulties = [p.get("difficulty") for p in problems]
                
                details.update({
                    "total_problems": len(problems),
                    "lines_processed": line_count,
                    "scenarios": {scenario: scenarios.count(scenario) for scenario in set(scenarios) if scenario},
                    "languages": {lang: languages.count(lang) for lang in set(languages) if lang},
                    "difficulties": {diff: difficulties.count(diff) for diff in set(difficulties) if diff},
                    "ids": [p.get("id") for p in problems[:10]]  # First 10 IDs
                })
                
                # Check for duplicate IDs
                ids = [p.get("id") for p in problems if p.get("id")]
                duplicate_ids = [id for id in set(ids) if ids.count(id) > 1]
                if duplicate_ids:
                    errors.append(f"Duplicate problem IDs found: {duplicate_ids}")
                
                # Check for missing test files
                test_files_missing = []
                for problem in problems:
                    tests = problem.get("tests", [])
                    for test in tests:
                        test_file = test.get("file")
                        if test_file:
                            test_path = self.current_dir / test_file
                            if not test_path.exists():
                                test_files_missing.append(f"{problem.get('id', 'unknown')}: {test_file}")
                
                if test_files_missing:
                    warnings.append(f"Missing test files: {test_files_missing[:5]}")  # First 5
                    details["missing_test_files"] = len(test_files_missing)
            
            return ValidationResult(
                file_path=str(file_path),
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                file_path=str(file_path),
                is_valid=False,
                errors=[f"Unexpected error: {str(e)}"],
                warnings=warnings,
                details=details
            )
    
    def run_all_validations(self) -> List[ValidationResult]:
        """Run all configuration validations."""
        print("🔍 Validating Single Turn Scenarios Configuration Files")
        print("=" * 60)
        
        all_results = []
        
        # Validate context configs
        print("📋 Validating context configurations...")
        result = self.validate_context_configs()
        all_results.append(result)
        self._print_result(result)
        
        # Validate model configs
        print("\n🤖 Validating model configurations...")
        model_results = self.validate_model_configs()
        all_results.extend(model_results)
        for result in model_results:
            self._print_result(result)
        
        # Validate task configs
        print("\n📝 Validating task configurations...")
        task_results = self.validate_task_configs()
        all_results.extend(task_results)
        for result in task_results:
            self._print_result(result)
        
        # Validate suite config
        print("\n📦 Validating suite configuration...")
        result = self.validate_suite_config()
        all_results.append(result)
        self._print_result(result)
        
        # Validate problems dataset
        print("\n📊 Validating problems dataset...")
        result = self.validate_problems_dataset()
        all_results.append(result)
        self._print_result(result)
        
        # Summary
        valid_count = sum(1 for r in all_results if r.is_valid)
        total_count = len(all_results)
        
        print("\n" + "=" * 60)
        print("📋 Validation Summary")
        print("=" * 60)
        print(f"✅ Valid: {valid_count}/{total_count}")
        print(f"❌ Invalid: {total_count - valid_count}/{total_count}")
        
        if valid_count == total_count:
            print("\n🎉 All configuration files are valid!")
        else:
            print(f"\n⚠️  {total_count - valid_count} configuration files have issues.")
            print("Please review the errors above and fix the configuration files.")
        
        return all_results
    
    def _print_result(self, result: ValidationResult):
        """Print a single validation result."""
        status = "✅" if result.is_valid else "❌"
        file_name = Path(result.file_path).name
        print(f"   {status} {file_name}")
        
        if result.errors:
            for error in result.errors:
                print(f"      ❌ {error}")
        
        if result.warnings:
            for warning in result.warnings:
                print(f"      ⚠️  {warning}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Single Turn Scenarios Configuration Files")
    parser.add_argument("--output", "-o", help="Output validation report (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    results = validator.run_all_validations()
    
    # Save report if requested
    if args.output:
        report_data = {
            "timestamp": str(datetime.now()),
            "total_files": len(results),
            "valid_files": sum(1 for r in results if r.is_valid),
            "results": [
                {
                    "file_path": r.file_path,
                    "is_valid": r.is_valid,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "details": r.details
                }
                for r in results
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\n📄 Validation report saved to: {args.output}")
    
    # Exit with appropriate code
    invalid_count = sum(1 for r in results if not r.is_valid)
    if invalid_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    from datetime import datetime
    main()