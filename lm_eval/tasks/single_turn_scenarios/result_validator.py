#!/usr/bin/env python3
"""
Result Validation Module

Provides comprehensive validation for evaluation results to ensure compatibility
with analysis tools and adherence to the structured JSON format specification.

Requirements addressed: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import json
import jsonschema
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultValidator:
    """Validates evaluation results for analysis tool compatibility."""
    
    def __init__(self):
        self.result_schema = self._load_result_schema()
        self.validation_rules = self._load_validation_rules()
    
    def _load_result_schema(self) -> Dict[str, Any]:
        """Load the JSON schema for result validation."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": [
                "id", "model", "config", "prediction", "metrics", 
                "runtime", "seed", "commit", "requirements", "timestamp"
            ],
            "properties": {
                "id": {
                    "type": "string",
                    "pattern": "^st_[0-9]{4}$",
                    "description": "Problem identifier"
                },
                "model": {
                    "type": "string",
                    "description": "Model name and version"
                },
                "config": {
                    "type": "string",
                    "description": "Configuration string (context_mode|parameters)"
                },
                "prediction": {
                    "type": "string",
                    "description": "Model's generated code/response"
                },
                "metrics": {
                    "type": "object",
                    "required": ["exact_match", "codebleu", "pass_at_1", "syntax_valid"],
                    "properties": {
                        # Basic metrics
                        "exact_match": {"type": "number", "minimum": 0, "maximum": 1},
                        "bleu_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "codebleu": {"type": "number", "minimum": 0, "maximum": 1},
                        "rouge_l": {"type": "number", "minimum": 0, "maximum": 1},
                        "edit_distance": {"type": "number", "minimum": 0, "maximum": 1},
                        
                        # Code quality metrics
                        "syntax_valid": {"type": "number", "minimum": 0, "maximum": 1},
                        "cyclomatic_complexity": {"type": "number", "minimum": 0},
                        "security_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "performance_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "code_style_score": {"type": "number", "minimum": 0, "maximum": 1},
                        
                        # Functional metrics
                        "pass_at_1": {"type": "number", "minimum": 0, "maximum": 1},
                        "pass_at_5": {"type": "number", "minimum": 0, "maximum": 1},
                        "pass_at_10": {"type": "number", "minimum": 0, "maximum": 1},
                        "test_coverage": {"type": "number", "minimum": 0, "maximum": 1},
                        "runtime_correctness": {"type": "number", "minimum": 0, "maximum": 1},
                        "memory_efficiency": {"type": "number", "minimum": 0, "maximum": 1},
                        
                        # Consistency metrics (for complex scenarios)
                        "phase_coherence": {"type": "number", "minimum": 0, "maximum": 1},
                        "design_implementation_alignment": {"type": "number", "minimum": 0, "maximum": 1},
                        "information_flow": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "additionalProperties": False
                },
                "runtime": {
                    "type": "object",
                    "required": ["time_s", "exit_code", "peak_memory_mb"],
                    "properties": {
                        "time_s": {"type": "number", "minimum": 0},
                        "exit_code": {"type": "integer"},
                        "peak_memory_mb": {"type": "number", "minimum": 0},
                        "cpu_usage_percent": {"type": "number", "minimum": 0, "maximum": 100},
                        "security_violations": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "additionalProperties": False
                },
                "seed": {"type": "integer"},
                "commit": {"type": "string"},
                "requirements": {"type": "string"},
                "timestamp": {
                    "type": "string",
                    "format": "date-time"
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "scenario": {"type": "string"},
                        "difficulty": {"type": "string"},
                        "language": {"type": "string"},
                        "context_mode": {"type": "string"},
                        "evaluation_version": {"type": "string"}
                    }
                }
            },
            "additionalProperties": False
        }
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for analysis tool compatibility."""
        return {
            "required_metrics": [
                "exact_match", "codebleu", "pass_at_1", "syntax_valid"
            ],
            "optional_metrics": [
                "bleu_score", "rouge_l", "edit_distance", "cyclomatic_complexity",
                "security_score", "performance_score", "code_style_score",
                "pass_at_5", "pass_at_10", "test_coverage", "runtime_correctness",
                "memory_efficiency", "phase_coherence", "design_implementation_alignment",
                "information_flow"
            ],
            "metric_ranges": {
                "probability_metrics": [
                    "exact_match", "bleu_score", "codebleu", "rouge_l", "edit_distance",
                    "syntax_valid", "security_score", "performance_score", "code_style_score",
                    "pass_at_1", "pass_at_5", "pass_at_10", "test_coverage", 
                    "runtime_correctness", "memory_efficiency", "phase_coherence",
                    "design_implementation_alignment", "information_flow"
                ],
                "count_metrics": ["cyclomatic_complexity"]
            },
            "consistency_checks": [
                "pass_at_k_monotonic",
                "syntax_pass_consistency",
                "runtime_exit_consistency"
            ]
        }
    
    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single result against the schema and analysis tool requirements.
        
        Args:
            result: Result dictionary to validate
            
        Returns:
            Validation report with status, errors, warnings, and compatibility info
        """
        validation_report = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "missing_metrics": [],
            "invalid_values": [],
            "analysis_tool_compatible": False,
            "compatibility_issues": []
        }
        
        try:
            # Validate against JSON schema
            jsonschema.validate(result, self.result_schema)
            validation_report["valid"] = True
            
        except jsonschema.ValidationError as e:
            validation_report["errors"].append(f"Schema validation error: {e.message}")
            validation_report["valid"] = False
        
        # Additional validation checks
        self._validate_required_fields(result, validation_report)
        self._validate_metrics_completeness(result, validation_report)
        self._validate_metric_values(result, validation_report)
        self._validate_consistency(result, validation_report)
        self._validate_analysis_tool_compatibility(result, validation_report)
        
        # Set overall compatibility status
        validation_report["analysis_tool_compatible"] = (
            validation_report["valid"] and 
            len(validation_report["compatibility_issues"]) == 0
        )
        
        return validation_report
    
    def _validate_required_fields(self, result: Dict[str, Any], report: Dict[str, Any]):
        """Validate presence and format of required fields."""
        required_fields = ["id", "model", "config", "prediction", "metrics", 
                          "runtime", "seed", "commit", "requirements", "timestamp"]
        
        for field in required_fields:
            if field not in result:
                report["errors"].append(f"Missing required field: {field}")
        
        # Validate ID format
        if "id" in result:
            if not re.match(r"^st_\d{4}$", result["id"]):
                report["errors"].append(f"Invalid ID format: {result['id']} (should be st_XXXX)")
        
        # Validate timestamp format
        if "timestamp" in result:
            try:
                datetime.fromisoformat(result["timestamp"].replace('Z', '+00:00'))
            except ValueError:
                report["errors"].append(f"Invalid timestamp format: {result['timestamp']}")
    
    def _validate_metrics_completeness(self, result: Dict[str, Any], report: Dict[str, Any]):
        """Check for missing required metrics."""
        if "metrics" not in result:
            report["errors"].append("Missing metrics section")
            return
        
        metrics = result["metrics"]
        
        # Check required metrics
        for metric in self.validation_rules["required_metrics"]:
            if metric not in metrics:
                report["missing_metrics"].append(metric)
                report["compatibility_issues"].append(f"Missing required metric: {metric}")
        
        # Check for consistency metrics in complex scenarios
        metadata = result.get("metadata", {})
        if metadata.get("difficulty") == "complex":
            consistency_metrics = ["phase_coherence", "design_implementation_alignment", "information_flow"]
            for metric in consistency_metrics:
                if metric not in metrics:
                    report["warnings"].append(f"Missing consistency metric for complex scenario: {metric}")
    
    def _validate_metric_values(self, result: Dict[str, Any], report: Dict[str, Any]):
        """Validate metric value ranges and types."""
        if "metrics" not in result:
            return
        
        metrics = result["metrics"]
        
        # Check probability metrics (0-1 range)
        for metric in self.validation_rules["metric_ranges"]["probability_metrics"]:
            if metric in metrics:
                value = metrics[metric]
                if not isinstance(value, (int, float)):
                    report["invalid_values"].append(f"{metric}: {value} (should be numeric)")
                elif not (0 <= value <= 1):
                    report["invalid_values"].append(f"{metric}: {value} (should be 0-1)")
        
        # Check count metrics (non-negative)
        for metric in self.validation_rules["metric_ranges"]["count_metrics"]:
            if metric in metrics:
                value = metrics[metric]
                if not isinstance(value, (int, float)):
                    report["invalid_values"].append(f"{metric}: {value} (should be numeric)")
                elif value < 0:
                    report["invalid_values"].append(f"{metric}: {value} (should be non-negative)")
    
    def _validate_consistency(self, result: Dict[str, Any], report: Dict[str, Any]):
        """Validate internal consistency of metrics and runtime data."""
        if "metrics" not in result:
            return
        
        metrics = result["metrics"]
        runtime = result.get("runtime", {})
        
        # Check Pass@K monotonicity
        pass_at_metrics = ["pass_at_1", "pass_at_5", "pass_at_10"]
        pass_at_values = [metrics.get(m) for m in pass_at_metrics if m in metrics]
        if len(pass_at_values) > 1:
            for i in range(1, len(pass_at_values)):
                if pass_at_values[i] < pass_at_values[i-1]:
                    report["warnings"].append("Pass@K metrics are not monotonically increasing")
                    break
        
        # Check syntax-pass consistency
        if "syntax_valid" in metrics and "pass_at_1" in metrics:
            if metrics["syntax_valid"] == 0 and metrics["pass_at_1"] > 0:
                report["warnings"].append("Invalid syntax but positive pass@1 score")
        
        # Check runtime-exit consistency
        if "exit_code" in runtime and "pass_at_1" in metrics:
            if runtime["exit_code"] != 0 and metrics["pass_at_1"] > 0:
                report["warnings"].append("Non-zero exit code but positive pass@1 score")
    
    def _validate_analysis_tool_compatibility(self, result: Dict[str, Any], report: Dict[str, Any]):
        """Validate compatibility with analysis tools."""
        # Check for required fields for analysis tools
        analysis_required_fields = {
            "model": "Model comparison analysis",
            "metadata.scenario": "Scenario analysis",
            "metadata.difficulty": "Difficulty analysis", 
            "metadata.language": "Language analysis",
            "metadata.context_mode": "Context impact analysis"
        }
        
        for field_path, tool_name in analysis_required_fields.items():
            if "." in field_path:
                # Nested field
                parts = field_path.split(".")
                value = result
                for part in parts:
                    value = value.get(part, {}) if isinstance(value, dict) else None
                    if value is None:
                        break
                if value is None:
                    report["compatibility_issues"].append(f"Missing {field_path} required for {tool_name}")
            else:
                # Top-level field
                if field_path not in result:
                    report["compatibility_issues"].append(f"Missing {field_path} required for {tool_name}")
        
        # Check metric completeness for visualization tools
        visualization_metrics = ["exact_match", "codebleu", "pass_at_1", "syntax_valid"]
        missing_viz_metrics = [m for m in visualization_metrics if m not in result.get("metrics", {})]
        if missing_viz_metrics:
            report["compatibility_issues"].append(f"Missing visualization metrics: {missing_viz_metrics}")
    
    def validate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of results.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Batch validation report
        """
        batch_report = {
            "total_results": len(results),
            "valid_results": 0,
            "invalid_results": 0,
            "analysis_compatible_results": 0,
            "validation_details": [],
            "summary_errors": [],
            "summary_warnings": [],
            "batch_issues": []
        }
        
        if not results:
            batch_report["batch_issues"].append("Empty results batch")
            return batch_report
        
        # Validate each result
        for i, result in enumerate(results):
            validation = self.validate_result(result)
            
            validation_detail = {
                "index": i,
                "id": result.get("id", f"unknown_{i}"),
                "valid": validation["valid"],
                "analysis_compatible": validation["analysis_tool_compatible"],
                "error_count": len(validation["errors"]),
                "warning_count": len(validation["warnings"])
            }
            
            if validation["valid"]:
                batch_report["valid_results"] += 1
            else:
                batch_report["invalid_results"] += 1
                batch_report["summary_errors"].extend(validation["errors"])
            
            if validation["analysis_tool_compatible"]:
                batch_report["analysis_compatible_results"] += 1
            
            batch_report["summary_warnings"].extend(validation["warnings"])
            batch_report["validation_details"].append(validation_detail)
        
        # Check for batch-level issues
        self._validate_batch_consistency(results, batch_report)
        
        return batch_report
    
    def _validate_batch_consistency(self, results: List[Dict[str, Any]], report: Dict[str, Any]):
        """Validate consistency across the batch of results."""
        if not results:
            return
        
        # Check for duplicate IDs
        ids = [r.get("id") for r in results if "id" in r]
        duplicate_ids = [id for id in set(ids) if ids.count(id) > 1]
        if duplicate_ids:
            report["batch_issues"].append(f"Duplicate result IDs: {duplicate_ids}")
        
        # Check for consistent evaluation version
        versions = set(r.get("metadata", {}).get("evaluation_version") for r in results)
        versions.discard(None)
        if len(versions) > 1:
            report["batch_issues"].append(f"Inconsistent evaluation versions: {list(versions)}")
        
        # Check for reasonable timestamp spread
        timestamps = [r.get("timestamp") for r in results if "timestamp" in r]
        if len(timestamps) > 1:
            try:
                parsed_timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
                time_span = max(parsed_timestamps) - min(parsed_timestamps)
                if time_span.total_seconds() > 86400:  # More than 24 hours
                    report["batch_issues"].append(f"Large timestamp spread: {time_span}")
            except Exception:
                report["batch_issues"].append("Unable to parse timestamps for consistency check")
    
    def generate_validation_report(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            results: List of result dictionaries
            output_path: Optional path to save the report
            
        Returns:
            HTML validation report
        """
        batch_validation = self.validate_batch(results)
        
        html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Result Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .error {{ color: #d32f2f; }}
        .warning {{ color: #f57c00; }}
        .success {{ color: #388e3c; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .status-valid {{ background-color: #e8f5e8; }}
        .status-invalid {{ background-color: #ffeaea; }}
    </style>
</head>
<body>
    <h1>Result Validation Report</h1>
    
    <div class="summary">
        <h2>Validation Summary</h2>
        <p><strong>Total Results:</strong> {batch_validation['total_results']}</p>
        <p><strong class="success">Valid Results:</strong> {batch_validation['valid_results']}</p>
        <p><strong class="error">Invalid Results:</strong> {batch_validation['invalid_results']}</p>
        <p><strong class="success">Analysis Tool Compatible:</strong> {batch_validation['analysis_compatible_results']}</p>
        <p><strong>Validation Rate:</strong> {batch_validation['valid_results']/max(batch_validation['total_results'], 1)*100:.1f}%</p>
        <p><strong>Generated:</strong> {datetime.now().isoformat()}</p>
    </div>
"""
        
        # Add batch issues if any
        if batch_validation["batch_issues"]:
            html_report += """
    <div class="error">
        <h3>Batch-Level Issues</h3>
        <ul>
"""
            for issue in batch_validation["batch_issues"]:
                html_report += f"            <li>{issue}</li>\n"
            html_report += """        </ul>
    </div>
"""
        
        # Add detailed validation results
        html_report += """
    <h3>Detailed Validation Results</h3>
    <table>
        <tr>
            <th>Index</th>
            <th>ID</th>
            <th>Status</th>
            <th>Analysis Compatible</th>
            <th>Errors</th>
            <th>Warnings</th>
        </tr>
"""
        
        for detail in batch_validation["validation_details"]:
            status_class = "status-valid" if detail["valid"] else "status-invalid"
            status_text = "Valid" if detail["valid"] else "Invalid"
            compatible_text = "Yes" if detail["analysis_compatible"] else "No"
            
            html_report += f"""
        <tr class="{status_class}">
            <td>{detail['index']}</td>
            <td>{detail['id']}</td>
            <td>{status_text}</td>
            <td>{compatible_text}</td>
            <td>{detail['error_count']}</td>
            <td>{detail['warning_count']}</td>
        </tr>
"""
        
        html_report += """
    </table>
</body>
</html>
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_report)
            logger.info(f"Validation report saved to {output_path}")
        
        return html_report


def main():
    """CLI interface for result validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate evaluation results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--output", "-o", help="Output path for validation report")
    parser.add_argument("--format", choices=["json", "html"], default="html", 
                       help="Output format for validation report")
    
    args = parser.parse_args()
    
    # Load results
    try:
        with open(args.results_file, 'r') as f:
            data = json.load(f)
        
        # Handle different result file formats
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict) and "results" in data:
            results = data["results"]
        else:
            results = [data]
        
    except Exception as e:
        logger.error(f"Error loading results file: {e}")
        return 1
    
    # Validate results
    validator = ResultValidator()
    
    if args.format == "json":
        validation_report = validator.validate_batch(results)
        output_path = args.output or args.results_file.replace(".json", "_validation.json")
        
        with open(output_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        print(f"Validation report saved to {output_path}")
        
    else:  # HTML format
        output_path = args.output or args.results_file.replace(".json", "_validation.html")
        validator.generate_validation_report(results, output_path)
        print(f"Validation report saved to {output_path}")
    
    # Print summary to console
    batch_validation = validator.validate_batch(results)
    print(f"\nValidation Summary:")
    print(f"  Total Results: {batch_validation['total_results']}")
    print(f"  Valid: {batch_validation['valid_results']}")
    print(f"  Invalid: {batch_validation['invalid_results']}")
    print(f"  Analysis Compatible: {batch_validation['analysis_compatible_results']}")
    
    return 0 if batch_validation['invalid_results'] == 0 else 1


if __name__ == "__main__":
    exit(main())