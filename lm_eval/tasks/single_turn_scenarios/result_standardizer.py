#!/usr/bin/env python3
"""
Result Output Standardization Module

This module ensures all result outputs follow the structured JSON format specification
and provides validation, aggregation, and summary reporting functionality.

Requirements addressed: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import json
import jsonschema
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
import hashlib
import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultStandardizer:
    """Standardizes and validates evaluation results according to the specification."""
    
    def __init__(self):
        self.result_schema = self._load_result_schema()
        self.metrics_schema = self._load_metrics_schema()
    
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
                    "required": [
                        "exact_match", "codebleu", "pass_at_1", "syntax_valid"
                    ],
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
    
    def _load_metrics_schema(self) -> Dict[str, Any]:
        """Load the schema for metrics validation."""
        return {
            "basic_metrics": [
                "exact_match", "bleu_score", "codebleu", "rouge_l", "edit_distance"
            ],
            "code_quality_metrics": [
                "syntax_valid", "cyclomatic_complexity", "security_score", 
                "performance_score", "code_style_score"
            ],
            "functional_metrics": [
                "pass_at_1", "pass_at_5", "pass_at_10", "test_coverage", 
                "runtime_correctness", "memory_efficiency"
            ],
            "consistency_metrics": [
                "phase_coherence", "design_implementation_alignment", "information_flow"
            ]
        }
    
    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single result against the schema.
        
        Args:
            result: Result dictionary to validate
            
        Returns:
            Validation report with status and errors
        """
        validation_report = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "missing_metrics": [],
            "invalid_values": []
        }
        
        try:
            # Validate against JSON schema
            jsonschema.validate(result, self.result_schema)
            validation_report["valid"] = True
            
        except jsonschema.ValidationError as e:
            validation_report["errors"].append(f"Schema validation error: {e.message}")
            validation_report["valid"] = False
        
        # Additional validation checks
        self._validate_metrics_completeness(result, validation_report)
        self._validate_metric_values(result, validation_report)
        self._validate_consistency(result, validation_report)
        
        return validation_report
    
    def _validate_metrics_completeness(self, result: Dict[str, Any], report: Dict[str, Any]):
        """Check for missing required metrics."""
        if "metrics" not in result:
            report["errors"].append("Missing metrics section")
            return
        
        metrics = result["metrics"]
        all_metrics = (
            self.metrics_schema["basic_metrics"] + 
            self.metrics_schema["code_quality_metrics"] + 
            self.metrics_schema["functional_metrics"]
        )
        
        for metric in all_metrics:
            if metric not in metrics:
                report["missing_metrics"].append(metric)
        
        # Check for consistency metrics in complex scenarios
        if result.get("metadata", {}).get("difficulty") == "complex":
            for metric in self.metrics_schema["consistency_metrics"]:
                if metric not in metrics:
                    report["warnings"].append(f"Missing consistency metric for complex scenario: {metric}")
    
    def _validate_metric_values(self, result: Dict[str, Any], report: Dict[str, Any]):
        """Validate metric value ranges and consistency."""
        if "metrics" not in result:
            return
        
        metrics = result["metrics"]
        
        # Check value ranges
        for metric, value in metrics.items():
            if metric in ["exact_match", "bleu_score", "codebleu", "rouge_l", "edit_distance",
                         "syntax_valid", "security_score", "performance_score", "code_style_score",
                         "pass_at_1", "pass_at_5", "pass_at_10", "test_coverage", 
                         "runtime_correctness", "memory_efficiency", "phase_coherence",
                         "design_implementation_alignment", "information_flow"]:
                if not (0 <= value <= 1):
                    report["invalid_values"].append(f"{metric}: {value} (should be 0-1)")
        
        # Check logical consistency
        if "pass_at_1" in metrics and "pass_at_5" in metrics and "pass_at_10" in metrics:
            if not (metrics["pass_at_1"] <= metrics["pass_at_5"] <= metrics["pass_at_10"]):
                report["warnings"].append("Pass@K metrics are not monotonically increasing")
        
        if "syntax_valid" in metrics and metrics["syntax_valid"] == 0:
            if "pass_at_1" in metrics and metrics["pass_at_1"] > 0:
                report["warnings"].append("Invalid syntax but positive pass@1 score")
    
    def _validate_consistency(self, result: Dict[str, Any], report: Dict[str, Any]):
        """Validate internal consistency of the result."""
        # Check timestamp format
        try:
            datetime.fromisoformat(result["timestamp"].replace('Z', '+00:00'))
        except (ValueError, KeyError):
            report["errors"].append("Invalid timestamp format")
        
        # Check runtime consistency
        if "runtime" in result:
            runtime = result["runtime"]
            if runtime.get("exit_code", 0) != 0 and result.get("metrics", {}).get("pass_at_1", 0) > 0:
                report["warnings"].append("Non-zero exit code but positive pass@1 score")
    
    def standardize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize a result to ensure it follows the specification.
        
        Args:
            result: Raw result dictionary
            
        Returns:
            Standardized result dictionary
        """
        standardized = result.copy()
        
        # Ensure required fields exist
        if "timestamp" not in standardized:
            standardized["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        if "commit" not in standardized:
            standardized["commit"] = self._get_git_commit()
        
        if "requirements" not in standardized:
            standardized["requirements"] = self._get_requirements_hash()
        
        # Standardize metrics
        if "metrics" in standardized:
            standardized["metrics"] = self._standardize_metrics(standardized["metrics"])
        
        # Ensure runtime section is complete
        if "runtime" in standardized:
            standardized["runtime"] = self._standardize_runtime(standardized["runtime"])
        
        # Add metadata if missing
        if "metadata" not in standardized:
            standardized["metadata"] = {}
        
        standardized["metadata"]["evaluation_version"] = "1.0.0"
        
        return standardized
    
    def _standardize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize metrics section."""
        standardized = metrics.copy()
        
        # Ensure all required metrics exist with default values
        required_defaults = {
            "exact_match": 0.0,
            "codebleu": 0.0,
            "pass_at_1": 0.0,
            "syntax_valid": 0.0
        }
        
        for metric, default_value in required_defaults.items():
            if metric not in standardized:
                standardized[metric] = default_value
        
        # Round all metric values to 4 decimal places
        for metric, value in standardized.items():
            if isinstance(value, (int, float)):
                standardized[metric] = round(float(value), 4)
        
        return standardized
    
    def _standardize_runtime(self, runtime: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize runtime section."""
        standardized = runtime.copy()
        
        # Ensure required fields exist
        if "time_s" not in standardized:
            standardized["time_s"] = 0.0
        if "exit_code" not in standardized:
            standardized["exit_code"] = 0
        if "peak_memory_mb" not in standardized:
            standardized["peak_memory_mb"] = 0.0
        
        # Round numeric values
        standardized["time_s"] = round(float(standardized["time_s"]), 3)
        standardized["peak_memory_mb"] = round(float(standardized["peak_memory_mb"]), 2)
        
        return standardized
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, 
                text=True, 
                cwd=Path(__file__).parent
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return "unknown"
    
    def _get_requirements_hash(self) -> str:
        """Get hash of requirements file."""
        try:
            requirements_path = Path(__file__).parent / "requirements.txt"
            if requirements_path.exists():
                with open(requirements_path, 'r') as f:
                    content = f.read()
                return hashlib.md5(content.encode()).hexdigest()[:8]
        except Exception:
            pass
        return "unknown"
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple results into a summary report.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Aggregated summary report
        """
        if not results:
            return {"error": "No results to aggregate"}
        
        # Validate all results first
        validation_summary = {
            "total_results": len(results),
            "valid_results": 0,
            "invalid_results": 0,
            "validation_errors": []
        }
        
        valid_results = []
        for i, result in enumerate(results):
            validation = self.validate_result(result)
            if validation["valid"]:
                validation_summary["valid_results"] += 1
                valid_results.append(result)
            else:
                validation_summary["invalid_results"] += 1
                validation_summary["validation_errors"].append({
                    "result_index": i,
                    "errors": validation["errors"]
                })
        
        if not valid_results:
            return {
                "error": "No valid results to aggregate",
                "validation_summary": validation_summary
            }
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(valid_results)
        
        # Aggregate by scenario, difficulty, language
        breakdowns = self._create_breakdowns(valid_results)
        
        # Performance statistics
        performance_stats = self._calculate_performance_stats(valid_results)
        
        return {
            "summary": {
                "total_problems": len(valid_results),
                "models_tested": len(set(r["model"] for r in valid_results)),
                "scenarios_covered": len(set(r.get("metadata", {}).get("scenario", "unknown") for r in valid_results)),
                "languages_tested": len(set(r.get("metadata", {}).get("language", "unknown") for r in valid_results)),
                "evaluation_period": {
                    "start": min(r["timestamp"] for r in valid_results),
                    "end": max(r["timestamp"] for r in valid_results)
                }
            },
            "aggregated_metrics": aggregated_metrics,
            "breakdowns": breakdowns,
            "performance_statistics": performance_stats,
            "validation_summary": validation_summary,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all results."""
        import statistics
        
        all_metrics = {}
        
        # Collect all metric values
        for result in results:
            metrics = result.get("metrics", {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
        
        # Calculate statistics for each metric
        aggregated = {}
        for metric, values in all_metrics.items():
            if values:
                aggregated[metric] = {
                    "mean": round(statistics.mean(values), 4),
                    "median": round(statistics.median(values), 4),
                    "std": round(statistics.stdev(values) if len(values) > 1 else 0.0, 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "count": len(values)
                }
        
        return aggregated
    
    def _create_breakdowns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create breakdowns by various dimensions."""
        breakdowns = {
            "by_scenario": {},
            "by_difficulty": {},
            "by_language": {},
            "by_model": {},
            "by_context_mode": {}
        }
        
        # Group results by different dimensions
        for result in results:
            metadata = result.get("metadata", {})
            metrics = result.get("metrics", {})
            
            # Key metrics for breakdown
            key_metrics = ["exact_match", "codebleu", "pass_at_1", "syntax_valid"]
            
            # By scenario
            scenario = metadata.get("scenario", "unknown")
            if scenario not in breakdowns["by_scenario"]:
                breakdowns["by_scenario"][scenario] = {metric: [] for metric in key_metrics}
            for metric in key_metrics:
                if metric in metrics:
                    breakdowns["by_scenario"][scenario][metric].append(metrics[metric])
            
            # By difficulty
            difficulty = metadata.get("difficulty", "unknown")
            if difficulty not in breakdowns["by_difficulty"]:
                breakdowns["by_difficulty"][difficulty] = {metric: [] for metric in key_metrics}
            for metric in key_metrics:
                if metric in metrics:
                    breakdowns["by_difficulty"][difficulty][metric].append(metrics[metric])
            
            # By language
            language = metadata.get("language", "unknown")
            if language not in breakdowns["by_language"]:
                breakdowns["by_language"][language] = {metric: [] for metric in key_metrics}
            for metric in key_metrics:
                if metric in metrics:
                    breakdowns["by_language"][language][metric].append(metrics[metric])
            
            # By model
            model = result.get("model", "unknown")
            if model not in breakdowns["by_model"]:
                breakdowns["by_model"][model] = {metric: [] for metric in key_metrics}
            for metric in key_metrics:
                if metric in metrics:
                    breakdowns["by_model"][model][metric].append(metrics[metric])
            
            # By context mode
            context_mode = metadata.get("context_mode", "unknown")
            if context_mode not in breakdowns["by_context_mode"]:
                breakdowns["by_context_mode"][context_mode] = {metric: [] for metric in key_metrics}
            for metric in key_metrics:
                if metric in metrics:
                    breakdowns["by_context_mode"][context_mode][metric].append(metrics[metric])
        
        # Calculate averages for each breakdown
        import statistics
        for breakdown_type, breakdown_data in breakdowns.items():
            for category, metrics in breakdown_data.items():
                for metric, values in metrics.items():
                    if values:
                        breakdowns[breakdown_type][category][metric] = {
                            "mean": round(statistics.mean(values), 4),
                            "count": len(values)
                        }
                    else:
                        breakdowns[breakdown_type][category][metric] = {
                            "mean": 0.0,
                            "count": 0
                        }
        
        return breakdowns
    
    def _calculate_performance_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance statistics."""
        import statistics
        
        execution_times = []
        memory_usage = []
        success_rates = []
        
        for result in results:
            runtime = result.get("runtime", {})
            metrics = result.get("metrics", {})
            
            if "time_s" in runtime:
                execution_times.append(runtime["time_s"])
            
            if "peak_memory_mb" in runtime:
                memory_usage.append(runtime["peak_memory_mb"])
            
            if "pass_at_1" in metrics:
                success_rates.append(metrics["pass_at_1"])
        
        stats = {}
        
        if execution_times:
            stats["execution_time"] = {
                "mean_seconds": round(statistics.mean(execution_times), 3),
                "median_seconds": round(statistics.median(execution_times), 3),
                "max_seconds": round(max(execution_times), 3),
                "total_seconds": round(sum(execution_times), 3)
            }
        
        if memory_usage:
            stats["memory_usage"] = {
                "mean_mb": round(statistics.mean(memory_usage), 2),
                "median_mb": round(statistics.median(memory_usage), 2),
                "max_mb": round(max(memory_usage), 2),
                "total_mb": round(sum(memory_usage), 2)
            }
        
        if success_rates:
            stats["success_rate"] = {
                "overall": round(statistics.mean(success_rates), 4),
                "problems_solved": sum(1 for rate in success_rates if rate > 0),
                "total_problems": len(success_rates)
            }
        
        return stats
    
    def export_results(self, results: List[Dict[str, Any]], output_path: str, format: str = "json"):
        """
        Export results in various formats.
        
        Args:
            results: List of result dictionaries
            output_path: Output file path
            format: Export format ("json", "csv", "html")
        """
        if format == "json":
            self._export_json(results, output_path)
        elif format == "csv":
            self._export_csv(results, output_path)
        elif format == "html":
            self._export_html(results, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, results: List[Dict[str, Any]], output_path: str):
        """Export results as JSON."""
        aggregated = self.aggregate_results(results)
        
        export_data = {
            "results": results,
            "aggregated_summary": aggregated,
            "export_metadata": {
                "format": "json",
                "version": "1.0.0",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total_results": len(results)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(results)} results to {output_path}")
    
    def _export_csv(self, results: List[Dict[str, Any]], output_path: str):
        """Export results as CSV."""
        import csv
        
        if not results:
            return
        
        # Flatten results for CSV export
        fieldnames = set()
        flattened_results = []
        
        for result in results:
            flattened = {}
            
            # Basic fields
            for field in ["id", "model", "config", "seed", "commit", "timestamp"]:
                flattened[field] = result.get(field, "")
            
            # Metrics
            metrics = result.get("metrics", {})
            for metric, value in metrics.items():
                flattened[f"metric_{metric}"] = value
            
            # Runtime
            runtime = result.get("runtime", {})
            for field, value in runtime.items():
                if field != "security_violations":  # Skip complex fields
                    flattened[f"runtime_{field}"] = value
            
            # Metadata
            metadata = result.get("metadata", {})
            for field, value in metadata.items():
                flattened[f"metadata_{field}"] = value
            
            fieldnames.update(flattened.keys())
            flattened_results.append(flattened)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(flattened_results)
        
        logger.info(f"Exported {len(results)} results to CSV: {output_path}")
    
    def _export_html(self, results: List[Dict[str, Any]], output_path: str):
        """Export results as HTML report."""
        aggregated = self.aggregate_results(results)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Single Turn Scenarios Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .metric-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .breakdown {{ margin-top: 30px; }}
    </style>
</head>
<body>
    <h1>Single Turn Scenarios Evaluation Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Problems:</strong> {aggregated.get('summary', {}).get('total_problems', 0)}</p>
        <p><strong>Models Tested:</strong> {aggregated.get('summary', {}).get('models_tested', 0)}</p>
        <p><strong>Scenarios Covered:</strong> {aggregated.get('summary', {}).get('scenarios_covered', 0)}</p>
        <p><strong>Languages Tested:</strong> {aggregated.get('summary', {}).get('languages_tested', 0)}</p>
        <p><strong>Generated:</strong> {aggregated.get('generated_at', 'Unknown')}</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <h3>Key Metrics</h3>
            <table>
                <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std Dev</th></tr>
"""
        
        # Add key metrics to HTML
        key_metrics = ["exact_match", "codebleu", "pass_at_1", "syntax_valid"]
        aggregated_metrics = aggregated.get("aggregated_metrics", {})
        
        for metric in key_metrics:
            if metric in aggregated_metrics:
                data = aggregated_metrics[metric]
                html_content += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{data.get('mean', 0):.4f}</td>
                    <td>{data.get('median', 0):.4f}</td>
                    <td>{data.get('std', 0):.4f}</td>
                </tr>
"""
        
        html_content += """
            </table>
        </div>
    </div>
    
    <div class="breakdown">
        <h2>Performance by Scenario</h2>
        <table>
            <tr><th>Scenario</th><th>Exact Match</th><th>CodeBLEU</th><th>Pass@1</th><th>Count</th></tr>
"""
        
        # Add scenario breakdown
        scenario_breakdown = aggregated.get("breakdowns", {}).get("by_scenario", {})
        for scenario, metrics in scenario_breakdown.items():
            exact_match = metrics.get("exact_match", {})
            codebleu = metrics.get("codebleu", {})
            pass_at_1 = metrics.get("pass_at_1", {})
            
            html_content += f"""
            <tr>
                <td>{scenario}</td>
                <td>{exact_match.get('mean', 0):.4f}</td>
                <td>{codebleu.get('mean', 0):.4f}</td>
                <td>{pass_at_1.get('mean', 0):.4f}</td>
                <td>{exact_match.get('count', 0)}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="breakdown">
        <h2>Performance by Model</h2>
        <table>
            <tr><th>Model</th><th>Exact Match</th><th>CodeBLEU</th><th>Pass@1</th><th>Count</th></tr>
"""
        
        # Add model breakdown
        model_breakdown = aggregated.get("breakdowns", {}).get("by_model", {})
        for model, metrics in model_breakdown.items():
            exact_match = metrics.get("exact_match", {})
            codebleu = metrics.get("codebleu", {})
            pass_at_1 = metrics.get("pass_at_1", {})
            
            html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{exact_match.get('mean', 0):.4f}</td>
                <td>{codebleu.get('mean', 0):.4f}</td>
                <td>{pass_at_1.get('mean', 0):.4f}</td>
                <td>{exact_match.get('count', 0)}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="breakdown">
        <h2>Performance Statistics</h2>
        <div class="metrics">
"""
        
        # Add performance statistics
        perf_stats = aggregated.get("performance_statistics", {})
        
        if "execution_time" in perf_stats:
            exec_time = perf_stats["execution_time"]
            html_content += f"""
            <div class="metric-card">
                <h3>Execution Time</h3>
                <p><strong>Mean:</strong> {exec_time.get('mean_seconds', 0):.3f}s</p>
                <p><strong>Median:</strong> {exec_time.get('median_seconds', 0):.3f}s</p>
                <p><strong>Max:</strong> {exec_time.get('max_seconds', 0):.3f}s</p>
                <p><strong>Total:</strong> {exec_time.get('total_seconds', 0):.3f}s</p>
            </div>
"""
        
        if "memory_usage" in perf_stats:
            mem_usage = perf_stats["memory_usage"]
            html_content += f"""
            <div class="metric-card">
                <h3>Memory Usage</h3>
                <p><strong>Mean:</strong> {mem_usage.get('mean_mb', 0):.2f} MB</p>
                <p><strong>Median:</strong> {mem_usage.get('median_mb', 0):.2f} MB</p>
                <p><strong>Max:</strong> {mem_usage.get('max_mb', 0):.2f} MB</p>
                <p><strong>Total:</strong> {mem_usage.get('total_mb', 0):.2f} MB</p>
            </div>
"""
        
        if "success_rate" in perf_stats:
            success = perf_stats["success_rate"]
            html_content += f"""
            <div class="metric-card">
                <h3>Success Rate</h3>
                <p><strong>Overall:</strong> {success.get('overall', 0):.4f}</p>
                <p><strong>Problems Solved:</strong> {success.get('problems_solved', 0)}</p>
                <p><strong>Total Problems:</strong> {success.get('total_problems', 0)}</p>
            </div>
"""
        
        html_content += """
        </div>
    </div>
    
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Exported HTML report to {output_path}")


def main():
    """Command-line interface for result standardization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize and validate evaluation results")
    parser.add_argument("--input", required=True, help="Input results file (JSON)")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["json", "csv", "html"], default="json", help="Output format")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't export")
    parser.add_argument("--aggregate", action="store_true", help="Generate aggregated summary")
    
    args = parser.parse_args()
    
    standardizer = ResultStandardizer()
    
    # Load input results
    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        # Handle different input formats
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict) and "results" in data:
            results = data["results"]
        else:
            results = [data]
        
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Validate results
    validation_errors = []
    valid_results = []
    
    for i, result in enumerate(results):
        validation = standardizer.validate_result(result)
        if not validation["valid"]:
            validation_errors.append(f"Result {i}: {validation['errors']}")
        else:
            valid_results.append(standardizer.standardize_result(result))
    
    if validation_errors:
        logger.warning(f"Found {len(validation_errors)} validation errors:")
        for error in validation_errors[:10]:  # Show first 10 errors
            logger.warning(f"  {error}")
        if len(validation_errors) > 10:
            logger.warning(f"  ... and {len(validation_errors) - 10} more errors")
    
    logger.info(f"Validated {len(results)} results, {len(valid_results)} are valid")
    
    if args.validate_only:
        if validation_errors:
            sys.exit(1)
        else:
            logger.info("All results are valid")
            sys.exit(0)
    
    # Export results
    if args.output and valid_results:
        if args.aggregate:
            # Export aggregated summary
            aggregated = standardizer.aggregate_results(valid_results)
            with open(args.output, 'w') as f:
                json.dump(aggregated, f, indent=2)
            logger.info(f"Exported aggregated summary to {args.output}")
        else:
            # Export standardized results
            standardizer.export_results(valid_results, args.output, args.format)
    
    if validation_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()