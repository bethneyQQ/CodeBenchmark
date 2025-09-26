#!/usr/bin/env python3
"""
Result Output Integration Module

Integrates all result output standardization components to ensure comprehensive
compliance with the structured JSON format specification and analysis tool compatibility.

Requirements addressed: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timezone

try:
    from .result_validator import ResultValidator
    from .result_standardizer import ResultStandardizer
    from .result_aggregator import ResultAggregator
except ImportError:
    # Handle direct execution
    from result_validator import ResultValidator
    from result_standardizer import ResultStandardizer
    from result_aggregator import ResultAggregator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultOutputManager:
    """
    Comprehensive result output management system that ensures all results
    follow the structured JSON format and are compatible with analysis tools.
    """
    
    def __init__(self):
        self.validator = ResultValidator()
        self.standardizer = ResultStandardizer()
        self.aggregator = ResultAggregator()
        self.schema_path = Path(__file__).parent / "result_schema.json"
        
        # Load schema for validation
        self.schema = self._load_schema()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load the JSON schema for result validation."""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load schema from {self.schema_path}: {e}")
            return {}
    
    def process_results(self, results: List[Dict[str, Any]], 
                       output_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process evaluation results through the complete standardization pipeline.
        
        Args:
            results: List of raw result dictionaries
            output_config: Configuration for output processing
            
        Returns:
            Processing report with standardized results and validation info
        """
        if not results:
            return {"error": "No results to process", "results": []}
        
        config = output_config or {}
        
        # Step 1: Validate raw results
        logger.info("Step 1: Validating raw results...")
        validation_report = self.validator.validate_batch(results)
        
        # Step 2: Standardize results
        logger.info("Step 2: Standardizing results...")
        standardized_results = []
        standardization_errors = []
        
        for i, result in enumerate(results):
            try:
                standardized = self.standardizer.standardize_result(result)
                standardized_results.append(standardized)
            except Exception as e:
                standardization_errors.append({
                    "index": i,
                    "id": result.get("id", f"unknown_{i}"),
                    "error": str(e)
                })
                logger.error(f"Standardization error for result {i}: {e}")
        
        # Step 3: Re-validate standardized results
        logger.info("Step 3: Re-validating standardized results...")
        final_validation = self.validator.validate_batch(standardized_results)
        
        # Step 4: Generate aggregation if requested
        aggregation_data = None
        if config.get("generate_aggregation", True):
            logger.info("Step 4: Generating aggregation...")
            try:
                aggregation_data = self.aggregator.aggregate_results(
                    standardized_results,
                    config.get("group_by")
                )
            except Exception as e:
                logger.error(f"Aggregation error: {e}")
        
        # Compile processing report
        processing_report = {
            "processing_metadata": {
                "input_count": len(results),
                "standardized_count": len(standardized_results),
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "config": config
            },
            "validation_reports": {
                "initial_validation": validation_report,
                "final_validation": final_validation
            },
            "standardization_report": {
                "successful": len(standardized_results),
                "errors": standardization_errors
            },
            "standardized_results": standardized_results,
            "aggregation_data": aggregation_data
        }
        
        return processing_report
    
    def export_complete_results(self, processing_report: Dict[str, Any], 
                              output_dir: str, formats: List[str] = None) -> Dict[str, str]:
        """
        Export complete results in multiple formats.
        
        Args:
            processing_report: Complete processing report
            output_dir: Output directory path
            formats: List of formats to export ("json", "csv", "html")
            
        Returns:
            Dictionary mapping format to output file path
        """
        formats = formats or ["json", "html"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        standardized_results = processing_report.get("standardized_results", [])
        
        # Export standardized results
        for format_type in formats:
            try:
                if format_type == "json":
                    output_path = output_dir / "standardized_results.json"
                    self.standardizer.export_results(standardized_results, str(output_path), "json")
                    exported_files["standardized_json"] = str(output_path)
                
                elif format_type == "csv":
                    output_path = output_dir / "standardized_results.csv"
                    self.standardizer.export_results(standardized_results, str(output_path), "csv")
                    exported_files["standardized_csv"] = str(output_path)
                
                elif format_type == "html":
                    output_path = output_dir / "standardized_results.html"
                    self.standardizer.export_results(standardized_results, str(output_path), "html")
                    exported_files["standardized_html"] = str(output_path)
                
            except Exception as e:
                logger.error(f"Export error for format {format_type}: {e}")
        
        # Export validation reports
        try:
            validation_path = output_dir / "validation_report.html"
            self.validator.generate_validation_report(standardized_results, str(validation_path))
            exported_files["validation_report"] = str(validation_path)
        except Exception as e:
            logger.error(f"Validation report export error: {e}")
        
        # Export aggregation summary
        aggregation_data = processing_report.get("aggregation_data")
        if aggregation_data:
            try:
                # JSON aggregation
                agg_json_path = output_dir / "aggregation_summary.json"
                with open(agg_json_path, 'w') as f:
                    json.dump(aggregation_data, f, indent=2)
                exported_files["aggregation_json"] = str(agg_json_path)
                
                # HTML summary report
                summary_html_path = output_dir / "summary_report.html"
                self.aggregator.generate_summary_report(aggregation_data, str(summary_html_path))
                exported_files["summary_html"] = str(summary_html_path)
                
            except Exception as e:
                logger.error(f"Aggregation export error: {e}")
        
        # Export complete processing report
        try:
            report_path = output_dir / "complete_processing_report.json"
            with open(report_path, 'w') as f:
                json.dump(processing_report, f, indent=2)
            exported_files["processing_report"] = str(report_path)
        except Exception as e:
            logger.error(f"Processing report export error: {e}")
        
        return exported_files
    
    def validate_analysis_tool_compatibility(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive validation for analysis tool compatibility.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Compatibility report
        """
        compatibility_report = {
            "overall_compatible": False,
            "compatibility_issues": [],
            "tool_compatibility": {},
            "recommendations": []
        }
        
        if not results:
            compatibility_report["compatibility_issues"].append("No results to validate")
            return compatibility_report
        
        # Check each analysis tool's requirements
        analysis_tools = {
            "model_comparison": {
                "required_fields": ["model", "metrics.exact_match", "metrics.codebleu", "metrics.pass_at_1"],
                "min_results": 2
            },
            "scenario_analysis": {
                "required_fields": ["metadata.scenario", "metrics.exact_match", "metrics.pass_at_1"],
                "min_results": 1
            },
            "difficulty_analysis": {
                "required_fields": ["metadata.difficulty", "metrics.exact_match", "metrics.pass_at_1"],
                "min_results": 1
            },
            "context_impact": {
                "required_fields": ["metadata.context_mode", "metrics.exact_match", "metrics.codebleu"],
                "min_results": 2
            },
            "language_analysis": {
                "required_fields": ["metadata.language", "metrics.syntax_valid", "metrics.pass_at_1"],
                "min_results": 1
            }
        }
        
        for tool_name, requirements in analysis_tools.items():
            tool_compatible = True
            tool_issues = []
            
            # Check minimum results requirement
            if len(results) < requirements["min_results"]:
                tool_compatible = False
                tool_issues.append(f"Insufficient results: {len(results)} < {requirements['min_results']}")
            
            # Check required fields
            missing_fields = []
            for field in requirements["required_fields"]:
                field_present = all(self._has_field(result, field) for result in results)
                if not field_present:
                    missing_fields.append(field)
            
            if missing_fields:
                tool_compatible = False
                tool_issues.append(f"Missing required fields: {missing_fields}")
            
            compatibility_report["tool_compatibility"][tool_name] = {
                "compatible": tool_compatible,
                "issues": tool_issues
            }
            
            if not tool_compatible:
                compatibility_report["compatibility_issues"].extend(
                    [f"{tool_name}: {issue}" for issue in tool_issues]
                )
        
        # Overall compatibility
        compatibility_report["overall_compatible"] = all(
            tool_data["compatible"] 
            for tool_data in compatibility_report["tool_compatibility"].values()
        )
        
        # Generate recommendations
        if not compatibility_report["overall_compatible"]:
            compatibility_report["recommendations"] = self._generate_compatibility_recommendations(
                compatibility_report["tool_compatibility"]
            )
        
        return compatibility_report
    
    def _has_field(self, data: Dict[str, Any], field_path: str) -> bool:
        """Check if a nested field exists in the data."""
        parts = field_path.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        
        return current is not None
    
    def _generate_compatibility_recommendations(self, tool_compatibility: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving compatibility."""
        recommendations = []
        
        for tool_name, tool_data in tool_compatibility.items():
            if not tool_data["compatible"]:
                for issue in tool_data["issues"]:
                    if "Missing required fields" in issue:
                        recommendations.append(
                            f"Add missing fields for {tool_name}: ensure all results include required metadata and metrics"
                        )
                    elif "Insufficient results" in issue:
                        recommendations.append(
                            f"Increase result count for {tool_name}: collect more evaluation results"
                        )
        
        # Add general recommendations
        if recommendations:
            recommendations.extend([
                "Ensure all results follow the standardized JSON schema",
                "Include complete metadata for all results (scenario, difficulty, language, context_mode)",
                "Validate results using the ResultValidator before analysis",
                "Use the ResultStandardizer to ensure consistent formatting"
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def create_analysis_ready_dataset(self, results: List[Dict[str, Any]], 
                                    output_path: str) -> Dict[str, Any]:
        """
        Create an analysis-ready dataset with all necessary components.
        
        Args:
            results: List of result dictionaries
            output_path: Path to save the analysis-ready dataset
            
        Returns:
            Dataset creation report
        """
        # Process results through complete pipeline
        processing_report = self.process_results(results, {
            "generate_aggregation": True,
            "group_by": ["model", "scenario", "difficulty"]
        })
        
        # Validate analysis tool compatibility
        compatibility_report = self.validate_analysis_tool_compatibility(
            processing_report.get("standardized_results", [])
        )
        
        # Create analysis-ready dataset
        analysis_dataset = {
            "dataset_metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_results": len(processing_report.get("standardized_results", [])),
                "analysis_ready": compatibility_report["overall_compatible"],
                "schema_version": "1.0.0"
            },
            "results": processing_report.get("standardized_results", []),
            "aggregation_summary": processing_report.get("aggregation_data"),
            "validation_summary": processing_report.get("validation_reports", {}).get("final_validation"),
            "compatibility_report": compatibility_report,
            "schema": self.schema
        }
        
        # Save dataset
        try:
            with open(output_path, 'w') as f:
                json.dump(analysis_dataset, f, indent=2)
            
            creation_report = {
                "success": True,
                "output_path": output_path,
                "dataset_size": len(analysis_dataset["results"]),
                "analysis_ready": compatibility_report["overall_compatible"],
                "issues": compatibility_report.get("compatibility_issues", [])
            }
            
        except Exception as e:
            creation_report = {
                "success": False,
                "error": str(e),
                "dataset_size": 0,
                "analysis_ready": False
            }
        
        return creation_report


def main():
    """CLI interface for result output integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate and standardize evaluation results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--output-dir", "-o", required=True, 
                       help="Output directory for processed results")
    parser.add_argument("--formats", nargs="+", choices=["json", "csv", "html"], 
                       default=["json", "html"], help="Export formats")
    parser.add_argument("--group-by", nargs="+", 
                       help="Fields to group by for aggregation")
    parser.add_argument("--analysis-ready", action="store_true",
                       help="Create analysis-ready dataset")
    
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
    
    # Process results
    manager = ResultOutputManager()
    
    config = {
        "generate_aggregation": True,
        "group_by": args.group_by
    }
    
    processing_report = manager.process_results(results, config)
    
    # Export results
    exported_files = manager.export_complete_results(
        processing_report, 
        args.output_dir, 
        args.formats
    )
    
    # Create analysis-ready dataset if requested
    if args.analysis_ready:
        dataset_path = Path(args.output_dir) / "analysis_ready_dataset.json"
        creation_report = manager.create_analysis_ready_dataset(results, str(dataset_path))
        
        if creation_report["success"]:
            exported_files["analysis_dataset"] = str(dataset_path)
            print(f"Analysis-ready dataset created: {dataset_path}")
        else:
            print(f"Failed to create analysis dataset: {creation_report.get('error')}")
    
    # Print summary
    print(f"\nProcessing Summary:")
    print(f"  Input Results: {processing_report['processing_metadata']['input_count']}")
    print(f"  Standardized Results: {processing_report['processing_metadata']['standardized_count']}")
    print(f"  Exported Files: {len(exported_files)}")
    
    for file_type, file_path in exported_files.items():
        print(f"    {file_type}: {file_path}")
    
    # Check for issues
    final_validation = processing_report.get("validation_reports", {}).get("final_validation", {})
    if final_validation.get("invalid_results", 0) > 0:
        print(f"\nWarning: {final_validation['invalid_results']} results failed final validation")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())