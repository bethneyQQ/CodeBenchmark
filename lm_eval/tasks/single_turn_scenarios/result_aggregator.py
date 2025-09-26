#!/usr/bin/env python3
"""
Result Aggregation Module

Provides comprehensive result aggregation and summary reporting functionality
for analysis tool compatibility and cross-evaluation comparison.

Requirements addressed: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import json
import statistics
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from pathlib import Path
import logging
from collections import defaultdict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultAggregator:
    """Aggregates and summarizes evaluation results across multiple dimensions."""
    
    def __init__(self):
        self.aggregation_functions = {
            "mean": statistics.mean,
            "median": statistics.median,
            "std": lambda x: statistics.stdev(x) if len(x) > 1 else 0.0,
            "min": min,
            "max": max,
            "count": len
        }
    
    def aggregate_results(self, results: List[Dict[str, Any]], 
                         group_by: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Aggregate results with comprehensive statistical analysis.
        
        Args:
            results: List of result dictionaries
            group_by: Optional list of fields to group by (e.g., ['model', 'scenario'])
            
        Returns:
            Comprehensive aggregation report
        """
        if not results:
            return {"error": "No results to aggregate"}
        
        # Basic aggregation
        basic_aggregation = self._create_basic_aggregation(results)
        
        # Dimensional breakdowns
        dimensional_breakdowns = self._create_dimensional_breakdowns(results)
        
        # Statistical analysis
        statistical_analysis = self._create_statistical_analysis(results)
        
        # Performance trends
        performance_trends = self._analyze_performance_trends(results)
        
        # Custom grouping if specified
        custom_grouping = {}
        if group_by:
            custom_grouping = self._create_custom_grouping(results, group_by)
        
        # Correlation analysis
        correlation_analysis = self._analyze_metric_correlations(results)
        
        return {
            "aggregation_metadata": {
                "total_results": len(results),
                "aggregation_timestamp": datetime.now(timezone.utc).isoformat(),
                "group_by_fields": group_by or [],
                "aggregation_version": "1.0.0"
            },
            "basic_aggregation": basic_aggregation,
            "dimensional_breakdowns": dimensional_breakdowns,
            "statistical_analysis": statistical_analysis,
            "performance_trends": performance_trends,
            "custom_grouping": custom_grouping,
            "correlation_analysis": correlation_analysis
        }
    
    def _create_basic_aggregation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create basic metric aggregations."""
        metric_values = defaultdict(list)
        
        # Collect all metric values
        for result in results:
            metrics = result.get("metrics", {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_values[metric].append(value)
        
        # Calculate aggregations
        aggregated_metrics = {}
        for metric, values in metric_values.items():
            if values:
                aggregated_metrics[metric] = {
                    func_name: round(func(values), 4) 
                    for func_name, func in self.aggregation_functions.items()
                }
                
                # Add percentiles
                aggregated_metrics[metric].update({
                    "p25": round(np.percentile(values, 25), 4),
                    "p75": round(np.percentile(values, 75), 4),
                    "p90": round(np.percentile(values, 90), 4),
                    "p95": round(np.percentile(values, 95), 4)
                })
        
        return aggregated_metrics
    
    def _create_dimensional_breakdowns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create breakdowns by various dimensions."""
        dimensions = {
            "by_scenario": "metadata.scenario",
            "by_difficulty": "metadata.difficulty", 
            "by_language": "metadata.language",
            "by_model": "model",
            "by_context_mode": "metadata.context_mode"
        }
        
        breakdowns = {}
        
        for dimension_name, field_path in dimensions.items():
            breakdowns[dimension_name] = self._create_dimension_breakdown(results, field_path)
        
        return breakdowns
    
    def _create_dimension_breakdown(self, results: List[Dict[str, Any]], field_path: str) -> Dict[str, Any]:
        """Create breakdown for a specific dimension."""
        grouped_results = defaultdict(list)
        
        # Group results by dimension value
        for result in results:
            value = self._get_nested_value(result, field_path)
            if value is not None:
                grouped_results[str(value)].append(result)
        
        # Aggregate each group
        dimension_breakdown = {}
        for group_value, group_results in grouped_results.items():
            dimension_breakdown[group_value] = self._aggregate_group(group_results)
        
        return dimension_breakdown
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        parts = field_path.split(".")
        value = data
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    def _aggregate_group(self, group_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics for a group of results."""
        if not group_results:
            return {}
        
        metric_values = defaultdict(list)
        
        # Collect metric values for the group
        for result in group_results:
            metrics = result.get("metrics", {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_values[metric].append(value)
        
        # Calculate group statistics
        group_stats = {
            "count": len(group_results),
            "metrics": {}
        }
        
        key_metrics = ["exact_match", "codebleu", "pass_at_1", "syntax_valid"]
        
        for metric in key_metrics:
            if metric in metric_values and metric_values[metric]:
                values = metric_values[metric]
                group_stats["metrics"][metric] = {
                    "mean": round(statistics.mean(values), 4),
                    "median": round(statistics.median(values), 4),
                    "std": round(statistics.stdev(values) if len(values) > 1 else 0.0, 4),
                    "count": len(values)
                }
        
        return group_stats
    
    def _create_statistical_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive statistical analysis."""
        analysis = {
            "distribution_analysis": {},
            "outlier_detection": {},
            "confidence_intervals": {}
        }
        
        # Collect metric values
        metric_values = defaultdict(list)
        for result in results:
            metrics = result.get("metrics", {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_values[metric].append(value)
        
        # Analyze distributions
        for metric, values in metric_values.items():
            if len(values) >= 3:  # Need minimum samples for analysis
                analysis["distribution_analysis"][metric] = self._analyze_distribution(values)
                analysis["outlier_detection"][metric] = self._detect_outliers(values)
                analysis["confidence_intervals"][metric] = self._calculate_confidence_intervals(values)
        
        return analysis
    
    def _analyze_distribution(self, values: List[float]) -> Dict[str, Any]:
        """Analyze the distribution of metric values."""
        return {
            "skewness": round(self._calculate_skewness(values), 4),
            "kurtosis": round(self._calculate_kurtosis(values), 4),
            "is_normal": self._test_normality(values),
            "range": round(max(values) - min(values), 4),
            "iqr": round(np.percentile(values, 75) - np.percentile(values, 25), 4)
        }
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of the distribution."""
        n = len(values)
        if n < 3:
            return 0.0
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:
            return 0.0
        
        skew = sum(((x - mean_val) / std_val) ** 3 for x in values) / n
        return skew
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of the distribution."""
        n = len(values)
        if n < 4:
            return 0.0
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:
            return 0.0
        
        kurt = sum(((x - mean_val) / std_val) ** 4 for x in values) / n - 3
        return kurt
    
    def _test_normality(self, values: List[float]) -> bool:
        """Simple normality test based on skewness and kurtosis."""
        if len(values) < 8:
            return False  # Too few samples
        
        skew = abs(self._calculate_skewness(values))
        kurt = abs(self._calculate_kurtosis(values))
        
        # Simple thresholds for normality
        return skew < 2.0 and kurt < 7.0
    
    def _detect_outliers(self, values: List[float]) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        if len(values) < 4:
            return {"outliers": [], "outlier_count": 0}
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [v for v in values if v < lower_bound or v > upper_bound]
        
        return {
            "outliers": outliers,
            "outlier_count": len(outliers),
            "outlier_percentage": round(len(outliers) / len(values) * 100, 2),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4)
        }
    
    def _calculate_confidence_intervals(self, values: List[float], confidence: float = 0.95) -> Dict[str, Any]:
        """Calculate confidence intervals for the mean."""
        if len(values) < 2:
            return {"error": "Insufficient data for confidence interval"}
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        n = len(values)
        
        # Use t-distribution for small samples
        if n < 30:
            # Approximate t-value for 95% confidence
            t_value = 2.0 if n > 10 else 2.5
        else:
            t_value = 1.96  # z-value for 95% confidence
        
        margin_error = t_value * (std_val / (n ** 0.5))
        
        return {
            "mean": round(mean_val, 4),
            "margin_of_error": round(margin_error, 4),
            "lower_bound": round(mean_val - margin_error, 4),
            "upper_bound": round(mean_val + margin_error, 4),
            "confidence_level": confidence
        }
    
    def _analyze_performance_trends(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # Sort results by timestamp
        timestamped_results = []
        for result in results:
            if "timestamp" in result:
                try:
                    timestamp = datetime.fromisoformat(result["timestamp"].replace('Z', '+00:00'))
                    timestamped_results.append((timestamp, result))
                except ValueError:
                    continue
        
        if len(timestamped_results) < 2:
            return {"error": "Insufficient timestamped data for trend analysis"}
        
        timestamped_results.sort(key=lambda x: x[0])
        
        # Analyze trends for key metrics
        trends = {}
        key_metrics = ["exact_match", "codebleu", "pass_at_1", "syntax_valid"]
        
        for metric in key_metrics:
            metric_values = []
            timestamps = []
            
            for timestamp, result in timestamped_results:
                if metric in result.get("metrics", {}):
                    metric_values.append(result["metrics"][metric])
                    timestamps.append(timestamp)
            
            if len(metric_values) >= 2:
                trends[metric] = self._calculate_trend(metric_values, timestamps)
        
        return trends
    
    def _calculate_trend(self, values: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """Calculate trend statistics for a metric over time."""
        if len(values) < 2:
            return {"error": "Insufficient data for trend calculation"}
        
        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))  # Use indices as x-values
        
        # Calculate slope (trend)
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"
        
        return {
            "slope": round(slope, 6),
            "direction": trend_direction,
            "start_value": round(values[0], 4),
            "end_value": round(values[-1], 4),
            "change": round(values[-1] - values[0], 4),
            "percent_change": round((values[-1] - values[0]) / values[0] * 100, 2) if values[0] != 0 else 0,
            "time_span": str(timestamps[-1] - timestamps[0])
        }
    
    def _create_custom_grouping(self, results: List[Dict[str, Any]], group_by: List[str]) -> Dict[str, Any]:
        """Create custom grouping based on specified fields."""
        grouped_results = defaultdict(list)
        
        # Group results by combination of specified fields
        for result in results:
            group_key_parts = []
            for field in group_by:
                value = self._get_nested_value(result, field)
                group_key_parts.append(str(value) if value is not None else "unknown")
            
            group_key = "|".join(group_key_parts)
            grouped_results[group_key].append(result)
        
        # Aggregate each group
        custom_aggregation = {}
        for group_key, group_results in grouped_results.items():
            custom_aggregation[group_key] = self._aggregate_group(group_results)
        
        return custom_aggregation
    
    def _analyze_metric_correlations(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between different metrics."""
        # Collect metric values
        metric_data = defaultdict(list)
        
        for result in results:
            metrics = result.get("metrics", {})
            # Only include results that have all key metrics
            key_metrics = ["exact_match", "codebleu", "pass_at_1", "syntax_valid"]
            if all(metric in metrics for metric in key_metrics):
                for metric in key_metrics:
                    metric_data[metric].append(metrics[metric])
        
        if len(metric_data["exact_match"]) < 3:
            return {"error": "Insufficient data for correlation analysis"}
        
        # Calculate correlations
        correlations = {}
        metrics = list(metric_data.keys())
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i < j:  # Avoid duplicate pairs
                    correlation = self._calculate_correlation(
                        metric_data[metric1], 
                        metric_data[metric2]
                    )
                    correlations[f"{metric1}_vs_{metric2}"] = correlation
        
        return correlations
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> Dict[str, Any]:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return {"error": "Invalid data for correlation"}
        
        n = len(x_values)
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)
        
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        x_variance = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        y_variance = sum((y_values[i] - y_mean) ** 2 for i in range(n))
        
        denominator = (x_variance * y_variance) ** 0.5
        
        if denominator == 0:
            return {"correlation": 0.0, "strength": "no correlation"}
        
        correlation = numerator / denominator
        
        # Interpret correlation strength
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            strength = "very strong"
        elif abs_corr >= 0.6:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        return {
            "correlation": round(correlation, 4),
            "strength": strength,
            "direction": "positive" if correlation > 0 else "negative" if correlation < 0 else "none"
        }
    
    def generate_summary_report(self, aggregated_data: Dict[str, Any], 
                              output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            aggregated_data: Aggregated results data
            output_path: Optional path to save the report
            
        Returns:
            HTML summary report
        """
        metadata = aggregated_data.get("aggregation_metadata", {})
        basic_agg = aggregated_data.get("basic_aggregation", {})
        breakdowns = aggregated_data.get("dimensional_breakdowns", {})
        stats = aggregated_data.get("statistical_analysis", {})
        trends = aggregated_data.get("performance_trends", {})
        correlations = aggregated_data.get("correlation_analysis", {})
        
        html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Evaluation Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; 
                   border-left: 4px solid #007bff; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                       gap: 15px; margin: 15px 0; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px; 
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #007bff; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .correlation-positive {{ color: #28a745; }}
        .correlation-negative {{ color: #dc3545; }}
        .trend-improving {{ color: #28a745; }}
        .trend-declining {{ color: #dc3545; }}
        .trend-stable {{ color: #6c757d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Comprehensive Evaluation Summary Report</h1>
        <p><strong>Total Results:</strong> {metadata.get('total_results', 0)}</p>
        <p><strong>Generated:</strong> {metadata.get('aggregation_timestamp', 'Unknown')}</p>
        <p><strong>Version:</strong> {metadata.get('aggregation_version', '1.0.0')}</p>
    </div>
"""
        
        # Key Metrics Overview
        if basic_agg:
            html_report += """
    <div class="section">
        <h2>Key Metrics Overview</h2>
        <div class="metric-grid">
"""
            key_metrics = ["exact_match", "codebleu", "pass_at_1", "syntax_valid"]
            for metric in key_metrics:
                if metric in basic_agg:
                    data = basic_agg[metric]
                    html_report += f"""
            <div class="metric-card">
                <h3>{metric.replace('_', ' ').title()}</h3>
                <div class="metric-value">{data.get('mean', 0):.4f}</div>
                <p>Median: {data.get('median', 0):.4f}</p>
                <p>Std Dev: {data.get('std', 0):.4f}</p>
                <p>Range: {data.get('min', 0):.4f} - {data.get('max', 0):.4f}</p>
            </div>
"""
            
            html_report += """
        </div>
    </div>
"""
        
        # Performance by Scenario
        if breakdowns.get("by_scenario"):
            html_report += """
    <div class="section">
        <h2>Performance by Scenario</h2>
        <table>
            <tr><th>Scenario</th><th>Count</th><th>Exact Match</th><th>CodeBLEU</th><th>Pass@1</th><th>Syntax Valid</th></tr>
"""
            for scenario, data in breakdowns["by_scenario"].items():
                metrics = data.get("metrics", {})
                count = data.get("count", 0)
                
                html_report += f"""
            <tr>
                <td>{scenario}</td>
                <td>{count}</td>
                <td>{metrics.get('exact_match', {}).get('mean', 0):.4f}</td>
                <td>{metrics.get('codebleu', {}).get('mean', 0):.4f}</td>
                <td>{metrics.get('pass_at_1', {}).get('mean', 0):.4f}</td>
                <td>{metrics.get('syntax_valid', {}).get('mean', 0):.4f}</td>
            </tr>
"""
            
            html_report += """
        </table>
    </div>
"""
        
        # Performance Trends
        if trends and not trends.get("error"):
            html_report += """
    <div class="section">
        <h2>Performance Trends</h2>
        <table>
            <tr><th>Metric</th><th>Direction</th><th>Change</th><th>Percent Change</th><th>Time Span</th></tr>
"""
            for metric, trend_data in trends.items():
                if not trend_data.get("error"):
                    direction = trend_data.get("direction", "unknown")
                    direction_class = f"trend-{direction}"
                    
                    html_report += f"""
            <tr>
                <td>{metric}</td>
                <td class="{direction_class}">{direction.title()}</td>
                <td>{trend_data.get('change', 0):+.4f}</td>
                <td>{trend_data.get('percent_change', 0):+.2f}%</td>
                <td>{trend_data.get('time_span', 'Unknown')}</td>
            </tr>
"""
            
            html_report += """
        </table>
    </div>
"""
        
        # Metric Correlations
        if correlations and not correlations.get("error"):
            html_report += """
    <div class="section">
        <h2>Metric Correlations</h2>
        <table>
            <tr><th>Metric Pair</th><th>Correlation</th><th>Strength</th><th>Direction</th></tr>
"""
            for pair, corr_data in correlations.items():
                if not corr_data.get("error"):
                    correlation = corr_data.get("correlation", 0)
                    direction = corr_data.get("direction", "none")
                    direction_class = f"correlation-{direction}" if direction != "none" else ""
                    
                    html_report += f"""
            <tr>
                <td>{pair.replace('_vs_', ' vs ').replace('_', ' ').title()}</td>
                <td class="{direction_class}">{correlation:.4f}</td>
                <td>{corr_data.get('strength', 'unknown').title()}</td>
                <td>{direction.title()}</td>
            </tr>
"""
            
            html_report += """
        </table>
    </div>
"""
        
        html_report += """
</body>
</html>
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_report)
            logger.info(f"Summary report saved to {output_path}")
        
        return html_report


def main():
    """CLI interface for result aggregation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--output", "-o", help="Output path for aggregation report")
    parser.add_argument("--format", choices=["json", "html"], default="html",
                       help="Output format")
    parser.add_argument("--group-by", nargs="+", 
                       help="Fields to group by (e.g., model scenario)")
    
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
    
    # Aggregate results
    aggregator = ResultAggregator()
    aggregated_data = aggregator.aggregate_results(results, args.group_by)
    
    # Output results
    if args.format == "json":
        output_path = args.output or args.results_file.replace(".json", "_aggregated.json")
        
        with open(output_path, 'w') as f:
            json.dump(aggregated_data, f, indent=2)
        
        print(f"Aggregated results saved to {output_path}")
        
    else:  # HTML format
        output_path = args.output or args.results_file.replace(".json", "_summary.html")
        aggregator.generate_summary_report(aggregated_data, output_path)
        print(f"Summary report saved to {output_path}")
    
    # Print summary to console
    metadata = aggregated_data.get("aggregation_metadata", {})
    print(f"\nAggregation Summary:")
    print(f"  Total Results: {metadata.get('total_results', 0)}")
    print(f"  Group By Fields: {metadata.get('group_by_fields', [])}")
    print(f"  Generated: {metadata.get('aggregation_timestamp', 'Unknown')}")
    
    return 0


if __name__ == "__main__":
    exit(main())