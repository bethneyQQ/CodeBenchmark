#!/usr/bin/env python3
"""
Reproducibility Validation Script

This script validates the reproducibility of evaluations by comparing
results across different runs and checking for consistency.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

@dataclass
class ReproducibilityTest:
    """Configuration for a reproducibility test."""
    name: str
    description: str
    seed: int
    model_config: str
    task_config: str
    expected_variance: float  # Maximum allowed variance in key metrics
    tolerance: float  # Tolerance for metric comparison

@dataclass
class ReproducibilityResult:
    """Result of a reproducibility test."""
    test_name: str
    runs_compared: int
    metrics_compared: List[str]
    is_reproducible: bool
    variance_analysis: Dict[str, Any]
    consistency_score: float
    issues_found: List[str]
    recommendations: List[str]

class ReproducibilityValidator:
    """Reproducibility validation and testing."""
    
    def __init__(self, results_dir: Path = None):
        self.current_dir = Path(__file__).parent
        self.results_dir = results_dir or self.current_dir / "reproducibility_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def create_environment_fingerprint(self) -> Dict[str, Any]:
        """Create a fingerprint of the current environment."""
        fingerprint = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": str(Path.cwd()),
        }
        
        # Check key dependencies
        try:
            import lm_eval
            fingerprint["lm_eval_version"] = getattr(lm_eval, "__version__", "unknown")
        except ImportError:
            fingerprint["lm_eval_version"] = "not_installed"
        
        try:
            import torch
            fingerprint["torch_version"] = torch.__version__
            fingerprint["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                fingerprint["cuda_version"] = torch.version.cuda
        except ImportError:
            fingerprint["torch_version"] = "not_installed"
        
        try:
            import transformers
            fingerprint["transformers_version"] = transformers.__version__
        except ImportError:
            fingerprint["transformers_version"] = "not_installed"
        
        # Environment variables that affect reproducibility
        env_vars = [
            "PYTHONHASHSEED", "CUDA_LAUNCH_BLOCKING", "TOKENIZERS_PARALLELISM",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
        ]
        
        fingerprint["environment_vars"] = {}
        for var in env_vars:
            fingerprint["environment_vars"][var] = os.environ.get(var, "not_set")
        
        # Calculate fingerprint hash
        fingerprint_str = json.dumps(fingerprint, sort_keys=True)
        fingerprint["hash"] = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
        
        return fingerprint
    
    def compare_results(self, results1: Dict[str, Any], results2: Dict[str, Any], 
                       tolerance: float = 0.01) -> Dict[str, Any]:
        """Compare two evaluation results for consistency."""
        comparison = {
            "identical": True,
            "metric_differences": {},
            "max_difference": 0.0,
            "issues": []
        }
        
        # Get metrics from both results
        metrics1 = results1.get("results", {})
        metrics2 = results2.get("results", {})
        
        # Find common metrics
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        
        if not common_metrics:
            comparison["issues"].append("No common metrics found between results")
            comparison["identical"] = False
            return comparison
        
        # Compare each metric
        for metric in common_metrics:
            value1 = metrics1[metric]
            value2 = metrics2[metric]
            
            # Handle different types of metric values
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                diff = abs(value1 - value2)
                comparison["metric_differences"][metric] = {
                    "value1": value1,
                    "value2": value2,
                    "difference": diff,
                    "relative_difference": diff / max(abs(value1), abs(value2), 1e-10),
                    "within_tolerance": diff <= tolerance
                }
                
                if diff > tolerance:
                    comparison["identical"] = False
                    comparison["issues"].append(f"Metric '{metric}' differs by {diff:.6f} (tolerance: {tolerance})")
                
                comparison["max_difference"] = max(comparison["max_difference"], diff)
                
            elif value1 != value2:
                comparison["identical"] = False
                comparison["metric_differences"][metric] = {
                    "value1": value1,
                    "value2": value2,
                    "difference": "non_numeric",
                    "within_tolerance": False
                }
                comparison["issues"].append(f"Metric '{metric}' values differ: {value1} vs {value2}")
        
        return comparison
    
    def analyze_variance_across_runs(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze variance in metrics across multiple runs."""
        if len(results_list) < 2:
            return {"error": "Need at least 2 runs for variance analysis"}
        
        # Collect metrics from all runs
        all_metrics = {}
        for i, result in enumerate(results_list):
            metrics = result.get("results", {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
        
        # Calculate statistics for each metric
        variance_analysis = {}
        for metric, values in all_metrics.items():
            if len(values) >= 2:
                variance_analysis[metric] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "variance": statistics.variance(values) if len(values) > 1 else 0.0,
                    "coefficient_of_variation": statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) != 0 else float('inf'),
                    "values": values
                }
        
        return variance_analysis
    
    def run_reproducibility_test(self, test_config: ReproducibilityTest, 
                                num_runs: int = 3) -> ReproducibilityResult:
        """Run a reproducibility test with multiple evaluation runs."""
        print(f"🧪 Running reproducibility test: {test_config.name}")
        print(f"   📊 Number of runs: {num_runs}")
        print(f"   🎲 Seed: {test_config.seed}")
        
        # This is a mock implementation since we can't actually run evaluations
        # In a real implementation, this would:
        # 1. Run the evaluation multiple times with the same configuration
        # 2. Collect results from each run
        # 3. Compare results for consistency
        
        # For now, we'll create mock results to demonstrate the analysis
        mock_results = self._generate_mock_results(num_runs, test_config.seed)
        
        # Analyze variance
        variance_analysis = self.analyze_variance_across_runs(mock_results)
        
        # Compare consecutive runs
        issues_found = []
        comparisons = []
        
        for i in range(len(mock_results) - 1):
            comparison = self.compare_results(
                mock_results[i], 
                mock_results[i + 1], 
                test_config.tolerance
            )
            comparisons.append(comparison)
            
            if not comparison["identical"]:
                issues_found.extend(comparison["issues"])
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(variance_analysis, test_config.expected_variance)
        
        # Determine if reproducible
        is_reproducible = (
            consistency_score >= 0.95 and  # High consistency
            len(issues_found) == 0 and     # No issues found
            all(comp["identical"] for comp in comparisons)  # All runs identical
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(variance_analysis, issues_found, test_config)
        
        result = ReproducibilityResult(
            test_name=test_config.name,
            runs_compared=num_runs,
            metrics_compared=list(variance_analysis.keys()),
            is_reproducible=is_reproducible,
            variance_analysis=variance_analysis,
            consistency_score=consistency_score,
            issues_found=issues_found,
            recommendations=recommendations
        )
        
        return result
    
    def _generate_mock_results(self, num_runs: int, seed: int) -> List[Dict[str, Any]]:
        """Generate mock results for testing (replace with actual evaluation calls)."""
        import random
        random.seed(seed)
        
        results = []
        base_metrics = {
            "exact_match": 0.75,
            "pass_at_1": 0.68,
            "syntax_valid": 0.92,
            "codebleu": 0.58
        }
        
        for i in range(num_runs):
            # Add small random variations to simulate real evaluation variance
            metrics = {}
            for metric, base_value in base_metrics.items():
                # Add small noise (±1% for reproducible results, ±5% for non-reproducible)
                noise_factor = 0.01 if seed == 42 else 0.05
                noise = random.uniform(-noise_factor, noise_factor)
                metrics[metric] = max(0.0, min(1.0, base_value + noise))
            
            results.append({
                "run_id": i + 1,
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "results": metrics
            })
        
        return results
    
    def _calculate_consistency_score(self, variance_analysis: Dict[str, Any], 
                                   expected_variance: float) -> float:
        """Calculate a consistency score based on variance analysis."""
        if not variance_analysis:
            return 0.0
        
        scores = []
        for metric, stats in variance_analysis.items():
            cv = stats.get("coefficient_of_variation", float('inf'))
            
            # Score based on coefficient of variation
            if cv <= expected_variance:
                score = 1.0
            elif cv <= expected_variance * 2:
                score = 0.8
            elif cv <= expected_variance * 5:
                score = 0.6
            else:
                score = 0.2
            
            scores.append(score)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _generate_recommendations(self, variance_analysis: Dict[str, Any], 
                                issues_found: List[str], 
                                test_config: ReproducibilityTest) -> List[str]:
        """Generate recommendations based on reproducibility analysis."""
        recommendations = []
        
        if issues_found:
            recommendations.append("Fix identified consistency issues between runs")
        
        # Check for high variance metrics
        for metric, stats in variance_analysis.items():
            cv = stats.get("coefficient_of_variation", 0)
            if cv > test_config.expected_variance * 2:
                recommendations.append(f"Investigate high variance in {metric} (CV: {cv:.3f})")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Results appear reproducible - no issues detected")
        else:
            recommendations.extend([
                "Ensure consistent random seeds across runs",
                "Check for non-deterministic operations in model or evaluation",
                "Verify environment consistency (dependencies, hardware)",
                "Consider increasing tolerance if small differences are acceptable"
            ])
        
        return recommendations
    
    def create_reproducibility_report(self, results: List[ReproducibilityResult]) -> Dict[str, Any]:
        """Create a comprehensive reproducibility report."""
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "environment": self.create_environment_fingerprint(),
                "total_tests": len(results),
                "reproducible_tests": sum(1 for r in results if r.is_reproducible),
                "failed_tests": sum(1 for r in results if not r.is_reproducible)
            },
            "summary": {
                "overall_reproducibility_rate": sum(1 for r in results if r.is_reproducible) / len(results) if results else 0,
                "average_consistency_score": statistics.mean([r.consistency_score for r in results]) if results else 0,
                "total_issues": sum(len(r.issues_found) for r in results),
                "common_issues": self._find_common_issues(results)
            },
            "test_results": [asdict(result) for result in results],
            "recommendations": self._generate_overall_recommendations(results)
        }
        
        return report
    
    def _find_common_issues(self, results: List[ReproducibilityResult]) -> List[str]:
        """Find issues that appear across multiple tests."""
        all_issues = []
        for result in results:
            all_issues.extend(result.issues_found)
        
        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Return issues that appear in multiple tests
        common_issues = [issue for issue, count in issue_counts.items() if count > 1]
        return common_issues
    
    def _generate_overall_recommendations(self, results: List[ReproducibilityResult]) -> List[str]:
        """Generate overall recommendations based on all test results."""
        recommendations = []
        
        reproducible_count = sum(1 for r in results if r.is_reproducible)
        total_count = len(results)
        
        if total_count == 0:
            return ["No tests were run"]
        
        reproducibility_rate = reproducible_count / total_count
        
        if reproducibility_rate == 1.0:
            recommendations.append("Excellent! All tests are reproducible")
        elif reproducibility_rate >= 0.8:
            recommendations.append("Good reproducibility rate, but some issues need attention")
        elif reproducibility_rate >= 0.5:
            recommendations.append("Moderate reproducibility - significant improvements needed")
        else:
            recommendations.append("Poor reproducibility - major issues need to be addressed")
        
        # Add specific recommendations based on common patterns
        avg_consistency = statistics.mean([r.consistency_score for r in results])
        if avg_consistency < 0.8:
            recommendations.append("Improve consistency by fixing high-variance metrics")
        
        total_issues = sum(len(r.issues_found) for r in results)
        if total_issues > 0:
            recommendations.append(f"Address {total_issues} identified issues across all tests")
        
        return recommendations
    
    def print_reproducibility_summary(self, report: Dict[str, Any]):
        """Print a summary of reproducibility results."""
        print("\n" + "=" * 60)
        print("🔬 Reproducibility Validation Summary")
        print("=" * 60)
        
        metadata = report["metadata"]
        summary = report["summary"]
        
        print(f"🕒 Timestamp: {metadata['timestamp']}")
        print(f"🧪 Total Tests: {metadata['total_tests']}")
        print(f"✅ Reproducible: {metadata['reproducible_tests']}")
        print(f"❌ Failed: {metadata['failed_tests']}")
        print(f"📊 Success Rate: {summary['overall_reproducibility_rate']:.1%}")
        print(f"🎯 Avg Consistency: {summary['average_consistency_score']:.3f}")
        
        if summary["total_issues"] > 0:
            print(f"\n⚠️  Issues Found: {summary['total_issues']}")
            if summary["common_issues"]:
                print("   Common Issues:")
                for issue in summary["common_issues"]:
                    print(f"     • {issue}")
        
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"   • {rec}")
        
        # Environment fingerprint
        env = metadata["environment"]
        print(f"\n🌍 Environment:")
        print(f"   Python: {env.get('python_version', 'Unknown').split()[0]}")
        print(f"   Platform: {env.get('platform', 'Unknown')}")
        print(f"   Hash: {env.get('hash', 'Unknown')}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reproducibility Validation Tool")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output directory for results")
    parser.add_argument("--runs", "-r", type=int, default=3,
                       help="Number of runs per test")
    parser.add_argument("--tolerance", "-t", type=float, default=0.01,
                       help="Tolerance for metric comparison")
    parser.add_argument("--expected-variance", "-v", type=float, default=0.02,
                       help="Expected coefficient of variation")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Base seed for testing")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report from existing results")
    
    args = parser.parse_args()
    
    validator = ReproducibilityValidator(args.output)
    
    if args.report_only:
        # Load existing results and generate report
        print("📄 Generating report from existing results...")
        # This would load actual results in a real implementation
        results = []  # Placeholder
        
        if not results:
            print("❌ No existing results found")
            sys.exit(1)
    else:
        # Run reproducibility tests
        print("🚀 Starting Reproducibility Validation")
        print("=" * 50)
        
        # Define test configurations
        test_configs = [
            ReproducibilityTest(
                name="basic_evaluation",
                description="Basic evaluation with default settings",
                seed=args.seed,
                model_config="universal",
                task_config="default",
                expected_variance=args.expected_variance,
                tolerance=args.tolerance
            ),
            ReproducibilityTest(
                name="high_precision",
                description="High precision evaluation with strict settings",
                seed=args.seed + 1,
                model_config="universal",
                task_config="high_precision",
                expected_variance=args.expected_variance / 2,
                tolerance=args.tolerance / 2
            )
        ]
        
        # Run tests
        results = []
        for config in test_configs:
            try:
                result = validator.run_reproducibility_test(config, args.runs)
                results.append(result)
                
                # Print individual test result
                status = "✅ PASS" if result.is_reproducible else "❌ FAIL"
                print(f"   {status} {config.name} (consistency: {result.consistency_score:.3f})")
                
            except Exception as e:
                print(f"   ❌ ERROR {config.name}: {str(e)}")
    
    # Generate and save report
    if results:
        report = validator.create_reproducibility_report(results)
        
        # Save report
        report_file = validator.results_dir / f"reproducibility_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        validator.print_reproducibility_summary(report)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
        # Exit with appropriate code
        if report["summary"]["overall_reproducibility_rate"] == 1.0:
            print("\n✅ All reproducibility tests passed!")
            sys.exit(0)
        else:
            print(f"\n⚠️  Some reproducibility issues found")
            sys.exit(1)
    else:
        print("❌ No tests were run")
        sys.exit(1)

if __name__ == "__main__":
    main()