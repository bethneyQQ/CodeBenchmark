"""Comprehensive test runner for all single_turn_scenarios tests."""

import pytest
import sys
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """Comprehensive test runner for all test categories."""
    
    def __init__(self, test_root: Path = None):
        """Initialize the test runner.
        
        Args:
            test_root: Root directory containing all tests
        """
        self.test_root = test_root or Path(__file__).parent
        self.results = {}
        
    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all unit tests.
        
        Args:
            verbose: Enable verbose output
            
        Returns:
            Test results dictionary
        """
        logger.info("Running unit tests...")
        
        unit_test_dir = self.test_root / "unit"
        args = [
            str(unit_test_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "not integration and not performance and not security",
            f"--junitxml={self.test_root}/results_unit.xml"
        ]
        
        start_time = time.time()
        exit_code = pytest.main(args)
        duration = time.time() - start_time
        
        result = {
            "category": "unit",
            "exit_code": exit_code,
            "duration": duration,
            "status": "PASSED" if exit_code == 0 else "FAILED"
        }
        
        logger.info(f"Unit tests completed: {result['status']} in {duration:.2f}s")
        return result
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all integration tests.
        
        Args:
            verbose: Enable verbose output
            
        Returns:
            Test results dictionary
        """
        logger.info("Running integration tests...")
        
        integration_test_dir = self.test_root / "integration"
        args = [
            str(integration_test_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "integration",
            f"--junitxml={self.test_root}/results_integration.xml"
        ]
        
        start_time = time.time()
        exit_code = pytest.main(args)
        duration = time.time() - start_time
        
        result = {
            "category": "integration",
            "exit_code": exit_code,
            "duration": duration,
            "status": "PASSED" if exit_code == 0 else "FAILED"
        }
        
        logger.info(f"Integration tests completed: {result['status']} in {duration:.2f}s")
        return result
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all performance tests.
        
        Args:
            verbose: Enable verbose output
            
        Returns:
            Test results dictionary
        """
        logger.info("Running performance tests...")
        
        performance_test_dir = self.test_root / "performance"
        args = [
            str(performance_test_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "performance",
            f"--junitxml={self.test_root}/results_performance.xml"
        ]
        
        start_time = time.time()
        exit_code = pytest.main(args)
        duration = time.time() - start_time
        
        result = {
            "category": "performance",
            "exit_code": exit_code,
            "duration": duration,
            "status": "PASSED" if exit_code == 0 else "FAILED"
        }
        
        logger.info(f"Performance tests completed: {result['status']} in {duration:.2f}s")
        return result
    
    def run_security_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all security tests.
        
        Args:
            verbose: Enable verbose output
            
        Returns:
            Test results dictionary
        """
        logger.info("Running security tests...")
        
        security_test_dir = self.test_root / "security"
        args = [
            str(security_test_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "security",
            f"--junitxml={self.test_root}/results_security.xml"
        ]
        
        start_time = time.time()
        exit_code = pytest.main(args)
        duration = time.time() - start_time
        
        result = {
            "category": "security",
            "exit_code": exit_code,
            "duration": duration,
            "status": "PASSED" if exit_code == 0 else "FAILED"
        }
        
        logger.info(f"Security tests completed: {result['status']} in {duration:.2f}s")
        return result
    
    def run_all_tests(self, categories: List[str] = None, verbose: bool = False) -> Dict[str, Any]:
        """Run all test categories.
        
        Args:
            categories: List of test categories to run (default: all)
            verbose: Enable verbose output
            
        Returns:
            Comprehensive test results
        """
        if categories is None:
            categories = ["unit", "integration", "performance", "security"]
        
        logger.info(f"Running comprehensive test suite: {', '.join(categories)}")
        
        overall_start_time = time.time()
        results = {
            "categories": [],
            "total_duration": 0,
            "passed_categories": 0,
            "failed_categories": 0,
            "overall_status": "PASSED"
        }
        
        # Run each category
        for category in categories:
            try:
                if category == "unit":
                    result = self.run_unit_tests(verbose)
                elif category == "integration":
                    result = self.run_integration_tests(verbose)
                elif category == "performance":
                    result = self.run_performance_tests(verbose)
                elif category == "security":
                    result = self.run_security_tests(verbose)
                else:
                    logger.warning(f"Unknown test category: {category}")
                    continue
                
                results["categories"].append(result)
                
                if result["status"] == "PASSED":
                    results["passed_categories"] += 1
                else:
                    results["failed_categories"] += 1
                    results["overall_status"] = "FAILED"
                    
            except Exception as e:
                logger.error(f"Error running {category} tests: {e}")
                error_result = {
                    "category": category,
                    "exit_code": -1,
                    "duration": 0,
                    "status": "ERROR",
                    "error": str(e)
                }
                results["categories"].append(error_result)
                results["failed_categories"] += 1
                results["overall_status"] = "FAILED"
        
        results["total_duration"] = time.time() - overall_start_time
        self.results = results
        
        logger.info(f"Comprehensive test suite completed: {results['overall_status']} in {results['total_duration']:.2f}s")
        return results
    
    def run_with_coverage(self, categories: List[str] = None) -> Dict[str, Any]:
        """Run tests with coverage analysis.
        
        Args:
            categories: List of test categories to run
            
        Returns:
            Test results with coverage information
        """
        logger.info("Running tests with coverage analysis...")
        
        if categories is None:
            categories = ["unit", "integration"]  # Skip performance/security for coverage
        
        test_dirs = []
        for category in categories:
            test_dir = self.test_root / category
            if test_dir.exists():
                test_dirs.append(str(test_dir))
        
        args = [
            *test_dirs,
            "-v",
            "--cov=../",
            "--cov-report=html:coverage_html",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=70",  # Require 70% coverage
            f"--junitxml={self.test_root}/results_coverage.xml"
        ]
        
        start_time = time.time()
        exit_code = pytest.main(args)
        duration = time.time() - start_time
        
        result = {
            "type": "coverage",
            "exit_code": exit_code,
            "duration": duration,
            "status": "PASSED" if exit_code == 0 else "FAILED",
            "coverage_html": "coverage_html/index.html",
            "coverage_xml": "coverage.xml"
        }
        
        logger.info(f"Coverage analysis completed: {result['status']} in {duration:.2f}s")
        return result
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report.
        
        Returns:
            Formatted test report string
        """
        if not self.results:
            return "No test results available. Run tests first."
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE TEST RESULTS SUMMARY")
        report.append("=" * 80)
        report.append(f"Overall Status: {self.results['overall_status']}")
        report.append(f"Total Duration: {self.results['total_duration']:.2f} seconds")
        report.append(f"Categories Passed: {self.results['passed_categories']}")
        report.append(f"Categories Failed: {self.results['failed_categories']}")
        report.append("")
        
        report.append("DETAILED RESULTS BY CATEGORY:")
        report.append("-" * 50)
        
        for category_result in self.results["categories"]:
            status_symbol = "✓" if category_result["status"] == "PASSED" else "✗"
            report.append(f"{status_symbol} {category_result['category'].upper()}: {category_result['status']}")
            report.append(f"  Duration: {category_result['duration']:.2f}s")
            report.append(f"  Exit Code: {category_result['exit_code']}")
            
            if category_result.get("error"):
                report.append(f"  Error: {category_result['error']}")
            
            report.append("")
        
        # Performance summary
        if any(r["category"] == "performance" for r in self.results["categories"]):
            report.append("PERFORMANCE SUMMARY:")
            report.append("-" * 30)
            perf_result = next(r for r in self.results["categories"] if r["category"] == "performance")
            if perf_result["status"] == "PASSED":
                report.append("✓ All performance benchmarks passed")
            else:
                report.append("✗ Some performance benchmarks failed")
            report.append("")
        
        # Security summary
        if any(r["category"] == "security" for r in self.results["categories"]):
            report.append("SECURITY SUMMARY:")
            report.append("-" * 25)
            sec_result = next(r for r in self.results["categories"] if r["category"] == "security")
            if sec_result["status"] == "PASSED":
                report.append("✓ All security tests passed")
            else:
                report.append("✗ Some security tests failed")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_smoke_tests(self) -> Dict[str, Any]:
        """Run a quick smoke test to verify basic functionality.
        
        Returns:
            Smoke test results
        """
        logger.info("Running smoke tests...")
        
        # Run a subset of fast unit tests
        args = [
            str(self.test_root / "unit"),
            "-v",
            "-k", "test_load_dataset or test_exact_match or test_validate_config",
            "--tb=short",
            f"--junitxml={self.test_root}/results_smoke.xml"
        ]
        
        start_time = time.time()
        exit_code = pytest.main(args)
        duration = time.time() - start_time
        
        result = {
            "type": "smoke",
            "exit_code": exit_code,
            "duration": duration,
            "status": "PASSED" if exit_code == 0 else "FAILED"
        }
        
        logger.info(f"Smoke tests completed: {result['status']} in {duration:.2f}s")
        return result


def main():
    """Main entry point for the comprehensive test runner."""
    parser = argparse.ArgumentParser(description="Comprehensive test runner for single_turn_scenarios")
    
    parser.add_argument(
        "--categories", 
        nargs="+", 
        choices=["unit", "integration", "performance", "security"],
        default=["unit", "integration", "performance", "security"],
        help="Test categories to run"
    )
    
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run with coverage analysis"
    )
    
    parser.add_argument(
        "--smoke", 
        action="store_true", 
        help="Run only smoke tests"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--report", 
        action="store_true", 
        help="Generate detailed report"
    )
    
    parser.add_argument(
        "--output", 
        help="Output file for report"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = ComprehensiveTestRunner()
    
    try:
        if args.smoke:
            # Run smoke tests
            results = runner.run_smoke_tests()
            print(f"Smoke tests: {results['status']}")
            
        elif args.coverage:
            # Run with coverage
            results = runner.run_with_coverage(args.categories)
            print(f"Coverage tests: {results['status']}")
            if results['status'] == 'PASSED':
                print(f"Coverage report: {results['coverage_html']}")
            
        else:
            # Run comprehensive tests
            results = runner.run_all_tests(args.categories, args.verbose)
            
            if args.report:
                # Generate detailed report
                report = runner.generate_report()
                
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(report)
                    print(f"Report saved to: {args.output}")
                else:
                    print(report)
            else:
                # Simple summary
                print(f"Overall status: {results['overall_status']}")
                print(f"Duration: {results['total_duration']:.2f}s")
                print(f"Passed: {results['passed_categories']}/{len(results['categories'])}")
        
        # Exit with appropriate code
        if hasattr(results, 'get'):
            exit_code = 0 if results.get('overall_status') == 'PASSED' else 1
        else:
            exit_code = 0 if results['status'] == 'PASSED' else 1
            
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Test run interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Test run failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()