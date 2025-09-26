"""Test runner for unit tests with comprehensive reporting."""

import pytest
import sys
import os
from pathlib import Path
import logging
from typing import List, Dict, Any

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class TestRunner:
    """Comprehensive test runner for unit tests."""
    
    def __init__(self, test_dir: Path = None):
        """Initialize test runner.
        
        Args:
            test_dir: Directory containing test files
        """
        self.test_dir = test_dir or Path(__file__).parent
        self.results = {}
        
    def discover_tests(self) -> List[Path]:
        """Discover all test files in the test directory.
        
        Returns:
            List of test file paths
        """
        test_files = []
        for test_file in self.test_dir.glob("test_*.py"):
            if test_file.name != "test_runner.py":
                test_files.append(test_file)
        
        logger.info(f"Discovered {len(test_files)} test files")
        return test_files
    
    def run_single_test_file(self, test_file: Path) -> Dict[str, Any]:
        """Run tests from a single test file.
        
        Args:
            test_file: Path to test file
            
        Returns:
            Test results dictionary
        """
        logger.info(f"Running tests from {test_file.name}")
        
        # Run pytest on the specific file
        exit_code = pytest.main([
            str(test_file),
            "-v",
            "--tb=short",
            "--no-header",
            f"--junitxml={self.test_dir}/results_{test_file.stem}.xml"
        ])
        
        result = {
            "file": test_file.name,
            "exit_code": exit_code,
            "status": "PASSED" if exit_code == 0 else "FAILED"
        }
        
        logger.info(f"Test file {test_file.name}: {result['status']}")
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all discovered tests.
        
        Returns:
            Comprehensive test results
        """
        logger.info("Starting comprehensive unit test run")
        
        test_files = self.discover_tests()
        results = {
            "total_files": len(test_files),
            "passed_files": 0,
            "failed_files": 0,
            "file_results": []
        }
        
        for test_file in test_files:
            try:
                file_result = self.run_single_test_file(test_file)
                results["file_results"].append(file_result)
                
                if file_result["status"] == "PASSED":
                    results["passed_files"] += 1
                else:
                    results["failed_files"] += 1
                    
            except Exception as e:
                logger.error(f"Error running tests from {test_file.name}: {e}")
                results["file_results"].append({
                    "file": test_file.name,
                    "exit_code": -1,
                    "status": "ERROR",
                    "error": str(e)
                })
                results["failed_files"] += 1
        
        self.results = results
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report.
        
        Returns:
            Formatted test report string
        """
        if not self.results:
            return "No test results available. Run tests first."
        
        report = []
        report.append("=" * 60)
        report.append("UNIT TEST RESULTS SUMMARY")
        report.append("=" * 60)
        report.append(f"Total test files: {self.results['total_files']}")
        report.append(f"Passed: {self.results['passed_files']}")
        report.append(f"Failed: {self.results['failed_files']}")
        report.append(f"Success rate: {(self.results['passed_files'] / self.results['total_files'] * 100):.1f}%")
        report.append("")
        
        report.append("DETAILED RESULTS:")
        report.append("-" * 40)
        
        for file_result in self.results["file_results"]:
            status_symbol = "✓" if file_result["status"] == "PASSED" else "✗"
            report.append(f"{status_symbol} {file_result['file']}: {file_result['status']}")
            
            if file_result.get("error"):
                report.append(f"  Error: {file_result['error']}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def run_with_coverage(self) -> Dict[str, Any]:
        """Run tests with coverage reporting.
        
        Returns:
            Test results with coverage information
        """
        logger.info("Running tests with coverage analysis")
        
        # Run pytest with coverage
        exit_code = pytest.main([
            str(self.test_dir),
            "-v",
            "--cov=../",
            "--cov-report=html:coverage_html",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage.xml"
        ])
        
        return {
            "exit_code": exit_code,
            "status": "PASSED" if exit_code == 0 else "FAILED",
            "coverage_html": "coverage_html/index.html",
            "coverage_xml": "coverage.xml"
        }


def run_specific_tests(test_patterns: List[str]) -> None:
    """Run specific tests matching patterns.
    
    Args:
        test_patterns: List of test patterns to match
    """
    logger.info(f"Running specific tests: {test_patterns}")
    
    args = []
    for pattern in test_patterns:
        args.extend(["-k", pattern])
    
    args.extend(["-v", "--tb=short"])
    
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        logger.info("Specific tests PASSED")
    else:
        logger.error("Specific tests FAILED")


def run_fast_tests() -> None:
    """Run only fast tests (excluding integration tests)."""
    logger.info("Running fast tests only")
    
    exit_code = pytest.main([
        str(Path(__file__).parent),
        "-v",
        "-m", "not integration",
        "--tb=short"
    ])
    
    if exit_code == 0:
        logger.info("Fast tests PASSED")
    else:
        logger.error("Fast tests FAILED")


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unit test runner for single_turn_scenarios")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--fast", action="store_true", help="Run only fast tests")
    parser.add_argument("--pattern", nargs="+", help="Run tests matching specific patterns")
    parser.add_argument("--file", help="Run tests from specific file")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.coverage:
        results = runner.run_with_coverage()
        print(f"Coverage test run: {results['status']}")
        
    elif args.fast:
        run_fast_tests()
        
    elif args.pattern:
        run_specific_tests(args.pattern)
        
    elif args.file:
        test_file = Path(args.file)
        if not test_file.exists():
            test_file = Path(__file__).parent / args.file
        
        if test_file.exists():
            result = runner.run_single_test_file(test_file)
            print(f"Test file {test_file.name}: {result['status']}")
        else:
            print(f"Test file not found: {args.file}")
            
    else:
        # Run all tests
        results = runner.run_all_tests()
        report = runner.generate_report()
        print(report)
        
        # Exit with appropriate code
        if results["failed_files"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()