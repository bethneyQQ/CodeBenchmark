#!/usr/bin/env python3
"""
Test script for result output standardization functionality.
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from result_validator import ResultValidator
from result_standardizer import ResultStandardizer
from result_aggregator import ResultAggregator
from result_output_integration import ResultOutputManager

def create_test_results():
    """Create sample test results for validation."""
    return [
        {
            "id": "st_0001",
            "model": "test-model-v1",
            "config": "minimal_context|temperature=0.0",
            "prediction": "def hello(): return 'world'",
            "metrics": {
                "exact_match": 0.0,
                "codebleu": 0.75,
                "pass_at_1": 1.0,
                "syntax_valid": 1.0,
                "cyclomatic_complexity": 1.0,
                "security_score": 0.95
            },
            "runtime": {
                "time_s": 0.45,
                "exit_code": 0,
                "peak_memory_mb": 12.5
            },
            "seed": 1234,
            "commit": "abc123",
            "requirements": "test-requirements-v1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "scenario": "code_completion",
                "difficulty": "simple",
                "language": "python",
                "context_mode": "minimal_context",
                "evaluation_version": "1.0.0"
            }
        },
        {
            "id": "st_0002",
            "model": "test-model-v1",
            "config": "full_context|temperature=0.2",
            "prediction": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "metrics": {
                "exact_match": 1.0,
                "codebleu": 0.95,
                "pass_at_1": 1.0,
                "syntax_valid": 1.0,
                "cyclomatic_complexity": 3.0,
                "security_score": 1.0
            },
            "runtime": {
                "time_s": 1.2,
                "exit_code": 0,
                "peak_memory_mb": 18.3
            },
            "seed": 5678,
            "commit": "def456",
            "requirements": "test-requirements-v1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "scenario": "algorithm_implementation",
                "difficulty": "intermediate",
                "language": "python",
                "context_mode": "full_context",
                "evaluation_version": "1.0.0"
            }
        }
    ]

def test_result_validator():
    """Test result validator functionality."""
    print("Testing Result Validator...")
    
    validator = ResultValidator()
    test_results = create_test_results()
    
    # Test single result validation
    validation_report = validator.validate_result(test_results[0])
    
    if validation_report["valid"]:
        print("  ✅ Single result validation: PASS")
    else:
        print("  ❌ Single result validation: FAIL")
        print(f"    Errors: {validation_report['errors']}")
        return False
    
    # Test batch validation
    batch_report = validator.validate_batch(test_results)
    
    if batch_report["valid_results"] == len(test_results):
        print("  ✅ Batch validation: PASS")
    else:
        print("  ❌ Batch validation: FAIL")
        print(f"    Valid: {batch_report['valid_results']}/{batch_report['total_results']}")
        return False
    
    return True

def test_result_standardizer():
    """Test result standardizer functionality."""
    print("Testing Result Standardizer...")
    
    standardizer = ResultStandardizer()
    test_results = create_test_results()
    
    # Test single result standardization
    standardized = standardizer.standardize_result(test_results[0])
    
    if "timestamp" in standardized and "commit" in standardized:
        print("  ✅ Single result standardization: PASS")
    else:
        print("  ❌ Single result standardization: FAIL")
        return False
    
    # Test aggregation
    aggregated = standardizer.aggregate_results(test_results)
    
    if "aggregated_metrics" in aggregated and "summary" in aggregated:
        print("  ✅ Result aggregation: PASS")
    else:
        print("  ❌ Result aggregation: FAIL")
        return False
    
    return True

def test_result_aggregator():
    """Test result aggregator functionality."""
    print("Testing Result Aggregator...")
    
    aggregator = ResultAggregator()
    test_results = create_test_results()
    
    # Test basic aggregation
    aggregated = aggregator.aggregate_results(test_results)
    
    if "basic_aggregation" in aggregated and "dimensional_breakdowns" in aggregated:
        print("  ✅ Basic aggregation: PASS")
    else:
        print("  ❌ Basic aggregation: FAIL")
        return False
    
    # Test custom grouping
    grouped = aggregator.aggregate_results(test_results, group_by=["model", "scenario"])
    
    if "custom_grouping" in grouped:
        print("  ✅ Custom grouping: PASS")
    else:
        print("  ❌ Custom grouping: FAIL")
        return False
    
    return True

def test_result_output_manager():
    """Test result output manager functionality."""
    print("Testing Result Output Manager...")
    
    manager = ResultOutputManager()
    test_results = create_test_results()
    
    # Test processing pipeline
    processing_report = manager.process_results(test_results)
    
    if "standardized_results" in processing_report and "validation_reports" in processing_report:
        print("  ✅ Processing pipeline: PASS")
    else:
        print("  ❌ Processing pipeline: FAIL")
        return False
    
    # Test analysis tool compatibility
    compatibility = manager.validate_analysis_tool_compatibility(test_results)
    
    if "overall_compatible" in compatibility:
        print("  ✅ Analysis tool compatibility check: PASS")
    else:
        print("  ❌ Analysis tool compatibility check: FAIL")
        return False
    
    return True

def test_export_functionality():
    """Test export functionality."""
    print("Testing Export Functionality...")
    
    standardizer = ResultStandardizer()
    test_results = create_test_results()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test JSON export
        json_path = temp_path / "test_results.json"
        try:
            standardizer.export_results(test_results, str(json_path), "json")
            if json_path.exists():
                print("  ✅ JSON export: PASS")
            else:
                print("  ❌ JSON export: FAIL - File not created")
                return False
        except Exception as e:
            print(f"  ❌ JSON export: FAIL - {e}")
            return False
        
        # Test CSV export
        csv_path = temp_path / "test_results.csv"
        try:
            standardizer.export_results(test_results, str(csv_path), "csv")
            if csv_path.exists():
                print("  ✅ CSV export: PASS")
            else:
                print("  ❌ CSV export: FAIL - File not created")
                return False
        except Exception as e:
            print(f"  ❌ CSV export: FAIL - {e}")
            return False
        
        # Test HTML export
        html_path = temp_path / "test_results.html"
        try:
            standardizer.export_results(test_results, str(html_path), "html")
            if html_path.exists():
                print("  ✅ HTML export: PASS")
            else:
                print("  ❌ HTML export: FAIL - File not created")
                return False
        except Exception as e:
            print(f"  ❌ HTML export: FAIL - {e}")
            return False
    
    return True

def main():
    """Run all result standardization tests."""
    print("🧪 Testing Result Output Standardization")
    print("=" * 50)
    
    tests = [
        ("Result Validator", test_result_validator),
        ("Result Standardizer", test_result_standardizer),
        ("Result Aggregator", test_result_aggregator),
        ("Result Output Manager", test_result_output_manager),
        ("Export Functionality", test_export_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All result standardization tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)