"""Performance tests for scalability and resource usage monitoring."""

import pytest
import time
import threading
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import load_dataset, process_docs, filter_by_metadata
from metrics import exact_match, bleu_score, syntax_validity, pass_at_k
from sandbox import SandboxExecutor
from config_manager import ConfigManager


class TestScalabilityPerformance:
    """Performance tests for scalability under load."""
    
    @pytest.mark.performance
    def test_dataset_loading_performance(self):
        """Test dataset loading performance with large datasets."""
        # Create a large mock dataset
        large_problem_set = []
        for i in range(1000):  # 1000 problems
            problem = {
                "id": f"perf_test_{i:04d}",
                "title": f"Performance Test Problem {i}",
                "language": "python",
                "scenario": "code_completion",
                "difficulty": "simple",
                "context_mode": "no_context",
                "prompt": f"Write a function for test case {i}",
                "reference": [f"def test_{i}(): pass"],
                "tests": [],
                "metadata": {
                    "time_limit_s": 10,
                    "memory_limit_mb": 100,
                    "seed": i,
                    "author": "perf_test",
                    "license": "MIT"
                }
            }
            large_problem_set.append(problem)
        
        # Mock the dataset loading with performance measurement
        with patch('utils.validate_problem_schema', return_value=True):
            with patch('datasets.Dataset.from_list') as mock_dataset:
                mock_ds = MagicMock()
                mock_ds.__len__.return_value = len(large_problem_set)
                mock_ds.__iter__.return_value = large_problem_set
                mock_dataset.return_value = mock_ds
                
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.readlines.return_value = [
                            json.dumps(p) + '\n' for p in large_problem_set
                        ]
                        
                        # Measure loading time
                        start_time = time.time()
                        dataset = load_dataset()
                        loading_time = time.time() - start_time
                        
                        # Performance assertions
                        assert len(dataset) == 1000
                        assert loading_time < 5.0, f"Dataset loading took {loading_time:.2f}s, should be < 5s"
    
    @pytest.mark.performance
    def test_concurrent_document_processing(self):
        """Test concurrent document processing performance."""
        # Create test documents
        test_docs = []
        for i in range(100):
            doc = {
                "id": f"concurrent_test_{i}",
                "prompt": f"Test prompt {i} with some content to process",
                "context_mode": "minimal_context",
                "language": "python",
                "scenario": "code_completion"
            }
            test_docs.append(doc)
        
        # Mock context loading
        mock_context = {
            "minimal_context": {
                "template": "{{prompt}}\n\nRequirements:\n- Follow best practices",
                "description": "Basic requirements"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_context):
            # Test sequential processing
            start_time = time.time()
            sequential_results = []
            for doc in test_docs:
                result = process_docs(doc)
                sequential_results.append(result)
            sequential_time = time.time() - start_time
            
            # Test concurrent processing
            start_time = time.time()
            concurrent_results = []
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_doc = {executor.submit(process_docs, doc): doc for doc in test_docs}
                
                for future in as_completed(future_to_doc):
                    result = future.result()
                    concurrent_results.append(result)
            
            concurrent_time = time.time() - start_time
            
            # Performance assertions
            assert len(sequential_results) == 100
            assert len(concurrent_results) == 100
            assert concurrent_time < sequential_time, f"Concurrent processing should be faster: {concurrent_time:.2f}s vs {sequential_time:.2f}s"
    
    @pytest.mark.performance
    def test_metrics_calculation_performance(self):
        """Test metrics calculation performance with large datasets."""
        # Generate test data
        predictions = [f"def function_{i}(x): return x + {i}" for i in range(500)]
        references = [f"def function_{i}(x): return x + {i}" for i in range(500)]
        
        # Test exact match performance
        start_time = time.time()
        em_score = exact_match(predictions, references)
        em_time = time.time() - start_time
        
        assert em_score == 1.0  # All should match
        assert em_time < 1.0, f"Exact match calculation took {em_time:.2f}s, should be < 1s"
        
        # Test BLEU score performance (mocked)
        with patch('metrics.sentence_bleu') as mock_bleu:
            mock_bleu.return_value = 0.95
            
            start_time = time.time()
            bleu_result = bleu_score(predictions[:100], references[:100])  # Smaller set for BLEU
            bleu_time = time.time() - start_time
            
            assert bleu_result == 0.95
            assert bleu_time < 2.0, f"BLEU calculation took {bleu_time:.2f}s, should be < 2s"
    
    @pytest.mark.performance
    def test_filtering_performance(self):
        """Test dataset filtering performance with large datasets."""
        # Create large mock dataset
        problems = []
        scenarios = ["code_completion", "bug_fix", "function_generation", "algorithm_implementation"]
        difficulties = ["simple", "intermediate", "complex"]
        languages = ["python", "javascript", "java", "cpp"]
        
        for i in range(2000):
            problem = {
                "id": f"filter_perf_{i}",
                "scenario": scenarios[i % len(scenarios)],
                "difficulty": difficulties[i % len(difficulties)],
                "language": languages[i % len(languages)],
                "prompt": f"Test {i}"
            }
            problems.append(problem)
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = len(problems)
        
        def mock_filter(func):
            filtered = [p for p in problems if func(p)]
            filtered_ds = MagicMock()
            filtered_ds.__len__.return_value = len(filtered)
            return filtered_ds
        
        mock_dataset.filter = mock_filter
        
        # Test filtering performance
        start_time = time.time()
        filtered_dataset = filter_by_metadata(mock_dataset, {"scenario": "code_completion"})
        filter_time = time.time() - start_time
        
        assert filter_time < 0.5, f"Filtering took {filter_time:.2f}s, should be < 0.5s"
        assert len(filtered_dataset) == 500  # 1/4 of the dataset


class TestResourceUsageMonitoring:
    """Tests for monitoring resource usage during evaluation."""
    
    @pytest.mark.performance
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring during evaluation."""
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        large_data = []
        for i in range(1000):
            problem = {
                "id": f"memory_test_{i}",
                "prompt": "x" * 1000,  # Large prompt
                "reference": ["y" * 1000],  # Large reference
                "metadata": {"data": list(range(100))}  # Additional data
            }
            large_data.append(problem)
        
        # Process all data
        with patch('utils.load_context_configs', return_value={"no_context": {"template": "{{prompt}}"}}):
            for problem in large_data:
                processed = process_docs(problem)
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable (< 100MB increase)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB, should be < 100MB"
        
        # Clean up
        del large_data
    
    @pytest.mark.performance
    def test_cpu_usage_monitoring(self):
        """Test CPU usage monitoring during intensive operations."""
        # Monitor CPU usage during intensive computation
        cpu_percentages = []
        
        def monitor_cpu():
            for _ in range(10):  # Monitor for 1 second
                cpu_percentages.append(psutil.cpu_percent(interval=0.1))
        
        # Start CPU monitoring in background
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform CPU-intensive operations
        predictions = []
        references = []
        
        for i in range(1000):
            pred = f"def complex_function_{i}(x, y, z): return x * y + z - {i}"
            ref = f"def complex_function_{i}(x, y, z): return x * y + z - {i}"
            predictions.append(pred)
            references.append(ref)
        
        # Calculate metrics (CPU intensive)
        em_score = exact_match(predictions, references)
        
        # Multiple syntax validations
        for pred in predictions[:100]:  # Limit to avoid excessive testing time
            syntax_validity(pred, "python")
        
        monitor_thread.join()
        
        # Analyze CPU usage
        avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
        max_cpu = max(cpu_percentages) if cpu_percentages else 0
        
        # CPU usage should be reasonable
        assert avg_cpu < 80, f"Average CPU usage was {avg_cpu:.1f}%, should be < 80%"
        assert max_cpu < 95, f"Maximum CPU usage was {max_cpu:.1f}%, should be < 95%"
    
    @pytest.mark.performance
    def test_sandbox_resource_limits(self):
        """Test that sandbox resource limits are properly enforced."""
        # Test memory limit enforcement
        memory_intensive_code = """
data = []
for i in range(100000):
    data.append([0] * 1000)  # Allocate large amounts of memory
print("Memory allocated")
"""
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                
                # Simulate memory limit exceeded
                mock_container.exec_run.side_effect = Exception("Memory limit exceeded")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 50 * 1024 * 1024},  # 50MB
                    'cpu_stats': {'cpu_usage': {'total_usage': 1000000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python", limits={"memory_limit_mb": 10})
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        result = executor.execute_code(memory_intensive_code, [])
                        
                        # Should fail due to resource limits
                        assert result.exit_code != 0
                        assert result.peak_memory > 0
    
    @pytest.mark.performance
    def test_concurrent_sandbox_execution(self):
        """Test concurrent sandbox execution performance."""
        test_codes = [
            f"print('Test {i}')" for i in range(10)
        ]
        
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                mock_container.exec_run.return_value = (0, b"Success")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 1024 * 1024},
                    'cpu_stats': {'cpu_usage': {'total_usage': 1000000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                # Test sequential execution
                start_time = time.time()
                sequential_results = []
                
                for code in test_codes:
                    executor = SandboxExecutor("python")
                    with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                        with patch.object(executor, '_cleanup'):
                            with patch('time.time', side_effect=[0, 0.1]):
                                result = executor.execute_code(code, [])
                                sequential_results.append(result)
                
                sequential_time = time.time() - start_time
                
                # Test concurrent execution
                start_time = time.time()
                concurrent_results = []
                
                def execute_code_wrapper(code):
                    executor = SandboxExecutor("python")
                    with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                        with patch.object(executor, '_cleanup'):
                            with patch('time.time', side_effect=[0, 0.1]):
                                return executor.execute_code(code, [])
                
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_code = {executor.submit(execute_code_wrapper, code): code for code in test_codes}
                    
                    for future in as_completed(future_to_code):
                        result = future.result()
                        concurrent_results.append(result)
                
                concurrent_time = time.time() - start_time
                
                # Performance assertions
                assert len(sequential_results) == 10
                assert len(concurrent_results) == 10
                # Concurrent should be faster (with proper mocking, times might be similar)
                assert concurrent_time <= sequential_time * 1.2  # Allow some overhead


class TestPerformanceBenchmarks:
    """Benchmark tests for performance regression detection."""
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_end_to_end_evaluation_benchmark(self):
        """Benchmark complete end-to-end evaluation performance."""
        # Create benchmark dataset
        benchmark_problems = []
        for i in range(50):  # Moderate size for benchmarking
            problem = {
                "id": f"benchmark_{i:03d}",
                "title": f"Benchmark Problem {i}",
                "language": "python",
                "scenario": "function_generation",
                "difficulty": "simple",
                "context_mode": "no_context",
                "prompt": f"Write a function that returns {i}",
                "reference": [f"def func_{i}(): return {i}"],
                "tests": [],
                "metadata": {
                    "time_limit_s": 5,
                    "memory_limit_mb": 50,
                    "seed": i,
                    "author": "benchmark",
                    "license": "MIT"
                }
            }
            benchmark_problems.append(problem)
        
        # Mock the complete evaluation pipeline
        with patch('utils.validate_problem_schema', return_value=True):
            with patch('datasets.Dataset.from_list') as mock_dataset:
                mock_ds = MagicMock()
                mock_ds.__len__.return_value = len(benchmark_problems)
                mock_ds.__iter__.return_value = benchmark_problems
                mock_dataset.return_value = mock_ds
                
                with patch('utils.load_context_configs', return_value={"no_context": {"template": "{{prompt}}"}}):
                    with patch('pathlib.Path.exists', return_value=True):
                        with patch('builtins.open', create=True):
                            
                            # Benchmark the complete pipeline
                            start_time = time.time()
                            
                            # Step 1: Load dataset
                            dataset = load_dataset()
                            
                            # Step 2: Process all documents
                            processed_docs = []
                            for problem in benchmark_problems:
                                processed_doc = process_docs(problem)
                                processed_docs.append(processed_doc)
                            
                            # Step 3: Calculate metrics for all
                            predictions = [f"def func_{i}(): return {i}" for i in range(50)]
                            references = [f"def func_{i}(): return {i}" for i in range(50)]
                            
                            em_score = exact_match(predictions, references)
                            
                            # Step 4: Syntax validation for all
                            syntax_scores = []
                            for pred in predictions:
                                score = syntax_validity(pred, "python")
                                syntax_scores.append(score)
                            
                            total_time = time.time() - start_time
                            
                            # Benchmark assertions
                            assert len(processed_docs) == 50
                            assert em_score == 1.0
                            assert len(syntax_scores) == 50
                            assert total_time < 10.0, f"End-to-end evaluation took {total_time:.2f}s, should be < 10s"
                            
                            # Performance metrics
                            throughput = len(benchmark_problems) / total_time
                            assert throughput > 5.0, f"Throughput was {throughput:.2f} problems/s, should be > 5/s"
    
    @pytest.mark.performance
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform repeated operations that might cause memory leaks
        for iteration in range(10):
            # Create and process documents
            problems = []
            for i in range(100):
                problem = {
                    "id": f"leak_test_{iteration}_{i}",
                    "prompt": f"Test prompt {i}" * 10,  # Some content
                    "context_mode": "no_context"
                }
                problems.append(problem)
            
            # Process all problems
            with patch('utils.load_context_configs', return_value={"no_context": {"template": "{{prompt}}"}}):
                for problem in problems:
                    processed = process_docs(problem)
            
            # Calculate metrics
            predictions = [f"result_{i}" for i in range(100)]
            references = [f"result_{i}" for i in range(100)]
            em_score = exact_match(predictions, references)
            
            # Check memory after each iteration
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory should not continuously increase
            assert memory_increase < 50 * (iteration + 1), f"Potential memory leak detected: {memory_increase:.2f}MB increase after {iteration + 1} iterations"
            
            # Clean up explicitly
            del problems, predictions, references


class TestPerformanceRegression:
    """Tests to detect performance regressions."""
    
    @pytest.mark.performance
    def test_dataset_loading_regression(self):
        """Test for dataset loading performance regression."""
        # Baseline: should load 100 problems in < 1 second
        problems = [{"id": f"reg_test_{i}", "prompt": f"test {i}"} for i in range(100)]
        
        with patch('utils.validate_problem_schema', return_value=True):
            with patch('datasets.Dataset.from_list') as mock_dataset:
                mock_ds = MagicMock()
                mock_ds.__len__.return_value = len(problems)
                mock_dataset.return_value = mock_ds
                
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('builtins.open', create=True):
                        
                        start_time = time.time()
                        dataset = load_dataset()
                        loading_time = time.time() - start_time
                        
                        # Regression test: should be fast
                        assert loading_time < 1.0, f"Dataset loading regression: {loading_time:.2f}s > 1.0s baseline"
    
    @pytest.mark.performance
    def test_metrics_calculation_regression(self):
        """Test for metrics calculation performance regression."""
        # Baseline: should calculate exact match for 200 items in < 0.1 seconds
        predictions = [f"test_{i}" for i in range(200)]
        references = [f"test_{i}" for i in range(200)]
        
        start_time = time.time()
        em_score = exact_match(predictions, references)
        calculation_time = time.time() - start_time
        
        assert em_score == 1.0
        assert calculation_time < 0.1, f"Metrics calculation regression: {calculation_time:.3f}s > 0.1s baseline"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])