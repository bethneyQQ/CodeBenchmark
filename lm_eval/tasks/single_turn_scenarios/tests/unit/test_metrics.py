"""Unit tests for metrics.py module."""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import dataclass
from typing import List, Optional

# Import the module under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metrics import (
    exact_match, bleu_score, codebleu_score, rouge_l_score, edit_distance_score,
    syntax_validity, cyclomatic_complexity, security_score, performance_score, code_style_score,
    pass_at_k, test_coverage, runtime_correctness, memory_efficiency,
    phase_coherence, design_implementation_alignment, information_flow
)

# Mock ExecutionResult for testing
@dataclass
class MockExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    wall_time: float
    peak_memory: int
    security_violations: List[str]


class TestBasicMetrics:
    """Test cases for basic text similarity metrics."""
    
    def test_exact_match_perfect(self):
        """Test exact match with identical strings."""
        predictions = ["hello world", "test string"]
        references = ["hello world", "test string"]
        result = exact_match(predictions, references)
        assert result == 1.0
    
    def test_exact_match_no_match(self):
        """Test exact match with completely different strings."""
        predictions = ["hello world", "test string"]
        references = ["goodbye world", "different string"]
        result = exact_match(predictions, references)
        assert result == 0.0
    
    def test_exact_match_partial(self):
        """Test exact match with partial matches."""
        predictions = ["hello world", "test string"]
        references = ["hello world", "different string"]
        result = exact_match(predictions, references)
        assert result == 0.5
    
    def test_exact_match_length_mismatch(self):
        """Test exact match with different length lists."""
        predictions = ["hello world"]
        references = ["hello world", "extra string"]
        result = exact_match(predictions, references)
        assert result == 0.0
    
    def test_exact_match_empty_lists(self):
        """Test exact match with empty lists."""
        predictions = []
        references = []
        result = exact_match(predictions, references)
        assert result == 0.0
    
    def test_bleu_score_basic(self):
        """Test BLEU score calculation."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat is on the mat"]
        
        with patch('metrics.sentence_bleu') as mock_bleu:
            mock_bleu.return_value = 0.8
            result = bleu_score(predictions, references)
            assert result == 0.8
    
    def test_bleu_score_nltk_not_available(self):
        """Test BLEU score when NLTK is not available."""
        predictions = ["test"]
        references = ["test"]
        
        with patch('metrics.sentence_bleu', side_effect=ImportError):
            result = bleu_score(predictions, references)
            assert result == 0.0
    
    def test_codebleu_score_basic(self):
        """Test CodeBLEU score calculation."""
        predictions = ["def add(a, b): return a + b"]
        references = ["def add(x, y): return x + y"]
        
        with patch('metrics.calc_codebleu') as mock_codebleu:
            mock_codebleu.return_value = {"codebleu": 0.85}
            result = codebleu_score(predictions, references)
            assert result == 0.85
    
    def test_codebleu_score_not_available(self):
        """Test CodeBLEU score when library is not available."""
        predictions = ["test"]
        references = ["test"]
        
        with patch('metrics.calc_codebleu', side_effect=ImportError):
            result = codebleu_score(predictions, references)
            assert result == 0.0
    
    def test_rouge_l_score_basic(self):
        """Test ROUGE-L score calculation."""
        predictions = ["the quick brown fox"]
        references = ["the fast brown fox"]
        
        with patch('metrics.rouge_scorer.RougeScorer') as mock_scorer:
            mock_instance = MagicMock()
            mock_instance.score.return_value = {"rougeL": MagicMock(fmeasure=0.75)}
            mock_scorer.return_value = mock_instance
            
            result = rouge_l_score(predictions, references)
            assert result == 0.75
    
    def test_rouge_l_score_not_available(self):
        """Test ROUGE-L score when library is not available."""
        predictions = ["test"]
        references = ["test"]
        
        with patch('metrics.rouge_scorer', None):
            result = rouge_l_score(predictions, references)
            assert result == 0.0
    
    def test_edit_distance_score_identical(self):
        """Test edit distance with identical strings."""
        predictions = ["hello"]
        references = ["hello"]
        result = edit_distance_score(predictions, references)
        assert result == 1.0
    
    def test_edit_distance_score_different(self):
        """Test edit distance with different strings."""
        predictions = ["hello"]
        references = ["world"]
        result = edit_distance_score(predictions, references)
        assert result < 1.0 and result >= 0.0


class TestCodeQualityMetrics:
    """Test cases for code quality assessment metrics."""
    
    def test_syntax_validity_python_valid(self):
        """Test syntax validity with valid Python code."""
        code = "def add(a, b):\n    return a + b"
        result = syntax_validity(code, "python")
        assert result == 1.0
    
    def test_syntax_validity_python_invalid(self):
        """Test syntax validity with invalid Python code."""
        code = "def add(a, b\n    return a + b"  # Missing closing parenthesis
        result = syntax_validity(code, "python")
        assert result == 0.0
    
    def test_syntax_validity_javascript_valid(self):
        """Test syntax validity with valid JavaScript code."""
        code = "function add(a, b) { return a + b; }"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = syntax_validity(code, "javascript")
            assert result == 1.0
    
    def test_syntax_validity_javascript_invalid(self):
        """Test syntax validity with invalid JavaScript code."""
        code = "function add(a, b { return a + b; }"  # Missing closing parenthesis
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = syntax_validity(code, "javascript")
            assert result == 0.0
    
    def test_syntax_validity_unsupported_language(self):
        """Test syntax validity with unsupported language."""
        code = "some code"
        result = syntax_validity(code, "unsupported")
        assert result == 0.0
    
    def test_cyclomatic_complexity_python(self):
        """Test cyclomatic complexity calculation for Python."""
        code = """
def simple_function(x):
    if x > 0:
        return x
    else:
        return -x
"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="2.0"
            )
            result = cyclomatic_complexity(code, "python")
            assert result == 2.0
    
    def test_cyclomatic_complexity_tool_not_available(self):
        """Test cyclomatic complexity when analysis tool is not available."""
        code = "def test(): pass"
        
        with patch('subprocess.run', side_effect=FileNotFoundError):
            result = cyclomatic_complexity(code, "python")
            assert result == 1.0  # Default fallback
    
    def test_security_score_safe_code(self):
        """Test security score with safe code."""
        code = "def add(a, b): return a + b"
        result = security_score(code, "python")
        assert result == 1.0
    
    def test_security_score_dangerous_code(self):
        """Test security score with potentially dangerous code."""
        code = "import os; os.system('rm -rf /')"
        result = security_score(code, "python")
        assert result < 1.0
    
    def test_performance_score_fast_execution(self):
        """Test performance score with fast execution."""
        execution_result = MockExecutionResult(
            stdout="", stderr="", exit_code=0,
            wall_time=0.1, peak_memory=10, security_violations=[]
        )
        result = performance_score("def test(): pass", execution_result)
        assert result > 0.5
    
    def test_performance_score_slow_execution(self):
        """Test performance score with slow execution."""
        execution_result = MockExecutionResult(
            stdout="", stderr="", exit_code=0,
            wall_time=10.0, peak_memory=1000, security_violations=[]
        )
        result = performance_score("def test(): pass", execution_result)
        assert result < 0.5
    
    def test_code_style_score_good_style(self):
        """Test code style score with well-formatted code."""
        code = """
def add_numbers(first_number, second_number):
    \"\"\"Add two numbers and return the result.\"\"\"
    return first_number + second_number
"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            result = code_style_score(code, "python")
            assert result == 1.0
    
    def test_code_style_score_poor_style(self):
        """Test code style score with poorly formatted code."""
        code = "def add(a,b):return a+b"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="E302 expected 2 blank lines")
            result = code_style_score(code, "python")
            assert result < 1.0


class TestFunctionalMetrics:
    """Test cases for functional correctness metrics."""
    
    def test_pass_at_k_all_pass(self):
        """Test pass@k when all predictions pass tests."""
        predictions = ["def add(a, b): return a + b"] * 5
        tests = [{"cmd": "python -c 'assert add(2, 3) == 5'"}]
        
        with patch('metrics.SandboxExecutor') as mock_executor:
            mock_instance = MagicMock()
            mock_instance.execute_code.return_value = MockExecutionResult(
                stdout="", stderr="", exit_code=0,
                wall_time=0.1, peak_memory=10, security_violations=[]
            )
            mock_executor.return_value = mock_instance
            
            result = pass_at_k(predictions, tests, k=1)
            assert result == 1.0
    
    def test_pass_at_k_none_pass(self):
        """Test pass@k when no predictions pass tests."""
        predictions = ["def add(a, b): return a - b"] * 5  # Wrong implementation
        tests = [{"cmd": "python -c 'assert add(2, 3) == 5'"}]
        
        with patch('metrics.SandboxExecutor') as mock_executor:
            mock_instance = MagicMock()
            mock_instance.execute_code.return_value = MockExecutionResult(
                stdout="", stderr="AssertionError", exit_code=1,
                wall_time=0.1, peak_memory=10, security_violations=[]
            )
            mock_executor.return_value = mock_instance
            
            result = pass_at_k(predictions, tests, k=1)
            assert result == 0.0
    
    def test_test_coverage_full_coverage(self):
        """Test test coverage with full coverage."""
        code = "def add(a, b): return a + b"
        tests = [{"cmd": "python -m pytest --cov=. test_add.py"}]
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="TOTAL 100%"
            )
            result = test_coverage(code, tests)
            assert result == 1.0
    
    def test_test_coverage_partial_coverage(self):
        """Test test coverage with partial coverage."""
        code = "def add(a, b): return a + b"
        tests = [{"cmd": "python -m pytest --cov=. test_add.py"}]
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="TOTAL 75%"
            )
            result = test_coverage(code, tests)
            assert result == 0.75
    
    def test_runtime_correctness_success(self):
        """Test runtime correctness with successful execution."""
        execution_result = MockExecutionResult(
            stdout="Success", stderr="", exit_code=0,
            wall_time=0.1, peak_memory=10, security_violations=[]
        )
        result = runtime_correctness(execution_result)
        assert result == 1.0
    
    def test_runtime_correctness_failure(self):
        """Test runtime correctness with failed execution."""
        execution_result = MockExecutionResult(
            stdout="", stderr="Error", exit_code=1,
            wall_time=0.1, peak_memory=10, security_violations=[]
        )
        result = runtime_correctness(execution_result)
        assert result == 0.0
    
    def test_memory_efficiency_low_usage(self):
        """Test memory efficiency with low memory usage."""
        execution_result = MockExecutionResult(
            stdout="", stderr="", exit_code=0,
            wall_time=0.1, peak_memory=10, security_violations=[]
        )
        result = memory_efficiency(execution_result)
        assert result > 0.8
    
    def test_memory_efficiency_high_usage(self):
        """Test memory efficiency with high memory usage."""
        execution_result = MockExecutionResult(
            stdout="", stderr="", exit_code=0,
            wall_time=0.1, peak_memory=1000, security_violations=[]
        )
        result = memory_efficiency(execution_result)
        assert result < 0.5


class TestConsistencyMetrics:
    """Test cases for consistency metrics for complex scenarios."""
    
    def test_phase_coherence_coherent(self):
        """Test phase coherence with coherent multi-phase output."""
        prediction = """
        Analysis: The problem requires sorting.
        Design: Use quicksort algorithm.
        Implementation: def quicksort(arr): ...
        """
        result = phase_coherence(prediction)
        assert result > 0.5
    
    def test_phase_coherence_incoherent(self):
        """Test phase coherence with incoherent output."""
        prediction = "Random text without clear phases or structure."
        result = phase_coherence(prediction)
        assert result < 0.5
    
    def test_design_implementation_alignment_aligned(self):
        """Test design-implementation alignment with aligned output."""
        prediction = """
        Design: Create a function that adds two numbers.
        Implementation: def add(a, b): return a + b
        """
        result = design_implementation_alignment(prediction)
        assert result > 0.5
    
    def test_design_implementation_alignment_misaligned(self):
        """Test design-implementation alignment with misaligned output."""
        prediction = """
        Design: Create a function that adds two numbers.
        Implementation: def multiply(a, b): return a * b
        """
        result = design_implementation_alignment(prediction)
        assert result < 0.5
    
    def test_information_flow_good_flow(self):
        """Test information flow with good logical flow."""
        prediction = """
        First, we analyze the requirements.
        Then, we design the solution.
        Finally, we implement the code.
        """
        result = information_flow(prediction)
        assert result > 0.5
    
    def test_information_flow_poor_flow(self):
        """Test information flow with poor logical flow."""
        prediction = """
        Implementation: def test(): pass
        Requirements: unclear
        Design: maybe use a loop
        """
        result = information_flow(prediction)
        assert result < 0.5


class TestMetricErrorHandling:
    """Test cases for metric error handling and edge cases."""
    
    def test_metric_with_empty_input(self):
        """Test metrics handle empty input gracefully."""
        assert exact_match([], []) == 0.0
        assert bleu_score([], []) == 0.0
        assert edit_distance_score([], []) == 0.0
    
    def test_metric_with_none_input(self):
        """Test metrics handle None input gracefully."""
        with pytest.raises((TypeError, AttributeError)):
            exact_match(None, None)
    
    def test_syntax_validity_with_empty_code(self):
        """Test syntax validity with empty code."""
        result = syntax_validity("", "python")
        assert result == 0.0
    
    def test_pass_at_k_with_invalid_k(self):
        """Test pass@k with invalid k value."""
        predictions = ["def test(): pass"]
        tests = []
        
        # k > number of predictions
        result = pass_at_k(predictions, tests, k=5)
        assert result >= 0.0 and result <= 1.0
        
        # k = 0
        result = pass_at_k(predictions, tests, k=0)
        assert result >= 0.0 and result <= 1.0


# Test fixtures
@pytest.fixture
def sample_execution_result():
    """Sample execution result for testing."""
    return MockExecutionResult(
        stdout="Test output",
        stderr="",
        exit_code=0,
        wall_time=0.5,
        peak_memory=50,
        security_violations=[]
    )


@pytest.fixture
def sample_code():
    """Sample code for testing."""
    return """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""


if __name__ == "__main__":
    pytest.main([__file__])