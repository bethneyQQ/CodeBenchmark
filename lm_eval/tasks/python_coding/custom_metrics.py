"""
Custom metrics for Python coding evaluation tasks.
"""

import ast
import re
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Union
from difflib import SequenceMatcher
import evaluate as hf_evaluate

# Load evaluation metrics
try:
    bleu_metric = hf_evaluate.load("bleu")
    rouge_metric = hf_evaluate.load("rouge")
    meteor_metric = hf_evaluate.load("meteor")
    code_eval_metric = hf_evaluate.load("code_eval")
except Exception as e:
    print(f"Warning: Could not load some metrics: {e}")


def bleu_score(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate BLEU score for code/text generation."""
    try:
        result = bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        return {"bleu": result["bleu"]}
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return {"bleu": 0.0}


def rouge_score(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate ROUGE score for docstring generation."""
    try:
        result = rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"]
        }
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def meteor_score(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate METEOR score for docstring generation."""
    try:
        result = meteor_metric.compute(
            predictions=predictions,
            references=references
        )
        return {"meteor": result["meteor"]}
    except Exception as e:
        print(f"METEOR calculation error: {e}")
        return {"meteor": 0.0}


def code_bleu(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate CodeBLEU score for code translation."""
    try:
        # Simplified CodeBLEU - use standard BLEU on code
        result = bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        return {"code_bleu": result["bleu"]}
    except Exception as e:
        print(f"CodeBLEU calculation error: {e}")
        return {"code_bleu": 0.0}


def edit_distance(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate normalized edit distance for code repair."""
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    distances = []
    for ref, pred in zip(references, predictions):
        distance = levenshtein_distance(ref, pred)
        max_len = max(len(ref), len(pred))
        normalized_distance = distance / max_len if max_len > 0 else 0
        distances.append(normalized_distance)
    
    return {"edit_distance": sum(distances) / len(distances) if distances else 1.0}


def pass_at_k(references: List[str], predictions: List[List[str]], k: List[int] = None) -> Dict[str, float]:
    """Calculate pass@k metric for code execution."""
    if k is None:
        k = [1]
    if isinstance(k, int):
        k = [k]
    
    try:
        result = code_eval_metric.compute(
            references=references,
            predictions=predictions,
            k=k
        )
        return {f"pass@{ki}": result[f"pass@{ki}"] for ki in k}
    except Exception as e:
        print(f"Pass@k calculation error: {e}")
        return {f"pass@{ki}": 0.0 for ki in k}


def execute_code_safely(code: str, test_cases: List[str]) -> bool:
    """Safely execute code with test cases."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code + '\n')
            for test_case in test_cases:
                f.write(test_case + '\n')
            temp_file = f.name
        
        # Execute the code
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Clean up
        os.unlink(temp_file)
        
        return result.returncode == 0
    except Exception:
        return False


def build_repair_predictions(resps: List[List[str]], docs: List[Dict]) -> List[List[str]]:
    """Build predictions for code repair tasks."""
    predictions = []
    for resp_list, doc in zip(resps, docs):
        pred_list = []
        for resp in resp_list:
            # Clean up the response
            cleaned_resp = resp.strip()
            if cleaned_resp.startswith('```python'):
                cleaned_resp = cleaned_resp[9:]
            if cleaned_resp.endswith('```'):
                cleaned_resp = cleaned_resp[:-3]
            pred_list.append(cleaned_resp.strip())
        predictions.append(pred_list)
    return predictions


def build_function_predictions(resps: List[List[str]], docs: List[Dict]) -> List[List[str]]:
    """Build predictions for function generation tasks."""
    predictions = []
    for resp_list, doc in zip(resps, docs):
        pred_list = []
        for resp in resp_list:
            # Extract code from response
            cleaned_resp = resp.strip()
            if cleaned_resp.startswith('```python'):
                cleaned_resp = cleaned_resp[9:]
            if cleaned_resp.endswith('```'):
                cleaned_resp = cleaned_resp[:-3]
            
            # Add test cases for execution
            test_cases = doc.get('test_cases', [])
            full_code = cleaned_resp + '\n' + '\n'.join(test_cases)
            pred_list.append(full_code)
        predictions.append(pred_list)
    return predictions


def validate_python_syntax(code: str) -> bool:
    """Validate Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def calculate_code_similarity(code1: str, code2: str) -> float:
    """Calculate similarity between two code snippets."""
    # Normalize whitespace and remove comments
    def normalize_code(code):
        lines = []
        for line in code.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        return '\n'.join(lines)
    
    norm_code1 = normalize_code(code1)
    norm_code2 = normalize_code(code2)
    
    return SequenceMatcher(None, norm_code1, norm_code2).ratio()


# Additional utility functions for specific metrics
def extract_function_signature(code: str) -> str:
    """Extract function signature from code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return f"def {node.name}({', '.join(arg.arg for arg in node.args.args)}):"
    except:
        pass
    return ""


def count_code_complexity(code: str) -> int:
    """Simple cyclomatic complexity calculation."""
    try:
        tree = ast.parse(code)
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
    except:
        return 1