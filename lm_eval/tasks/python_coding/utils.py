"""
Utility functions for Python coding evaluation tasks.
"""

import datasets
import evaluate as hf_evaluate
import os
import json
import logging
import re
import ast
from typing import Dict, List, Any, Optional

def load_dataset(**kwargs):
    """Load the Python coding dataset with configurable path support.
    
    This function supports custom dataset paths via metadata args and 
    filters data by task category.
    
    Returns:
        dict: Dataset with test split containing the problems
    """
    # Get metadata from kwargs 
    metadata_args = kwargs.get('metadata_args') or kwargs.get('metadata') or {}
    
    dataset_path = None
    if metadata_args and isinstance(metadata_args, dict):
        # Check for dataset_path in metadata
        if 'dataset_path' in metadata_args:
            dataset_path = metadata_args['dataset_path']
        # Check for multiple dataset paths  
        elif 'dataset_paths' in metadata_args:
            paths = metadata_args['dataset_paths']
            if isinstance(paths, list) and len(paths) > 0:
                dataset_path = paths[0]  # Use first path by default
    
    if not dataset_path:
        # Default to problems.jsonl in the task directory
        current_dir = os.path.dirname(__file__)
        dataset_path = os.path.join(current_dir, 'problems.jsonl')
    
    # Load the dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Filter data by task category if available
    # Try to determine task type from the calling context
    task_category_map = {
        'python_code_completion': 'code_completion',
        'python_code_completion_minimal_context': 'code_completion',
        'python_code_completion_no_context': 'code_completion',
        'python_code_repair': 'code_repair',
        'python_code_repair_no_context': 'code_repair',
        'python_function_generation': 'function_generation',
        'python_docstring_generation': 'docstring_generation',
        'python_code_translation': 'code_translation'
    }
    
    # Try to get task name from calling context
    import inspect
    frame = inspect.currentframe()
    task_name = None
    try:
        # Look through the call stack to find the task name
        for i in range(10):  # Look up to 10 frames
            caller_frame = frame
            for _ in range(i):
                if caller_frame.f_back:
                    caller_frame = caller_frame.f_back
                else:
                    break
            
            # Check if we can find task name in local variables
            local_vars = caller_frame.f_locals
            if 'self' in local_vars and hasattr(local_vars['self'], 'config'):
                config = local_vars['self'].config
                if hasattr(config, 'task'):
                    task_name = config.task
                    break
    except:
        pass
    finally:
        del frame
    
    # Filter by category if we found the task name
    if task_name and task_name in task_category_map:
        expected_category = task_category_map[task_name]
        filtered_data = [item for item in data if item.get('category') == expected_category]
        if filtered_data:
            data = filtered_data
    
    # Create Dataset object
    from datasets import Dataset
    dataset = Dataset.from_list(data)
    
    return {"test": dataset}

# Avoid noisy repeated NLTK downloads: check resources before attempting to download
def _ensure_nltk_resources(resources: List[str]):
    try:
        import nltk
        from nltk.data import find
    except Exception:
        # If nltk isn't installed, skip silently; callers can decide to install if needed
        return

    for res in resources:
        try:
            find(res)
        except LookupError:
            try:
                nltk.download(res.split('/')[-1], quiet=True)
            except Exception:
                # Ignore download failures; downstream code should handle missing data
                pass

# Load metrics
try:
    # Ensure common NLTK resources are present to avoid repeated download messages
    _ensure_nltk_resources(['corpora/wordnet', 'tokenizers/punkt', 'corpora/omw-1.4'])
    bleu_metric = hf_evaluate.load("bleu")
    rouge_metric = hf_evaluate.load("rouge")
    meteor_metric = hf_evaluate.load("meteor")
    code_eval_metric = hf_evaluate.load("code_eval")
except Exception as e:
    print(f"Warning: Could not load some metrics: {e}")


def extract_python_code(text: str) -> str:
    """Extract Python code from the generated text, handling various response formats."""
    if not text or not text.strip():
        return ""
    
    # First, try to find code blocks with ```python or ``` markers
    code_block_patterns = [
        r'```python\s*\n?(.*?)\n?```',  # ```python ... ```
        r'```\s*\n?(.*?)\n?```',        # ``` ... ```
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            # If multiple code blocks, prioritize ones with function definitions
            selected_code = None
            for match in matches:
                code = match.strip()
                # For the second pattern, remove 'python' from the beginning if present
                if code.startswith('python\n'):
                    code = code[7:]  # Remove 'python\n'
                
                if code and _is_valid_python_code(code):
                    # Prefer matches that contain function definitions
                    if 'def ' in code:
                        selected_code = code
                        break
                    elif selected_code is None:
                        selected_code = code
            
            if selected_code:
                return _clean_extracted_code(selected_code)
    
    # If no code blocks found, try to extract function definitions
    lines = text.split('\n')
    all_functions = []
    current_function = []
    in_function = False
    indent_level = 0
    
    for line in lines:
        stripped_line = line.strip()
        
        # Check if this line starts a function definition
        if stripped_line.startswith('def '):
            # If we were already in a function, save it
            if in_function and current_function:
                all_functions.append('\n'.join(current_function))
            # Start new function
            in_function = True
            current_function = [line]
            indent_level = len(line) - len(line.lstrip())
        elif in_function:
            # Check if we're still inside the function
            current_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 4
            
            # Stop at lines that look like test cases or examples
            if (stripped_line.startswith('assert ') or 
                stripped_line.startswith('# Test') or
                stripped_line.startswith('# test') or
                stripped_line.startswith('# Example') or
                stripped_line.startswith('print(') or
                stripped_line.startswith('Example usage:')):
                # End function here, don't include test cases
                if current_function:
                    all_functions.append('\n'.join(current_function))
                in_function = False
                current_function = []
            elif (line.strip() == '' or 
                  current_indent > indent_level or 
                  (current_indent == indent_level and not stripped_line.startswith('def '))):
                # Still part of the function
                current_function.append(line)
            else:
                # End of function - save it and stop tracking
                if current_function:
                    all_functions.append('\n'.join(current_function))
                in_function = False
                current_function = []
                
                # Check if this line starts a new function
                if stripped_line.startswith('def '):
                    in_function = True
                    current_function = [line]
                    indent_level = len(line) - len(line.lstrip())
    
    # Don't forget the last function if the text ends while in a function
    if in_function and current_function:
        all_functions.append('\n'.join(current_function))
    
    if all_functions:
        # Return the LAST function found (should be the actual solution)
        return _clean_extracted_code(all_functions[-1])
    
    # If no functions found, try to extract any Python-like code
    # Look for lines that contain Python keywords or patterns
    python_lines = []
    for line in lines:
        stripped = line.strip()
        if (stripped and 
            (stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'return ', 'import ', 'from ')) or
             '=' in stripped or
             stripped.endswith(':'))):
            python_lines.append(line)
    
    if python_lines:
        extracted = '\n'.join(python_lines)
        if _is_valid_python_code(extracted):
            return _clean_extracted_code(extracted)
    
    # Fallback: return the entire text, cleaned
    return _clean_extracted_code(text)


def _clean_extracted_code(code: str) -> str:
    """Clean up extracted code by removing explanatory text and test cases."""
    if not code:
        return ""
    
    lines = code.split('\n')
    clean_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip explanatory text and test cases
        if (stripped.startswith('# Test') or 
            stripped.startswith('# test') or
            stripped.startswith('# Example') or
            stripped.startswith('assert ') or
            stripped.startswith('print(') or
            'Example usage:' in stripped or
            'Corrections and improvements:' in stripped or
            'This implementation:' in stripped):
            continue
        
        # Skip lines that look like explanatory text
        if (stripped and not stripped.startswith('#') and 
            any(phrase in stripped.lower() for phrase in [
                'here\'s the', 'this function', 'the code', 'corrected', 
                'improved', 'implementation', 'solution', 'fixed'
            ]) and 
            not any(keyword in stripped for keyword in [
                'def ', 'class ', 'if ', 'for ', 'while ', 'return ', '='
            ])):
            continue
            
        clean_lines.append(line)
    
    return '\n'.join(clean_lines).strip()


def _is_valid_python_code(code: str) -> bool:
    """Check if the extracted code is valid Python syntax."""
    if not code or not code.strip():
        return False
    
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def get_context_config():
    """Get context configuration from environment variables or defaults."""
    config = {
        'enable_context': True,
        'context_types': None,  # None means all types, or specify list like ['style_guide', 'security_policy']
        'context_mode': 'full',  # 'full', 'minimal', 'none'
        'debug_mode': False  # Enable debug logging
    }
    
    # Override from environment variables if set
    if 'PYTHON_CODING_ENABLE_CONTEXT' in os.environ:
        config['enable_context'] = os.environ['PYTHON_CODING_ENABLE_CONTEXT'].lower() in ('true', '1', 'yes', 'on')
    
    if 'PYTHON_CODING_CONTEXT_TYPES' in os.environ:
        types_str = os.environ['PYTHON_CODING_CONTEXT_TYPES']
        if types_str.strip():
            config['context_types'] = [t.strip() for t in types_str.split(',')]
    
    if 'PYTHON_CODING_CONTEXT_MODE' in os.environ:
        mode = os.environ['PYTHON_CODING_CONTEXT_MODE'].lower()
        if mode in ['full', 'minimal', 'none']:
            config['context_mode'] = mode
    
    if 'PYTHON_CODING_DEBUG' in os.environ:
        config['debug_mode'] = os.environ['PYTHON_CODING_DEBUG'].lower() in ('true', '1', 'yes', 'on')
    
    return config


def should_use_context(doc, category=None):
    """Dynamically decide whether to use context based on configuration and document."""
    config = get_context_config()
    
    # If context is globally disabled, return False
    if not config['enable_context'] or config['context_mode'] == 'none':
        return False
    
    # If context types are specified, check if this document's context type is included
    if config['context_types']:
        doc_context_type = doc.get('context_type', 'general')
        if doc_context_type not in config['context_types']:
            return False
    
    # Check if document has context available
    if not doc.get('context'):
        return False
    
    return True


def get_dynamic_context_mode(doc, category=None):
    """Get the context mode to use for this specific document."""
    config = get_context_config()
    
    # If context is disabled, return none
    if not should_use_context(doc, category):
        return 'none'
    
    return config['context_mode']


def debug_log_problem(doc, prompt_text, category="unknown"):
    """Log problem details when debug mode is enabled."""
    config = get_context_config()
    
    if not config['debug_mode']:
        return
    
    print(f"\n{'='*80}")
    print(f"🐛 DEBUG MODE - {category.upper()} PROBLEM")
    print(f"{'='*80}")
    
    print(f"\n📋 Problem Details:")
    print(f"   Category: {doc.get('category', 'N/A')}")
    print(f"   Context Type: {doc.get('context_type', 'N/A')}")
    
    # Show problem-specific fields
    if doc.get('category') == 'code_completion':
        print(f"   Incomplete Code: {doc.get('incomplete_code', 'N/A')[:100]}...")
        print(f"   Expected: {doc.get('expected_completion', 'N/A')[:100]}...")
    elif doc.get('category') == 'code_repair':
        print(f"   Buggy Code: {doc.get('buggy_code', 'N/A')[:100]}...")
        print(f"   Error: {doc.get('error_description', 'N/A')}")
        print(f"   Fixed Code: {doc.get('fixed_code', 'N/A')[:100]}...")
    elif doc.get('category') == 'code_translation':
        print(f"   Source Language: {doc.get('source_language', 'N/A')}")
        print(f"   Source Code: {doc.get('source_code', 'N/A')[:100]}...")
        print(f"   Target Code: {doc.get('target_code', 'N/A')[:100]}...")
    elif doc.get('category') == 'docstring_generation':
        print(f"   Function Code: {doc.get('function_code', 'N/A')[:100]}...")
        print(f"   Expected Docstring: {doc.get('expected_docstring', 'N/A')[:100]}...")
    elif doc.get('category') == 'function_generation':
        print(f"   Description: {doc.get('function_description', 'N/A')[:100]}...")
        print(f"   Requirements: {doc.get('requirements', 'N/A')[:100]}...")
    
    print(f"\n🔧 Context Information:")
    context = doc.get('context', 'No context')
    print(f"   Raw Context: {context[:200]}...")
    
    formatted_context = format_context(doc)
    print(f"   Formatted Context: {formatted_context[:200]}...")
    
    print(f"\n📝 Model Input Prompt:")
    print(f"   Length: {len(prompt_text)} characters")
    print(f"   Preview: {prompt_text[:300]}...")
    
    print(f"\n{'='*80}")


def debug_log_response(response_text, expected_output=None, category="unknown"):
    """Log model response when debug mode is enabled."""
    config = get_context_config()
    
    if not config['debug_mode']:
        return
    
    print(f"\n🤖 MODEL RESPONSE - {category.upper()}")
    print(f"{'-'*60}")
    
    print(f"📤 Model Output:")
    print(f"   Length: {len(response_text)} characters")
    print(f"   Response: {response_text[:500]}...")
    
    if expected_output:
        print(f"\n🎯 Expected Output:")
        print(f"   Expected: {expected_output[:500]}...")
        
        # Simple similarity check
        if response_text.strip() == expected_output.strip():
            print(f"   ✅ Exact Match!")
        elif expected_output.lower() in response_text.lower():
            print(f"   🔍 Partial Match Found")
        else:
            print(f"   ❌ No Obvious Match")
    
    print(f"{'-'*60}\n")


def format_context(doc, context_config=None):
    """Format context based on configuration."""
    if context_config is None:
        context_config = get_context_config()
    
    # Ensure all required keys exist
    context_config.setdefault('enable_context', True)
    context_config.setdefault('context_types', None)
    context_config.setdefault('context_mode', 'full')
    
    # If context is disabled, return generic message
    if not context_config['enable_context'] or context_config['context_mode'] == 'none':
        return "No specific requirements"
    
    # Get the context from the document
    context = doc.get('context')
    context_type = doc.get('context_type', 'general')
    
    if not context:
        return "No specific requirements"
    
    # Filter by context types if specified
    if context_config['context_types'] and context_type not in context_config['context_types']:
        return "No specific requirements"
    
    # Format based on mode
    if context_config['context_mode'] == 'minimal':
        # Extract just the key requirement
        if 'must' in context.lower() or 'required' in context.lower():
            sentences = context.split('.')
            for sentence in sentences:
                if 'must' in sentence.lower() or 'required' in sentence.lower():
                    return sentence.strip() + '.'
        return context.split('.')[0] + '.'
    
    # Full context mode
    return f"[{context_type.replace('_', ' ').title()}] {context}"


def process_docs_code_completion(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter and process documents for code completion task."""
    def _process_doc(doc):
        if doc["category"] != "code_completion":
            return None
        return doc
    
    # Filter only code completion problems
    filtered = dataset.filter(lambda x: x["category"] == "code_completion")
    return filtered


def process_docs_code_repair(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter and process documents for code repair task."""
    def _process_doc(doc):
        if doc["category"] != "code_repair":
            return None
        return doc
    
    # Filter only code repair problems
    filtered = dataset.filter(lambda x: x["category"] == "code_repair")
    return filtered


def process_docs_code_translation(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter and process documents for code translation task."""
    def _process_doc(doc):
        if doc["category"] != "code_translation":
            return None
        return doc
    
    # Filter only code translation problems
    filtered = dataset.filter(lambda x: x["category"] == "code_translation")
    return filtered


def process_docs_docstring_generation(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter and process documents for docstring generation task."""
    def _process_doc(doc):
        if doc["category"] != "docstring_generation":
            return None
        return doc
    
    # Filter only docstring generation problems
    filtered = dataset.filter(lambda x: x["category"] == "docstring_generation")
    return filtered


def process_docs_function_generation(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter and process documents for function generation task."""
    def _process_doc(doc):
        if doc["category"] != "function_generation":
            return None
        return doc
    
    # Filter only function generation problems
    filtered = dataset.filter(lambda x: x["category"] == "function_generation")
    return filtered


# Code Completion functions
def doc_to_text(doc):
    """Generate input text for code completion with dynamic context."""
    context_mode = get_dynamic_context_mode(doc, 'code_completion')
    context_config = {'enable_context': context_mode != 'none', 'context_mode': context_mode}
    context = format_context(doc, context_config)
    
    if context_mode == 'none' or context == "No specific requirements":
        prompt = f"""Complete the following Python code snippet:

```python
{doc["incomplete_code"]}
```

Complete the missing parts to make the code functional:"""
    else:
        prompt = f"""Context: {context}

Complete the following Python code snippet:

```python
{doc["incomplete_code"]}
```

Complete the missing parts to make the code functional:"""
    
    debug_log_problem(doc, prompt, f"code_completion_{context_mode}")
    return prompt


def doc_to_text_no_context(doc):
    """Generate input text for code completion without context."""
    prompt = f"""Complete the following Python code snippet:

```python
{doc["incomplete_code"]}
```

Complete the missing parts to make the code functional:"""
    
    debug_log_problem(doc, prompt, "code_completion_no_context")
    return prompt


def doc_to_text_minimal_context(doc):
    """Generate input text for code completion with minimal context."""
    context = format_context(doc, {'enable_context': True, 'context_mode': 'minimal'})
    prompt = f"""Context: {context}

Complete the following Python code snippet:

```python
{doc["incomplete_code"]}
```

Complete the missing parts to make the code functional:"""
    
    debug_log_problem(doc, prompt, "code_completion_minimal_context")
    return prompt


def doc_to_target(doc):
    """Generate target text for code completion."""
    return doc["expected_completion"]


# Code Repair functions
def doc_to_text_repair(doc):
    """Generate input text for code repair with dynamic context."""
    context_mode = get_dynamic_context_mode(doc, 'code_repair')
    context_config = {'enable_context': context_mode != 'none', 'context_mode': context_mode}
    context = format_context(doc, context_config)
    
    if context_mode == 'none' or context == "No specific requirements":
        prompt = f"""Fix the following buggy Python code:

```python
{doc["buggy_code"]}
```

Error: {doc["error_description"]}

Provide the corrected code:"""
    else:
        prompt = f"""Context: {context}

Fix the following buggy Python code:

```python
{doc["buggy_code"]}
```

Error: {doc["error_description"]}

Provide the corrected code:"""
    
    debug_log_problem(doc, prompt, f"code_repair_{context_mode}")
    return prompt


def doc_to_target_repair(doc):
    """Generate target text for code repair."""
    return doc["fixed_code"]


# Code Translation functions
def doc_to_text_translation(doc):
    """Generate input text for code translation with dynamic context."""
    context_mode = get_dynamic_context_mode(doc, 'code_translation')
    context_config = {'enable_context': context_mode != 'none', 'context_mode': context_mode}
    context = format_context(doc, context_config)
    
    if context_mode == 'none' or context == "No specific requirements":
        prompt = f"""Translate the following {doc["source_language"]} code to Python:

```{doc["source_language"]}
{doc["source_code"]}
```

Provide the equivalent Python code:"""
    else:
        prompt = f"""Context: {context}

Translate the following {doc["source_language"]} code to Python:

```{doc["source_language"]}
{doc["source_code"]}
```

Provide the equivalent Python code:"""
    
    debug_log_problem(doc, prompt, f"code_translation_{context_mode}")
    return prompt


def doc_to_target_translation(doc):
    """Generate target text for code translation."""
    return doc["target_code"]


# Docstring Generation functions
def doc_to_text_docstring(doc):
    """Generate input text for docstring generation with dynamic context."""
    context_mode = get_dynamic_context_mode(doc, 'docstring_generation')
    context_config = {'enable_context': context_mode != 'none', 'context_mode': context_mode}
    context = format_context(doc, context_config)
    
    if context_mode == 'none' or context == "No specific requirements":
        prompt = f"""Generate a comprehensive docstring for the following Python function:

```python
{doc["function_code"]}
```

Provide a detailed docstring following Python conventions:"""
    else:
        prompt = f"""Context: {context}

Generate a comprehensive docstring for the following Python function:

```python
{doc["function_code"]}
```

Provide a detailed docstring following Python conventions:"""
    
    debug_log_problem(doc, prompt, f"docstring_generation_{context_mode}")
    return prompt


def doc_to_target_docstring(doc):
    """Generate target text for docstring generation."""
    return doc["expected_docstring"]


# Function Generation functions
def doc_to_text_function(doc):
    """Generate input text for function generation with dynamic context."""
    context_mode = get_dynamic_context_mode(doc, 'function_generation')
    context_config = {'enable_context': context_mode != 'none', 'context_mode': context_mode}
    context = format_context(doc, context_config)
    
    if context_mode == 'none' or context == "No specific requirements":
        prompt = f"""Create a Python function based on the following description:

{doc["function_description"]}

Requirements:
{doc["requirements"]}

Provide the complete function implementation:"""
    else:
        prompt = f"""Context: {context}

Create a Python function based on the following description:

{doc["function_description"]}

Requirements:
{doc["requirements"]}

Provide the complete function implementation:"""
    
    debug_log_problem(doc, prompt, f"function_generation_{context_mode}")
    return prompt


def doc_to_target_function(doc):
    """Generate target text for function generation."""
    # For function generation, we use test cases as the target for pass@k evaluation
    return "\n".join(doc["test_cases"])


# Custom metrics
def bleu_score(gold_and_result: List) -> Dict[str, float]:
    """Calculate BLEU score for code/text generation."""
    try:
        if len(gold_and_result) == 2:
            references, predictions = gold_and_result
        else:
            return {"bleu": 0.0}
            
        # Handle both single strings and lists of strings
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
            
        result = bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        return {"bleu": result["bleu"]}
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return {"bleu": 0.0}


def rouge_score(gold_and_result: List) -> Dict[str, float]:
    """Calculate ROUGE score for docstring generation."""
    try:
        if len(gold_and_result) == 2:
            references, predictions = gold_and_result
        else:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            
        # Handle both single strings and lists of strings
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
            
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


def meteor_score(gold_and_result: List) -> Dict[str, float]:
    """Calculate METEOR score for docstring generation."""
    try:
        if len(gold_and_result) == 2:
            references, predictions = gold_and_result
        else:
            return {"meteor": 0.0}
            
        # Handle both single strings and lists of strings
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
            
        result = meteor_metric.compute(
            predictions=predictions,
            references=references
        )
        return {"meteor": result["meteor"]}
    except Exception as e:
        print(f"METEOR calculation error: {e}")
        return {"meteor": 0.0}


def code_bleu(gold_and_result: List) -> Dict[str, float]:
    """Calculate CodeBLEU score for code translation."""
    try:
        if len(gold_and_result) == 2:
            references, predictions = gold_and_result
        else:
            return {"code_bleu": 0.0}
            
        # Handle both single strings and lists of strings
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
            
        result = bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        return {"code_bleu": result["bleu"]}
    except Exception as e:
        print(f"CodeBLEU calculation error: {e}")
        return {"code_bleu": 0.0}


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


def edit_distance(gold_and_result: List) -> Dict[str, float]:
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
    
    if len(gold_and_result) == 2:
        references, predictions = gold_and_result
    else:
        return {"edit_distance": 1.0}
        
    # Handle both single strings and lists of strings
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    
    distances = []
    for ref, pred in zip(references, predictions):
        distance = levenshtein_distance(ref, pred)
        max_len = max(len(ref), len(pred))
        normalized_distance = distance / max_len if max_len > 0 else 0
        distances.append(normalized_distance)
    
    return {"edit_distance": sum(distances) / len(distances) if distances else 1.0}


# Context-Aware Metrics
def context_adherence_score(gold_and_result: List) -> Dict[str, float]:
    """Calculate how well predictions adhere to context requirements."""
    total_score = 0.0
    scored_predictions = 0
    
    # Extract references and predictions from the input
    if len(gold_and_result) == 2:
        references, predictions = gold_and_result
    else:
        return {"context_adherence_score": 0.0}
    
    # Handle both single strings and lists of strings
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    
    for pred in predictions:
        if not pred:
            continue
            
        # For context adherence, we don't need the reference, just the prediction
        # We'll use a simple heuristic based on common patterns
        score = 0.0
        max_score = 5.0
        
        # Check for descriptive variable names (not single letters)
        if len([name for name in re.findall(r'\b[a-z_][a-z0-9_]*\b', pred) if len(name) > 2]) > 0:
            score += 1.0
        
        # Check for type hints
        if '->' in pred or ': ' in pred:
            score += 1.0
        
        # Check for docstrings
        if '"""' in pred or "'''" in pred:
            score += 1.0
        
        # Check line length (approximate)
        lines = pred.split('\n')
        if all(len(line) <= 80 for line in lines):
            score += 1.0
        
        # Check for proper function structure
        if 'def ' in pred and 'return' in pred:
            score += 1.0
        
        total_score += score / max_score
        scored_predictions += 1
    
    return {"context_adherence_score": total_score / scored_predictions if scored_predictions > 0 else 0.0}


def style_compliance_score(gold_and_result: List) -> Dict[str, float]:
    """Calculate compliance with Python style guidelines."""
    total_score = 0.0
    
    # Extract references and predictions from the input
    if len(gold_and_result) == 2:
        references, predictions = gold_and_result
    else:
        return {"style_compliance_score": 0.0}
    
    # Handle both single strings and lists of strings
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    
    for pred in predictions:
        if not pred:
            continue
            
        score = 0.0
        max_score = 6.0
        
        # Check for proper indentation (4 spaces)
        lines = pred.split('\n')
        proper_indent = True
        for line in lines:
            if line.strip() and line.startswith(' '):
                leading_spaces = len(line) - len(line.lstrip(' '))
                if leading_spaces % 4 != 0:
                    proper_indent = False
                    break
        if proper_indent:
            score += 1.0
        
        # Check for descriptive names (>2 characters)
        names = re.findall(r'\b[a-z_][a-z0-9_]*\b', pred)
        descriptive_names = [name for name in names if len(name) > 2]
        if len(descriptive_names) >= len(names) * 0.8:  # 80% of names are descriptive
            score += 1.0
        
        # Check for docstrings
        if '"""' in pred or "'''" in pred:
            score += 1.0
        
        # Check for type hints
        if '->' in pred or ': ' in pred:
            score += 1.0
        
        # Check line length
        if all(len(line) <= 80 for line in lines):
            score += 1.0
        
        # Check for proper spacing around operators
        if re.search(r'\w\s*[+\-*/=]\s*\w', pred):
            score += 1.0
        
        total_score += score / max_score
    
    return {"style_compliance_score": total_score / len(predictions) if predictions else 0.0}


def security_compliance_score(gold_and_result: List) -> Dict[str, float]:
    """Calculate compliance with security best practices."""
    total_score = 0.0
    
    # Extract references and predictions from the input
    if len(gold_and_result) == 2:
        references, predictions = gold_and_result
    else:
        return {"security_compliance_score": 0.0}
    
    # Handle both single strings and lists of strings
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    
    for pred in predictions:
        if not pred:
            continue
            
        score = 0.0
        max_score = 4.0
        
        # Check for input validation
        if any(keyword in pred.lower() for keyword in ['validate', 'check', 'verify', 'isinstance', 'assert']):
            score += 1.0
        
        # Check for error handling
        if any(keyword in pred for keyword in ['try:', 'except', 'raise', 'ValueError', 'TypeError']):
            score += 1.0
        
        # Check no hardcoded credentials
        hardcoded_secrets = re.search(r'(password|secret|key|token|api_key)\s*=\s*["\'][^"\']+["\']', pred, re.IGNORECASE)
        if not hardcoded_secrets:
            score += 1.0
        
        # Check for safe practices (no eval, exec, etc.)
        unsafe_functions = re.search(r'\b(eval|exec|compile)\s*\(', pred)
        if not unsafe_functions:
            score += 1.0
        
        total_score += score / max_score
    
    return {"security_compliance_score": total_score / len(predictions) if predictions else 0.0}


def performance_awareness_score(gold_and_result: List) -> Dict[str, float]:
    """Calculate awareness of performance best practices."""
    total_score = 0.0
    
    # Extract references and predictions from the input
    if len(gold_and_result) == 2:
        references, predictions = gold_and_result
    else:
        return {"performance_awareness_score": 0.0}
    
    # Handle both single strings and lists of strings
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    
    for pred in predictions:
        if not pred:
            continue
            
        score = 0.0
        max_score = 4.0
        
        # Check for list comprehensions instead of loops
        has_comprehension = bool(re.search(r'\[.*for.*in.*\]', pred))
        has_simple_loop = bool(re.search(r'for.*:\s*\n.*append', pred, re.MULTILINE))
        if has_comprehension or not has_simple_loop:
            score += 1.0
        
        # Check for generator expressions
        if re.search(r'\(.*for.*in.*\)', pred):
            score += 1.0
        
        # Check for avoiding nested loops
        nested_loops = len(re.findall(r'for.*:\s*\n.*for.*:', pred, re.MULTILINE))
        if nested_loops == 0:
            score += 1.0
        
        # Check for efficient operations (using built-ins)
        efficient_ops = any(func in pred for func in ['sum(', 'max(', 'min(', 'any(', 'all(', 'enumerate(', 'zip('])
        if efficient_ops:
            score += 1.0
        
        total_score += score / max_score
    
    return {"performance_awareness_score": total_score / len(predictions) if predictions else 0.0}


# Filter functions for building predictions
def build_predictions_with_debug(resps: List[List[str]], docs: List[Dict]) -> List[List[str]]:
    """Build predictions with debug logging and robust code extraction."""
    predictions = []
    for resp_list, doc in zip(resps, docs):
        pred_list = []
        for resp in resp_list:
            # Log the raw response in debug mode
            expected_output = doc.get('expected_completion') or doc.get('fixed_code') or doc.get('target_code') or doc.get('expected_docstring')
            debug_log_response(resp, expected_output, doc.get('category', 'unknown'))
            
            # Extract Python code using robust extraction
            extracted_code = extract_python_code(resp)
            
            # Log the extracted code in debug mode
            config = get_context_config()
            if config['debug_mode']:
                print(f"\n🔧 CODE EXTRACTION DEBUG:")
                print(f"   Raw response length: {len(resp)} characters")
                print(f"   Extracted code length: {len(extracted_code)} characters")
                print(f"   Raw response preview: {resp[:200]}...")
                print(f"   Extracted code:")
                print(f"   {'-'*40}")
                print(f"   {extracted_code}")
                print(f"   {'-'*40}")
            
            pred_list.append(extracted_code)
        predictions.append(pred_list)
    return predictions


# Global storage for context information (for metric calculation)
_context_storage = {}

def store_context_info(resps: List[List[str]], docs: List[Dict]) -> List[List[str]]:
    """Store context information globally for use in metrics."""
    global _context_storage
    
    # Build predictions normally
    predictions = build_predictions_with_debug(resps, docs)
    
    # Store context information indexed by prediction content
    for pred_list, doc in zip(predictions, docs):
        for pred in pred_list:
            if pred:  # Only store for non-empty predictions
                _context_storage[pred] = doc
    
    return predictions

def get_context_for_prediction(prediction: str) -> Dict:
    """Get stored context for a prediction."""
    global _context_storage
    return _context_storage.get(prediction, {})


def build_repair_predictions(resps: List[List[str]], docs: List[Dict]) -> List[List[str]]:
    """Build predictions for code repair tasks."""
    return build_predictions_with_debug(resps, docs)


def build_function_predictions(resps: List[List[str]], docs: List[Dict]) -> List[List[str]]:
    """Build predictions for function generation tasks."""
    predictions = []
    for resp_list, doc in zip(resps, docs):
        pred_list = []
        for resp in resp_list:
            # Log the raw response in debug mode
            debug_log_response(resp, None, "function_generation")
            
            # Extract code from response using robust extraction
            extracted_code = extract_python_code(resp)
            
            # Log the extracted code in debug mode
            config = get_context_config()
            if config['debug_mode']:
                print(f"\n🔧 FUNCTION GENERATION CODE EXTRACTION:")
                print(f"   Extracted function code:")
                print(f"   {'-'*40}")
                print(f"   {extracted_code}")
                print(f"   {'-'*40}")
            
            # Add test cases for execution
            test_cases = doc.get('test_cases', [])
            full_code = extracted_code + '\n' + '\n'.join(test_cases)
            pred_list.append(full_code)
        predictions.append(pred_list)
    return predictions