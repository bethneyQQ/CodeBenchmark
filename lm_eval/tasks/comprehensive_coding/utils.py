"""
Comprehensive Coding Evaluation - Utility Functions
"""

import json
import os
import re
import ast
import subprocess
import tempfile
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import datasets
from datasets import Dataset

# Language configurations
LANGUAGE_CONFIG = {
    'python': {
        'extension': '.py',
        'comment_style': '#',
        'execution_cmd': 'python',
        'syntax_checker': 'ast.parse',
        'test_framework': 'pytest'
    },
    'javascript': {
        'extension': '.js',
        'comment_style': '//',
        'execution_cmd': 'node',
        'syntax_checker': 'node --check',
        'test_framework': 'jest'
    },
    'java': {
        'extension': '.java',
        'comment_style': '//',
        'execution_cmd': 'java',
        'compile_cmd': 'javac',
        'syntax_checker': 'javac',
        'test_framework': 'junit'
    },
    'cpp': {
        'extension': '.cpp',
        'comment_style': '//',
        'execution_cmd': './a.out',
        'compile_cmd': 'g++',
        'syntax_checker': 'g++ -fsyntax-only',
        'test_framework': 'gtest'
    },
    'go': {
        'extension': '.go',
        'comment_style': '//',
        'execution_cmd': 'go run',
        'syntax_checker': 'go fmt',
        'test_framework': 'go test'
    }
}

def load_dataset(**kwargs):
    """Load the comprehensive coding dataset."""
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(current_dir, 'problems.jsonl')
    
    # Load problems
    problems = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    
    # Filter by task type if specified in metadata
    metadata = kwargs.get('metadata', {})
    if isinstance(metadata, dict):
        task_filter = metadata.get('task_filter')
        language_filter = metadata.get('language_filter')
        complexity_filter = metadata.get('complexity_filter')
        context_filter = metadata.get('context_filter')
        
        if task_filter:
            problems = [p for p in problems if p.get('task_type') == task_filter]
        if language_filter:
            problems = [p for p in problems if p.get('language') == language_filter]
        if complexity_filter:
            problems = [p for p in problems if p.get('complexity') == complexity_filter]
        if context_filter is not None:
            problems = [p for p in problems if bool(p.get('context')) == context_filter]
    
    return {"test": Dataset.from_list(problems)}

def process_docs(dataset):
    """Process documents for evaluation."""
    return dataset

def get_context_config():
    """Get context configuration from environment variables."""
    return {
        'enable_context': os.getenv('ENABLE_CONTEXT', 'true').lower() == 'true',
        'context_mode': os.getenv('CONTEXT_MODE', 'full'),  # full, minimal, none
        'include_examples': os.getenv('INCLUDE_EXAMPLES', 'true').lower() == 'true',
        'include_constraints': os.getenv('INCLUDE_CONSTRAINTS', 'true').lower() == 'true'
    }

def format_context(doc: Dict, config: Dict) -> str:
    """Format context information based on configuration."""
    if not config.get('enable_context', True) or not doc.get('context'):
        return ""
    
    context_parts = []
    context = doc['context']
    
    # Add coding standards
    if config.get('include_constraints', True) and context.get('coding_standards'):
        context_parts.append(f"Coding Standards: {context['coding_standards']}")
    
    # Add framework requirements
    if context.get('framework'):
        context_parts.append(f"Framework: {context['framework']}")
    
    # Add performance requirements
    if context.get('performance_requirements'):
        context_parts.append(f"Performance: {context['performance_requirements']}")
    
    # Add examples in full mode
    if config.get('include_examples', True) and config.get('context_mode') == 'full':
        if context.get('examples'):
            context_parts.append(f"Examples: {context['examples']}")
    
    return " | ".join(context_parts) if context_parts else ""

def format_prompt(doc: Dict) -> str:
    """Format the main prompt for comprehensive coding evaluation."""
    config = get_context_config()
    context = format_context(doc, config)
    
    # Base prompt structure
    language = doc.get('language', 'python')
    task_type = doc.get('task_type', 'implementation')
    complexity = doc.get('complexity', 'medium')
    
    # Context section
    context_section = f"\nContext: {context}\n" if context else ""
    
    # Task-specific prompts
    if task_type == 'completion':
        prompt = f"""Complete the following {language} code:

{context_section}
```{language}
{doc['incomplete_code']}
```

Complete the missing parts to make the code functional and correct."""

    elif task_type == 'bug_fix':
        prompt = f"""Fix the bugs in the following {language} code:

{context_section}
```{language}
{doc['buggy_code']}
```

Error description: {doc['error_description']}

Provide the corrected code."""

    elif task_type == 'translation':
        source_lang = doc.get('source_language', 'python')
        prompt = f"""Translate the following {source_lang} code to {language}:

{context_section}
```{source_lang}
{doc['source_code']}
```

Provide the equivalent {language} code with the same functionality."""

    elif task_type == 'documentation':
        prompt = f"""Generate comprehensive documentation for the following {language} code:

{context_section}
```{language}
{doc['code']}
```

Provide detailed comments, docstrings, and usage examples."""

    elif task_type == 'implementation':
        prompt = f"""Implement a {language} solution for the following problem:

{context_section}
**Problem**: {doc['problem_description']}

**Requirements**:
{chr(10).join(f"- {req}" for req in doc.get('requirements', []))}

**Input/Output Examples**:
{format_examples(doc.get('examples', []))}

Provide a complete, working {language} implementation."""

    elif task_type == 'system_design':
        prompt = f"""Design and implement a complete {language} system:

{context_section}
**System Requirements**: {doc['system_description']}

**Components Needed**:
{chr(10).join(f"- {comp}" for comp in doc.get('components', []))}

**Technical Constraints**:
{chr(10).join(f"- {constraint}" for constraint in doc.get('constraints', []))}

Provide a complete system implementation with all necessary components."""

    elif task_type == 'api_development':
        prompt = f"""Develop a complete {language} API:

{context_section}
**API Specification**: {doc['api_description']}

**Endpoints Required**:
{format_api_endpoints(doc.get('endpoints', []))}

**Data Models**:
{format_data_models(doc.get('models', []))}

Provide a complete API implementation with error handling and documentation."""

    else:
        # Default implementation prompt
        prompt = f"""Solve the following {language} programming problem:

{context_section}
**Problem**: {doc['problem_description']}

Provide a complete, working solution."""

    return prompt

def format_examples(examples: List[Dict]) -> str:
    """Format input/output examples."""
    if not examples:
        return "No examples provided."
    
    formatted = []
    for i, example in enumerate(examples, 1):
        formatted.append(f"Example {i}:")
        formatted.append(f"  Input: {example.get('input', 'N/A')}")
        formatted.append(f"  Output: {example.get('output', 'N/A')}")
        if example.get('explanation'):
            formatted.append(f"  Explanation: {example['explanation']}")
    
    return "\n".join(formatted)

def format_api_endpoints(endpoints: List[Dict]) -> str:
    """Format API endpoint specifications."""
    if not endpoints:
        return "No specific endpoints required."
    
    formatted = []
    for endpoint in endpoints:
        formatted.append(f"- {endpoint.get('method', 'GET')} {endpoint.get('path', '/')}")
        if endpoint.get('description'):
            formatted.append(f"  Description: {endpoint['description']}")
        if endpoint.get('parameters'):
            formatted.append(f"  Parameters: {endpoint['parameters']}")
    
    return "\n".join(formatted)

def format_data_models(models: List[Dict]) -> str:
    """Format data model specifications."""
    if not models:
        return "No specific models required."
    
    formatted = []
    for model in models:
        formatted.append(f"- {model.get('name', 'Model')}")
        if model.get('fields'):
            for field in model['fields']:
                formatted.append(f"  - {field}")
    
    return "\n".join(formatted)

def extract_code_from_response(response: str, language: str) -> str:
    """Extract code from model response."""
    # Try to find code blocks with language specification
    pattern = f"```{language}\\s*\\n(.*?)\\n```"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[0].strip()
    
    # Try to find generic code blocks
    pattern = r"```\s*\n(.*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # If no code blocks found, return the entire response
    return response.strip()

def check_syntax(code: str, language: str) -> Tuple[bool, str]:
    """Check if code has valid syntax."""
    try:
        if language == 'python':
            ast.parse(code)
            return True, "Syntax is valid"
        
        elif language == 'javascript':
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(['node', '--check', f.name], 
                                      capture_output=True, text=True, timeout=10)
                os.unlink(f.name)
                return result.returncode == 0, result.stderr or "Syntax is valid"
        
        elif language == 'java':
            # Extract class name from code
            class_match = re.search(r'class\s+(\w+)', code)
            if not class_match:
                return False, "No class definition found"
            
            class_name = class_match.group(1)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
                f.write(code)
                f.flush()
                # Rename to match class name
                java_file = f.name.replace('.java', f'/{class_name}.java')
                os.makedirs(os.path.dirname(java_file), exist_ok=True)
                os.rename(f.name, java_file)
                
                result = subprocess.run(['javac', java_file], 
                                      capture_output=True, text=True, timeout=30)
                os.unlink(java_file)
                return result.returncode == 0, result.stderr or "Syntax is valid"
        
        elif language == 'go':
            with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(['go', 'fmt', f.name], 
                                      capture_output=True, text=True, timeout=10)
                os.unlink(f.name)
                return result.returncode == 0, result.stderr or "Syntax is valid"
        
        else:
            return True, "Syntax checking not implemented for this language"
            
    except Exception as e:
        return False, f"Syntax error: {str(e)}"

def execute_code(code: str, language: str, test_input: str = "") -> Tuple[bool, str, str]:
    """Execute code and return success status, output, and error."""
    try:
        if language == 'python':
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                
                process = subprocess.Popen(['python', f.name], 
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE,
                                         text=True)
                stdout, stderr = process.communicate(input=test_input, timeout=30)
                os.unlink(f.name)
                
                return process.returncode == 0, stdout, stderr
        
        elif language == 'javascript':
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                f.flush()
                
                process = subprocess.Popen(['node', f.name], 
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE,
                                         text=True)
                stdout, stderr = process.communicate(input=test_input, timeout=30)
                os.unlink(f.name)
                
                return process.returncode == 0, stdout, stderr
        
        # Add more language execution logic here
        else:
            return False, "", f"Execution not implemented for {language}"
            
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    except Exception as e:
        return False, "", f"Execution error: {str(e)}"

def build_predictions(responses: List[str], docs: List[Dict]) -> List[str]:
    """Build predictions from model responses."""
    predictions = []
    
    for response, doc in zip(responses, docs):
        language = doc.get('language', 'python')
        extracted_code = extract_code_from_response(response, language)
        predictions.append(extracted_code)
    
    return predictions