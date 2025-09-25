import json
import os
from typing import List, Dict, Any
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)


def load_dataset(**kwargs):
    """Load the multi-turn coding dataset."""
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(current_dir, "problems.jsonl")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Check for difficulty filter in metadata
    difficulty_filter = None
    if 'metadata' in kwargs:
        metadata = kwargs['metadata']
        if isinstance(metadata, dict) and 'difficulty_filter' in metadata:
            difficulty_filter = metadata['difficulty_filter']
    
    # Process data for lm-eval format
    processed_data = []
    for item in data:
        # Apply difficulty filter if specified
        if difficulty_filter and item['complexity'] != difficulty_filter:
            continue
            
        doc = {
            'problem_id': item['problem_id'],
            'complexity': item['complexity'],
            'domain': item['domain'],
            'problem_description': item['problem_description'],
            'prd_context': item.get('prd_context'),
            'design_context': item.get('design_context'),
            'code_context': item.get('code_context'),
            'quality_context': item.get('quality_context')
        }
        processed_data.append(doc)
    
    dataset = Dataset.from_list(processed_data)
    return {"test": dataset}


def process_docs(dataset):
    """Process the dataset to match the expected format."""
    return dataset


def get_context_config():
    """Get context configuration from environment or defaults."""
    import os
    
    # Default configuration
    config = {
        'enable_prd_context': True,
        'enable_design_context': True,
        'enable_code_context': True,
        'enable_quality_context': True
    }
    
    # Override from environment variables if set
    for key in config.keys():
        env_var = key.upper()
        if env_var in os.environ:
            config[key] = os.environ[env_var].lower() in ('true', '1', 'yes', 'on')
    
    return config


def format_prd_prompt(doc):
    """Format PRD prompt with context configuration."""
    config = get_context_config()
    
    if config['enable_prd_context'] and doc.get('prd_context'):
        context_line = f"Context: {doc['prd_context']}"
    else:
        context_line = "Context: No specific requirements"
    
    return f"""{context_line}

Create a Product Requirements Document for: {doc['problem_description']}

Requirements:
1. Write concise and precise PRD (max 500 words) covering problem statement, user stories, acceptance criteria
2. Save the document to: ./output/{doc['problem_id']}/prd.md
3. Confirm file creation with "PRD saved to [filepath]"

CONSTRAINT: Maximum 2 iterations - focus on completion over perfection.

Please provide your PRD:"""


def format_design_prompt(doc):
    """Format design prompt with context configuration."""
    config = get_context_config()
    
    if config['enable_design_context'] and doc.get('design_context'):
        context_line = f"Context: {doc['design_context']}"
    else:
        context_line = "Context: No specific requirements"
    
    return f"""{context_line}

Based on the PRD at ./output/{doc['problem_id']}/prd.md, create technical design:

Requirements:
1. Concise and precise design (max 500 words) covering system architecture, API specs, data models
2. Save design document to: ./output/{doc['problem_id']}/design.md
3. Confirm file creation with "Design saved to [filepath]"

CONSTRAINT: Maximum 2 iterations - focus on completion over perfection.

Please provide your technical design:"""


def format_implementation_prompt(doc):
    """Format implementation prompt with context configuration."""
    config = get_context_config()
    
    if config['enable_code_context'] and doc.get('code_context'):
        context_line = f"Context: {doc['code_context']}"
    else:
        context_line = "Context: No specific requirements"
    
    return f"""{context_line}

Implement the solution based on ./output/{doc['problem_id']}/design.md:

Requirements:
1. Create complete Python project in ./output/{doc['problem_id']}/src/
2. Include: main modules, classes, tests
3. Ensure code runs without errors
4. Confirm with "Implementation complete in [dirpath]"

CONSTRAINTS:
- Maximum 2 iterations for implementation
- Run tests maximum 2 times only
- Partial test pass is acceptable (no need for 100% pass rate)
- Focus on core functionality over comprehensive testing

Please implement the solution:"""


def format_quality_prompt(doc):
    """Format quality metrics prompt with context configuration."""
    config = get_context_config()
    
    if config['enable_quality_context'] and doc.get('quality_context'):
        context_line = f"Context: {doc['quality_context']}"
    else:
        context_line = "Context: No specific requirements"
    
    return f"""{context_line}

Define measurable criteria for evaluating the PRD, design, and code quality:

Requirements:
1. Define specific metrics for each phase (PRD, Design, Code)
2. Include acceptance criteria and thresholds
3. Save metrics definition to: ./output/{doc['problem_id']}/quality_metrics.md
4. Confirm with "Quality metrics saved to [filepath]"

Please define quality metrics:"""


def format_combined_prompt(doc):
    """Format combined prompt for all phases in sequence."""
    config = get_context_config()
    
    # Build context sections
    contexts = []
    
    if config['enable_prd_context'] and doc.get('prd_context'):
        contexts.append(f"PRD Context: {doc['prd_context']}")
    
    if config['enable_design_context'] and doc.get('design_context'):
        contexts.append(f"Design Context: {doc['design_context']}")
    
    if config['enable_code_context'] and doc.get('code_context'):
        contexts.append(f"Code Context: {doc['code_context']}")
    
    if config['enable_quality_context'] and doc.get('quality_context'):
        contexts.append(f"Quality Context: {doc['quality_context']}")
    
    context_section = "\n".join(contexts) if contexts else "No specific context requirements"
    
    return f"""Multi-Turn Software Engineering Task

{context_section}

Problem: {doc['problem_description']}

You will complete this software engineering project in 3 phases. For each phase, create the required deliverables and save them to the specified locations.

IMPORTANT CONSTRAINTS:
- Maximum 2 iterations per phase (do not over-iterate)
- For testing: Run tests maximum 2 times, partial pass is acceptable
- Focus on functional implementation over perfect test coverage
- Prioritize completion over perfection

PHASE 1: Product Requirements Document (PRD)
Create a concise and precise PRD (max 500 words) covering:
- Problem statement and objectives
- User stories and acceptance criteria
- Functional and non-functional requirements
- Success metrics

Save to: ./output/{doc['problem_id']}/prd.md
Confirm with: "PRD saved to [filepath]"
Iteration limit: Maximum 2 attempts

PHASE 2: Technical Design
Based on your PRD, create a concise and precise technical design (max 500 words) covering:
- System architecture and components
- API specifications and data models
- Technology stack and infrastructure
- Security and scalability considerations

Save to: ./output/{doc['problem_id']}/design.md
Confirm with: "Design saved to [filepath]"
Iteration limit: Maximum 2 attempts

PHASE 3: Implementation
Implement the complete solution as a Python project:
- Create project structure in ./output/{doc['problem_id']}/src/
- Include main modules, classes, and functions
- Add comprehensive tests

Testing guidelines:
- Run tests maximum 2 times
- Partial test pass is acceptable (no need for 100% pass rate)
- Focus on core functionality over comprehensive testing

Confirm with: "Implementation complete in [dirpath]"
Iteration limit: Maximum 2 attempts

Please complete all 3 phases in sequence with the specified constraints:"""


def build_predictions(resps: List[List[str]], docs: List[Dict]) -> List[List[str]]:
    """Build predictions by extracting responses from multi-turn interactions and saving to files."""
    predictions = []
    
    # Clean processing without excessive debug output
    for i, resp_list in enumerate(resps):
        if not resp_list:
            predictions.append([""])
            continue
        
        # Get the document for this response
        if i < len(docs):
            doc = docs[i]
            problem_id = doc.get('problem_id', f'problem_{i}')
            
            # For multi-turn, we get responses from all phases
            combined_response = "\n\n".join(resp_list) if len(resp_list) > 1 else resp_list[0]
            
            # Parse and save the content from the response
            _parse_and_save_response(combined_response, problem_id)
            
            predictions.append([combined_response])
        else:
            combined_response = "\n\n".join(resp_list) if len(resp_list) > 1 else resp_list[0]
            predictions.append([combined_response])
    
    return predictions


def _parse_and_save_response(response: str, problem_id: str):
    """Parse the model response and save content to appropriate files."""
    import re
    import os
    
    # Create output directory
    output_dir = f"./lm_eval/tasks/multi_turn_coding/output/{problem_id}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/src", exist_ok=True)
    
    # Handle Claude Code SystemMessage format
    actual_response = response
    if response.startswith("SystemMessage") or "API Error" in response:
        # If it's an error, create basic placeholder content
        _create_placeholder_content(output_dir, problem_id)
        return
    
    # Save the full response for debugging
    with open(f"{output_dir}/full_response.txt", 'w', encoding='utf-8') as f:
        f.write(actual_response)
    
    try:
        # Try multiple patterns for PRD content extraction
        prd_patterns = [
            r'PRD \(max 500 words\):(.*?)(?=Technical Design|Phase 2|Phase 3|$)',
            r'Phase 1:.*?PRD.*?:(.*?)(?=Phase 2|Technical Design|$)', 
            r'Product Requirements Document.*?:(.*?)(?=Technical Design|Phase 2|$)',
            r'# PRD(.*?)(?=# Technical Design|Phase 2|$)'
        ]
        
        prd_content = None
        for pattern in prd_patterns:
            prd_match = re.search(pattern, actual_response, re.DOTALL | re.IGNORECASE)
            if prd_match:
                prd_content = prd_match.group(1).strip()
                break
                
        if prd_content:
            prd_file = f"{output_dir}/prd.md"
            with open(prd_file, 'w', encoding='utf-8') as f:
                f.write(f"# Product Requirements Document\n\n{prd_content}")
        
        # Try multiple patterns for Technical Design content
        design_patterns = [
            r'Technical Design \(max 500 words\):(.*?)(?=Phase 3|Implementation|$)',
            r'Phase 2:.*?Technical Design.*?:(.*?)(?=Phase 3|Implementation|$)',
            r'# Technical Design(.*?)(?=Phase 3|Implementation|$)'
        ]
        
        design_content = None
        for pattern in design_patterns:
            design_match = re.search(pattern, actual_response, re.DOTALL | re.IGNORECASE)
            if design_match:
                design_content = design_match.group(1).strip()
                break
                
        if design_content:
            design_file = f"{output_dir}/design.md"
            with open(design_file, 'w', encoding='utf-8') as f:
                f.write(f"# Technical Design\n\n{design_content}")
        
        # Try multiple patterns for Implementation content  
        impl_patterns = [
            r'Project structure in.*?:(.*?)(?=Confirmation:|Testing guidelines:|$)',
            r'Phase 3:.*?Implementation(.*?)(?=Confirmation:|Testing guidelines:|$)',
            r'# Implementation(.*?)(?=Confirmation:|Testing guidelines:|$)'
        ]
        
        impl_content = None
        for pattern in impl_patterns:
            impl_match = re.search(pattern, actual_response, re.DOTALL | re.IGNORECASE) 
            if impl_match:
                impl_content = impl_match.group(1).strip()
                break
                
        if impl_content:
            # Create basic implementation files based on description
            _create_basic_implementation_files(f"{output_dir}/src", impl_content, problem_id)
            
    except Exception as e:
        print(f"Error parsing response for {problem_id}: {e}")


def _create_placeholder_content(output_dir: str, problem_id: str):
    """Create placeholder content when model fails to respond properly."""
    
    # Create placeholder PRD
    with open(f"{output_dir}/prd.md", 'w', encoding='utf-8') as f:
        f.write(f"""# Product Requirements Document - {problem_id}

## Problem Statement
This is a placeholder PRD created due to model response error.

## Objectives  
- Basic functionality placeholder
- Error handling demonstration

## Requirements
- Functional: Basic operation
- Non-functional: Error tolerance

## Success Metrics
- Placeholder metric: Response generation successful
""")
    
    # Create placeholder design
    with open(f"{output_dir}/design.md", 'w', encoding='utf-8') as f:
        f.write(f"""# Technical Design - {problem_id}

## System Architecture
This is a placeholder design created due to model response error.

## Components
- Placeholder component structure
- Basic error handling

## Technology Stack
- Language: Python
- Framework: Basic implementation

## Considerations
- Error recovery mechanisms
- Placeholder functionality
""")
    
    # Create basic implementation
    with open(f"{output_dir}/src/main.py", 'w', encoding='utf-8') as f:
        f.write(f'''#!/usr/bin/env python3
"""
Placeholder implementation for {problem_id}
Created due to model response error.
"""

def main():
    """Main placeholder function."""
    print("Hello from {problem_id}")
    print("This is a placeholder implementation.")
    
if __name__ == "__main__":
    main()
''')


def _create_basic_implementation_files(src_dir: str, impl_content: str, problem_id: str):
    """Create basic implementation files based on the implementation description."""
    import re
    
    # Create a basic README
    with open(f"{src_dir}/README.md", 'w', encoding='utf-8') as f:
        f.write(f"# Implementation for {problem_id}\n\n{impl_content}")
    
    # Create a basic main.py based on the problem
    if 'web' in problem_id or 'html' in impl_content.lower():
        # Web application - create HTML/CSS files
        with open(f"{src_dir}/index.html", 'w', encoding='utf-8') as f:
            f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hello World</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <h1>Hello World</h1>
    </header>
    <main>
        <p>Welcome! Today's date is: <span id="current-date"></span></p>
    </main>
    <script src="script.js"></script>
</body>
</html>''')
        
        with open(f"{src_dir}/styles.css", 'w', encoding='utf-8') as f:
            f.write('''body { font-family: system-ui; margin: 0; padding: 20px; }
header { text-align: center; margin-bottom: 20px; }
main { max-width: 800px; margin: 0 auto; }''')
            
        with open(f"{src_dir}/script.js", 'w', encoding='utf-8') as f:
            f.write('''document.addEventListener('DOMContentLoaded', function() {
    const dateElement = document.getElementById('current-date');
    if (dateElement) {
        dateElement.textContent = new Date().toLocaleDateString();
    }
});''')
    else:
        # Python application
        with open(f"{src_dir}/main.py", 'w', encoding='utf-8') as f:
            f.write(f'''#!/usr/bin/env python3
"""
Main implementation for {problem_id}
"""
import datetime

def main():
    """Main application entry point."""
    print("Hello World!")
    print(f"Current date: {{datetime.date.today()}}")

if __name__ == "__main__":
    main()
''')
        
        # Create a basic test file
        with open(f"{src_dir}/test_main.py", 'w', encoding='utf-8') as f:
            f.write('''#!/usr/bin/env python3
"""
Basic tests for main.py
"""
import unittest
from main import main

class TestMain(unittest.TestCase):
    def test_main_runs(self):
        """Test that main function runs without error."""
        try:
            main()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"main() raised {e}")

if __name__ == "__main__":
    unittest.main()
''')


def extract_file_paths_from_response(response: str) -> Dict[str, str]:
    """Extract file paths mentioned in the response."""
    import re
    
    file_paths = {}
    
    # Look for common patterns like "saved to", "created at", etc.
    patterns = [
        r"saved to[:\s]+([^\s\n]+)",
        r"created at[:\s]+([^\s\n]+)",
        r"written to[:\s]+([^\s\n]+)",
        r"output[:\s]+([^\s\n]+)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            if match.endswith('.md'):
                if 'prd' in match.lower():
                    file_paths['prd'] = match
                elif 'design' in match.lower():
                    file_paths['design'] = match
                elif 'quality' in match.lower() or 'metrics' in match.lower():
                    file_paths['quality_metrics'] = match
            elif '/src/' in match or match.endswith('/src'):
                file_paths['implementation'] = match
    
    return file_paths


def setup_output_directory(problem_id: str) -> str:
    """Create output directory structure for a problem."""
    output_dir = f"./output/{problem_id}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/src", exist_ok=True)
    return output_dir


def cleanup_output_directory(problem_id: str = None):
    """Clean up output directories."""
    import shutil
    
    if problem_id:
        output_dir = f"./output/{problem_id}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    else:
        # Clean up all output directories
        if os.path.exists("./output"):
            shutil.rmtree("./output")
        os.makedirs("./output", exist_ok=True)


def get_model_info_from_args() -> Dict[str, Any]:
    """Extract model information from command line arguments or environment."""
    import sys
    
    model_name = "claude-code-local"  # default
    model_args = {"model": "claude-3-haiku-20240307", "multi_turn": True}
    
    # Try to extract from command line arguments
    args = sys.argv
    for i, arg in enumerate(args):
        if arg == "--model" and i + 1 < len(args):
            model_name = args[i + 1]
        elif arg == "--model_args" and i + 1 < len(args):
            # Parse model arguments
            from .model_adapter import parse_model_args
            model_args.update(parse_model_args(args[i + 1]))
    
    return {"model_name": model_name, "model_args": model_args}


def create_universal_model_interface():
    """Create a universal model interface for use in evaluations."""
    try:
        from .model_adapter import create_model_adapter
        
        model_info = get_model_info_from_args()
        adapter = create_model_adapter(
            model_info["model_name"], 
            model_info["model_args"]
        )
        
        logger.info(f"✅ Created universal model adapter: {adapter.get_model_info()}")
        return adapter
        
    except Exception as e:
        logger.error(f"❌ Failed to create universal model adapter: {e}")
        logger.info("Falling back to Claude Code default")
        
        # Fallback to Claude Code as default
        from .model_adapter import create_model_adapter
        return create_model_adapter(
            "claude-code-local", 
            {"model": "claude-3-haiku-20240307", "multi_turn": True}
        )