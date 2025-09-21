import json
import os
from typing import List, Dict, Any
from datasets import Dataset


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
    """Build predictions by extracting responses from multi-turn interactions."""
    predictions = []
    
    for i, resp_list in enumerate(resps):
        if not resp_list:
            predictions.append([""])
            continue
        
        # For multi-turn, we get responses from all phases
        # Combine them or use the last response depending on evaluation needs
        combined_response = "\n\n".join(resp_list) if len(resp_list) > 1 else resp_list[0]
        predictions.append([combined_response])
    
    return predictions


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