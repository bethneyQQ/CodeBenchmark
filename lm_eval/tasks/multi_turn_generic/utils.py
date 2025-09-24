"""
Multi-turn Generic Evaluation Utils

This module provides utilities for generic multi-turn evaluation that works
with any model backend (no file system dependencies).
"""

import json
import os
import re
from typing import List, Dict, Any, Tuple
from datasets import Dataset


def load_dataset(**kwargs):
    """Load the multi-turn generic dataset."""
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(current_dir, "problems.jsonl")
    
    # Create sample problems if file doesn't exist
    if not os.path.exists(dataset_path):
        create_sample_problems(dataset_path)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(json.loads(line.strip()))
    
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
        if difficulty_filter and item.get('complexity') != difficulty_filter:
            continue
            
        doc = {
            'problem_id': item['problem_id'],
            'complexity': item.get('difficulty', 'intermediate'),
            'domain': item.get('category', 'general'),
            'problem_description': item['description'],
            'expected_outputs': item.get('expected_components', []),
            'evaluation_criteria': item.get('evaluation_criteria', {}),
            'phase_dependencies': item.get('phase_dependencies', {}),
            'business_context': item.get('business_context', ''),
            'title': item.get('title', '')
        }
        processed_data.append(doc)
    
    dataset = Dataset.from_list(processed_data)
    return {"test": dataset}


def process_docs(dataset):
    """Process the dataset to match the expected format."""
    return dataset


def format_multi_turn_prompt(doc):
    """Format the prompt for multi-turn evaluation."""
    
    # Create a comprehensive prompt that includes all phases
    # This will be used by the lm-eval framework as a single generation task
    
    full_prompt = f"""# Multi-Turn Software Engineering Evaluation

**Problem Description:**
{doc['problem_description']}

You will complete this software engineering project in 3 sequential phases. Please complete ALL phases in a single response, with clear phase separators.

---

## Phase 1: Product Requirements Document (PRD) Generation

**Task:** Create a comprehensive Product Requirements Document (PRD) that includes:

1. **Problem Statement**: Clear definition of the problem to solve
2. **Objectives**: Main goals and success criteria  
3. **User Stories**: Key user scenarios and requirements
4. **Functional Requirements**: Core features and capabilities
5. **Non-functional Requirements**: Performance, security, scalability constraints
6. **Success Metrics**: Measurable criteria for evaluation

**Output Format:**
```
PRD_START
## Problem Statement
[Your problem statement here]

## Objectives  
[Your objectives here]

## User Stories
[Your user stories here]

## Functional Requirements
[Your functional requirements here]

## Non-functional Requirements  
[Your non-functional requirements here]

## Success Metrics
[Your success metrics here]
PRD_END
```

---

## Phase 2: Technical Design

**Task:** Based on your PRD above, create a detailed technical design that includes:

1. **System Architecture**: High-level system components and their relationships
2. **API Design**: Key endpoints, request/response formats
3. **Data Models**: Database schema and data structures  
4. **Technology Stack**: Recommended technologies and frameworks
5. **Security Considerations**: Authentication, authorization, data protection
6. **Scalability Plan**: How to handle growth and performance

**Output Format:**
```
DESIGN_START
## System Architecture
[Your architecture design here]

## API Design
[Your API specifications here]

## Data Models
[Your data models here]

## Technology Stack
[Your technology choices here]

## Security Considerations
[Your security approach here]

## Scalability Plan  
[Your scalability strategy here]
DESIGN_END
```

---

## Phase 3: Implementation

**Task:** Based on your PRD and Technical Design above, implement a complete Python solution that includes:

1. **Core Implementation**: Main classes, functions, and logic
2. **API/Interface Layer**: If applicable, API endpoints or CLI interface
3. **Data Layer**: Database models, data access patterns
4. **Configuration**: Settings, environment variables
5. **Error Handling**: Comprehensive error management
6. **Unit Tests**: Test cases for core functionality
7. **Documentation**: Code comments and usage examples

**Output Format:**
```
IMPLEMENTATION_START
## Core Implementation
```python
# Main application code here
```

## API/Interface Layer  
```python
# API or interface code here
```

## Data Layer
```python
# Data models and access code here  
```

## Configuration
```python
# Configuration code here
```

## Unit Tests
```python
# Test cases here
```

## Documentation
[Usage examples and documentation here]
IMPLEMENTATION_END
```

---

**Instructions:**
- Complete ALL three phases in sequence
- Each phase should build upon the previous ones
- Be specific, actionable, and comprehensive
- Include proper formatting with the specified markers
- Focus on practical, implementable solutions

Please complete all phases now:"""

    return full_prompt


def extract_phase_responses(resps, docs):
    """Extract responses from different phases using markers - filter function for lm-eval."""
    extracted_resps = []
    
    for resp in resps:
        response = resp if isinstance(resp, str) else str(resp)
        phases = {}
        
        # Extract PRD
        prd_match = re.search(r'PRD_START(.*?)PRD_END', response, re.DOTALL)
        if prd_match:
            phases['prd'] = prd_match.group(1).strip()
        
        # Extract Design  
        design_match = re.search(r'DESIGN_START(.*?)DESIGN_END', response, re.DOTALL)
        if design_match:
            phases['design'] = design_match.group(1).strip()
        
        # Extract Implementation
        impl_match = re.search(r'IMPLEMENTATION_START(.*?)IMPLEMENTATION_END', response, re.DOTALL)
        if impl_match:
            phases['implementation'] = impl_match.group(1).strip()
        
        # Store the extracted phases as a string for compatibility
        extracted_resps.append(str(phases))
    
    return extracted_resps


def extract_phase_responses_single(response: str) -> Dict[str, str]:
    """Extract responses from different phases using markers - single response version."""
    phases = {}
    
    # Extract PRD
    prd_match = re.search(r'PRD_START(.*?)PRD_END', response, re.DOTALL)
    if prd_match:
        phases['prd'] = prd_match.group(1).strip()
    
    # Extract Design  
    design_match = re.search(r'DESIGN_START(.*?)DESIGN_END', response, re.DOTALL)
    if design_match:
        phases['design'] = design_match.group(1).strip()
    
    # Extract Implementation
    impl_match = re.search(r'IMPLEMENTATION_START(.*?)IMPLEMENTATION_END', response, re.DOTALL)
    if impl_match:
        phases['implementation'] = impl_match.group(1).strip()
    
    return phases


class MultiTurnEvaluator:
    """Handles multi-turn evaluation with different models."""
    
    def __init__(self, model_backend, model_args: Dict[str, Any]):
        self.model_backend = model_backend
        self.model_args = model_args
        self.conversation_history = []
    
    def execute_phase(self, phase_name: str, prompt: str, context: Dict[str, Any]) -> str:
        """Execute a single phase with the model."""
        
        # Build the full context for this phase
        full_prompt = self._build_phase_context(phase_name, prompt, context)
        
        # Generate response using the model
        response = self._generate_with_model(full_prompt)
        
        # Store in conversation history
        self.conversation_history.append({
            'phase': phase_name,
            'prompt': full_prompt,
            'response': response
        })
        
        return response
    
    def _build_phase_context(self, phase_name: str, base_prompt: str, context: Dict[str, Any]) -> str:
        """Build context for a specific phase including previous outputs."""
        
        if phase_name == "prd_generation":
            # Phase 1: Only problem description
            return base_prompt
            
        elif phase_name == "technical_design":
            # Phase 2: Problem + PRD output
            prd_output = context.get('prd_output', '')
            return f"""# Phase 2: Technical Design

**Previous Context:**
- Problem Description: {context.get('problem_description', '')}

**Previous PRD Output:**
{prd_output}

**Current Task:** Based on the problem description and PRD above, create a detailed technical design that includes:

1. **System Architecture**: High-level system components and their relationships
2. **API Design**: Key endpoints, request/response formats
3. **Data Models**: Database schema and data structures  
4. **Technology Stack**: Recommended technologies and frameworks
5. **Security Considerations**: Authentication, authorization, data protection
6. **Scalability Plan**: How to handle growth and performance

**Output Format:**
```
DESIGN_START
## System Architecture
[Your architecture design here]

## API Design
[Your API specifications here]

## Data Models
[Your data models here]

## Technology Stack
[Your technology choices here]

## Security Considerations
[Your security approach here]

## Scalability Plan  
[Your scalability strategy here]
DESIGN_END
```

Please complete Phase 2 - Technical Design:"""
            
        elif phase_name == "implementation":
            # Phase 3: Problem + PRD + Design outputs
            prd_output = context.get('prd_output', '')
            design_output = context.get('design_output', '')
            
            return f"""# Phase 3: Implementation

**Previous Context:**
- Problem Description: {context.get('problem_description', '')}

**Previous PRD Output:**
{prd_output}

**Previous Design Output:**  
{design_output}

**Current Task:** Based on all previous context, implement a complete Python solution that includes:

1. **Core Implementation**: Main classes, functions, and logic
2. **API/Interface Layer**: If applicable, API endpoints or CLI interface
3. **Data Layer**: Database models, data access patterns
4. **Configuration**: Settings, environment variables
5. **Error Handling**: Comprehensive error management
6. **Unit Tests**: Test cases for core functionality
7. **Documentation**: Code comments and usage examples

**Output Format:**
```
IMPLEMENTATION_START
## Core Implementation
```python
# Main application code here
```

## API/Interface Layer  
```python
# API or interface code here
```

## Data Layer
```python
# Data models and access code here  
```

## Configuration
```python
# Configuration code here
```

## Unit Tests
```python
# Test cases here
```

## Documentation
[Usage examples and documentation here]
IMPLEMENTATION_END
```

Please complete Phase 3 - Implementation:"""
        
        return base_prompt
    
    def _generate_with_model(self, prompt: str) -> str:
        """Generate response using the configured model backend."""
        # This will be implemented by the actual model backend
        # For now, return a placeholder
        return f"[Generated response for: {prompt[:100]}...]"


def run_multi_turn_evaluation(doc: Dict[str, Any], model_backend, model_args: Dict[str, Any]) -> Dict[str, Any]:
    """Run complete multi-turn evaluation for a single problem."""
    
    evaluator = MultiTurnEvaluator(model_backend, model_args)
    results = {
        'problem_id': doc['problem_id'],
        'phases': {},
        'metrics': {}
    }
    
    # Phase 1: PRD Generation
    phase1_prompt = format_multi_turn_prompt(doc)
    prd_response = evaluator.execute_phase('prd_generation', phase1_prompt, {
        'problem_description': doc['problem_description']
    })
    results['phases']['prd'] = prd_response
    
    # Extract PRD content
    phase_responses = extract_phase_responses(prd_response)
    prd_content = phase_responses.get('prd', prd_response)
    
    # Phase 2: Technical Design  
    design_response = evaluator.execute_phase('technical_design', '', {
        'problem_description': doc['problem_description'],
        'prd_output': prd_content
    })
    results['phases']['design'] = design_response
    
    # Extract design content
    design_responses = extract_phase_responses(design_response)
    design_content = design_responses.get('design', design_response)
    
    # Phase 3: Implementation
    impl_response = evaluator.execute_phase('implementation', '', {
        'problem_description': doc['problem_description'], 
        'prd_output': prd_content,
        'design_output': design_content
    })
    results['phases']['implementation'] = impl_response
    
    return results


def create_sample_problems(output_path: str):
    """Create sample problems for testing."""
    sample_problems = [
        {
            "problem_id": "generic_001",
            "complexity": "simple", 
            "domain": "web_application",
            "problem_description": "Create a simple task management web application where users can add, edit, delete, and mark tasks as complete. The application should have a clean user interface and store data persistently.",
            "expected_outputs": {
                "prd": "Should include user stories for CRUD operations on tasks",
                "design": "Should specify REST API design and database schema", 
                "implementation": "Should include working Python code with tests"
            },
            "evaluation_criteria": {
                "prd_completeness": "All required sections present",
                "design_coherence": "Logical system architecture",
                "code_functionality": "Working implementation"
            }
        },
        {
            "problem_id": "generic_002",
            "complexity": "medium",
            "domain": "data_processing", 
            "problem_description": "Design and implement a data processing pipeline that reads CSV files, validates data quality, transforms the data according to business rules, and outputs clean data in multiple formats (JSON, Parquet). Include monitoring and error handling.",
            "expected_outputs": {
                "prd": "Should define data quality criteria and transformation rules",
                "design": "Should specify pipeline architecture and monitoring approach",
                "implementation": "Should include modular Python pipeline with error handling"
            }
        },
        {
            "problem_id": "generic_003", 
            "complexity": "complex",
            "domain": "microservices",
            "problem_description": "Create a microservices-based e-commerce system with user authentication, product catalog, shopping cart, and order processing. Include API gateway, service discovery, and distributed logging. Design for high availability and scalability.",
            "expected_outputs": {
                "prd": "Should include detailed user journeys and system requirements",
                "design": "Should specify microservices architecture, communication patterns, and infrastructure",
                "implementation": "Should include multiple service implementations with proper interfaces"
            }
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for problem in sample_problems:
            f.write(json.dumps(problem) + '\n')