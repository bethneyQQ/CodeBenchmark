"""
Multi-Turn Evaluation Orchestrator

This module handles the orchestration of multi-turn evaluations,
managing the flow between phases and integrating with different model backends.
"""

import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    """Results from a single phase evaluation."""
    phase_name: str
    input_context: str
    model_response: str
    extracted_content: str
    phase_metrics: Dict[str, float]
    execution_time: float
    

@dataclass
class MultiTurnResult:
    """Complete multi-turn evaluation result."""
    problem_id: str
    phases: List[PhaseResult]
    overall_metrics: Dict[str, float]
    total_execution_time: float
    conversation_flow: List[Dict[str, Any]]


class ModelAdapter:
    """Adapter to interface with different model backends."""
    
    def __init__(self, model_name: str, model_args: Dict[str, Any]):
        self.model_name = model_name
        self.model_args = model_args
        self._setup_model()
    
    def _setup_model(self):
        """Setup the specific model backend."""
        if self.model_name == "deepseek":
            self._setup_deepseek()
        elif self.model_name == "dashscope":
            self._setup_dashscope()
        elif self.model_name == "claude-local":
            self._setup_claude()
        else:
            logger.warning(f"Unknown model backend: {self.model_name}")
    
    def _setup_deepseek(self):
        """Setup DeepSeek model."""
        from lm_eval.models.deepseek_model import DeepSeekLM
        self.model = DeepSeekLM.create_from_arg_string(
            ",".join([f"{k}={v}" for k, v in self.model_args.items()])
        )
    
    def _setup_dashscope(self):
        """Setup DashScope model.""" 
        from lm_eval.models.dashscope_model import DashScopeLM
        self.model = DashScopeLM(**self.model_args)
        
    def _setup_claude(self):
        """Setup Claude model."""
        from lm_eval.models.claude_local import ClaudeLocal
        self.model = ClaudeLocal(**self.model_args)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using the model."""
        try:
            # Create a request in the format expected by lm-eval models
            request = {
                "context": prompt,
                "gen_kwargs": kwargs
            }
            
            # Call the model's generate_until method
            responses = self.model.generate_until([request])
            
            if responses and len(responses) > 0:
                return responses[0]
            else:
                logger.warning("Model returned empty response")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating with {self.model_name}: {e}")
            return f"Error: {str(e)}"


class MultiTurnOrchestrator:
    """Orchestrates multi-turn evaluation across different phases."""
    
    def __init__(self, model_adapter: ModelAdapter):
        self.model_adapter = model_adapter
        self.conversation_history = []
    
    def evaluate_problem(self, problem_doc: Dict[str, Any]) -> MultiTurnResult:
        """Evaluate a single problem through all phases."""
        start_time = time.time()
        
        problem_id = problem_doc['problem_id']
        logger.info(f"Starting multi-turn evaluation for {problem_id}")
        
        phases = []
        conversation_flow = []
        
        # Phase 1: PRD Generation
        prd_result = self._execute_phase_1(problem_doc)
        phases.append(prd_result)
        conversation_flow.append({
            'phase': 'prd_generation',
            'input': prd_result.input_context[:500] + "...",
            'output': prd_result.extracted_content[:500] + "..."
        })
        
        # Phase 2: Technical Design (using PRD output)
        design_result = self._execute_phase_2(problem_doc, prd_result.extracted_content)
        phases.append(design_result) 
        conversation_flow.append({
            'phase': 'technical_design',
            'input': design_result.input_context[:500] + "...",
            'output': design_result.extracted_content[:500] + "..."
        })
        
        # Phase 3: Implementation (using PRD + Design outputs)
        impl_result = self._execute_phase_3(
            problem_doc, 
            prd_result.extracted_content, 
            design_result.extracted_content
        )
        phases.append(impl_result)
        conversation_flow.append({
            'phase': 'implementation', 
            'input': impl_result.input_context[:500] + "...",
            'output': impl_result.extracted_content[:500] + "..."
        })
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(phases)
        
        total_time = time.time() - start_time
        
        return MultiTurnResult(
            problem_id=problem_id,
            phases=phases,
            overall_metrics=overall_metrics,
            total_execution_time=total_time,
            conversation_flow=conversation_flow
        )
    
    def _execute_phase_1(self, problem_doc: Dict[str, Any]) -> PhaseResult:
        """Execute PRD generation phase."""
        start_time = time.time()
        
        prompt = self._build_prd_prompt(problem_doc['problem_description'])
        
        response = self.model_adapter.generate(
            prompt,
            temperature=0.0,
            max_gen_toks=2000
        )
        
        # Extract PRD content
        extracted_content = self._extract_prd_content(response)
        
        # Calculate phase-specific metrics
        from .metrics import prd_completeness_score, prd_clarity_score
        
        phase_metrics = {
            'prd_completeness': prd_completeness_score([extracted_content], [""])[0],
            'prd_clarity': prd_clarity_score([extracted_content], [""])[0]
        }
        
        execution_time = time.time() - start_time
        
        return PhaseResult(
            phase_name="prd_generation",
            input_context=prompt,
            model_response=response,
            extracted_content=extracted_content,
            phase_metrics=phase_metrics,
            execution_time=execution_time
        )
    
    def _execute_phase_2(self, problem_doc: Dict[str, Any], prd_content: str) -> PhaseResult:
        """Execute technical design phase."""
        start_time = time.time()
        
        prompt = self._build_design_prompt(
            problem_doc['problem_description'], 
            prd_content
        )
        
        response = self.model_adapter.generate(
            prompt,
            temperature=0.0,
            max_gen_toks=2000
        )
        
        # Extract design content
        extracted_content = self._extract_design_content(response)
        
        # Calculate phase-specific metrics  
        from .metrics import design_coherence_score, design_feasibility_score
        
        phase_metrics = {
            'design_coherence': design_coherence_score([extracted_content], [""])[0],
            'design_feasibility': design_feasibility_score([extracted_content], [""])[0]
        }
        
        execution_time = time.time() - start_time
        
        return PhaseResult(
            phase_name="technical_design",
            input_context=prompt,
            model_response=response, 
            extracted_content=extracted_content,
            phase_metrics=phase_metrics,
            execution_time=execution_time
        )
    
    def _execute_phase_3(self, problem_doc: Dict[str, Any], prd_content: str, design_content: str) -> PhaseResult:
        """Execute implementation phase."""
        start_time = time.time()
        
        prompt = self._build_implementation_prompt(
            problem_doc['problem_description'],
            prd_content,
            design_content
        )
        
        response = self.model_adapter.generate(
            prompt,
            temperature=0.0,
            max_gen_toks=3000
        )
        
        # Extract implementation content
        extracted_content = self._extract_implementation_content(response)
        
        # Calculate phase-specific metrics
        from .metrics import (
            code_functionality_score, 
            code_structure_score, 
            code_test_coverage_score
        )
        
        phase_metrics = {
            'code_functionality': code_functionality_score([extracted_content], [""])[0],
            'code_structure': code_structure_score([extracted_content], [""])[0], 
            'code_test_coverage': code_test_coverage_score([extracted_content], [""])[0]
        }
        
        execution_time = time.time() - start_time
        
        return PhaseResult(
            phase_name="implementation",
            input_context=prompt,
            model_response=response,
            extracted_content=extracted_content,
            phase_metrics=phase_metrics,
            execution_time=execution_time
        )
    
    def _build_prd_prompt(self, problem_description: str) -> str:
        """Build PRD generation prompt."""
        return f"""# Phase 1: Product Requirements Document (PRD) Generation

**Problem Description:**
{problem_description}

**Task:** Create a comprehensive Product Requirements Document (PRD) that includes:

1. **Problem Statement**: Clear definition of the problem to solve
2. **Objectives**: Main goals and success criteria  
3. **User Stories**: Key user scenarios and requirements
4. **Functional Requirements**: Core features and capabilities
5. **Non-functional Requirements**: Performance, security, scalability constraints
6. **Success Metrics**: Measurable criteria for evaluation

**Output Format:**
Please structure your PRD response as follows:
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

**Instructions:**
- Be specific and actionable
- Include measurable criteria where possible
- Consider scalability and maintainability
- Focus on user value and business impact

Please complete Phase 1 - PRD Generation:"""
    
    def _build_design_prompt(self, problem_description: str, prd_content: str) -> str:
        """Build technical design prompt."""
        return f"""# Phase 2: Technical Design

**Problem Description:**
{problem_description}

**PRD Output from Phase 1:**
{prd_content}

**Task:** Based on the problem description and PRD above, create a detailed technical design that includes:

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

**Instructions:**
- Build upon the PRD requirements
- Choose appropriate and proven technologies
- Consider real-world constraints and trade-offs
- Design for maintainability and extensibility

Please complete Phase 2 - Technical Design:"""
    
    def _build_implementation_prompt(self, problem_description: str, prd_content: str, design_content: str) -> str:
        """Build implementation prompt.""" 
        return f"""# Phase 3: Implementation

**Problem Description:**
{problem_description}

**PRD Output from Phase 1:**
{prd_content}

**Design Output from Phase 2:**  
{design_content}

**Task:** Based on all previous context, implement a complete Python solution that includes:

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

**Instructions:**
- Follow the technical design from Phase 2
- Implement all functional requirements from the PRD
- Include proper error handling and logging
- Write comprehensive unit tests
- Add clear documentation and examples

Please complete Phase 3 - Implementation:"""
    
    def _extract_prd_content(self, response: str) -> str:
        """Extract PRD content from model response.""" 
        import re
        match = re.search(r'PRD_START(.*?)PRD_END', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response  # Fallback to full response
    
    def _extract_design_content(self, response: str) -> str:
        """Extract design content from model response."""
        import re
        match = re.search(r'DESIGN_START(.*?)DESIGN_END', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response
    
    def _extract_implementation_content(self, response: str) -> str:
        """Extract implementation content from model response."""
        import re
        match = re.search(r'IMPLEMENTATION_START(.*?)IMPLEMENTATION_END', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response
    
    def _calculate_overall_metrics(self, phases: List[PhaseResult]) -> Dict[str, float]:
        """Calculate overall metrics across all phases."""
        from .metrics import phase_consistency_score, information_flow_score
        
        # Combine all responses for cross-phase analysis
        combined_response = "\n\n".join([
            f"{phase.phase_name.upper()}_START\n{phase.extracted_content}\n{phase.phase_name.upper()}_END"
            for phase in phases
        ])
        
        overall_metrics = {
            'phase_consistency': phase_consistency_score([combined_response], [""])[0],
            'information_flow': information_flow_score([combined_response], [""])[0]
        }
        
        # Add average metrics from all phases
        all_phase_metrics = {}
        for phase in phases:
            for metric_name, metric_value in phase.phase_metrics.items():
                if metric_name not in all_phase_metrics:
                    all_phase_metrics[metric_name] = []
                all_phase_metrics[metric_name].append(metric_value)
        
        # Calculate averages
        for metric_name, values in all_phase_metrics.items():
            overall_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
        
        return overall_metrics


def run_multi_turn_evaluation(
    problems: List[Dict[str, Any]], 
    model_name: str,
    model_args: Dict[str, Any],
    output_path: Optional[str] = None
) -> List[MultiTurnResult]:
    """Run multi-turn evaluation on a list of problems."""
    
    logger.info(f"Starting multi-turn evaluation with {model_name}")
    logger.info(f"Evaluating {len(problems)} problems")
    
    # Setup model adapter
    model_adapter = ModelAdapter(model_name, model_args)
    
    # Create orchestrator
    orchestrator = MultiTurnOrchestrator(model_adapter)
    
    results = []
    
    for i, problem in enumerate(problems):
        logger.info(f"Evaluating problem {i+1}/{len(problems)}: {problem['problem_id']}")
        
        try:
            result = orchestrator.evaluate_problem(problem)
            results.append(result)
            
            logger.info(f"Completed {problem['problem_id']} in {result.total_execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error evaluating {problem['problem_id']}: {e}")
            continue
    
    # Save results if output path provided
    if output_path:
        save_results(results, output_path)
        
    return results


def save_results(results: List[MultiTurnResult], output_path: str):
    """Save evaluation results to file."""
    
    # Convert results to serializable format
    serializable_results = []
    
    for result in results:
        result_dict = {
            'problem_id': result.problem_id,
            'total_execution_time': result.total_execution_time,
            'overall_metrics': result.overall_metrics,
            'conversation_flow': result.conversation_flow,
            'phases': []
        }
        
        for phase in result.phases:
            phase_dict = {
                'phase_name': phase.phase_name,
                'execution_time': phase.execution_time,
                'phase_metrics': phase.phase_metrics,
                'extracted_content_length': len(phase.extracted_content),
                'model_response_length': len(phase.model_response)
            }
            result_dict['phases'].append(phase_dict)
        
        serializable_results.append(result_dict)
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    from .utils import create_sample_problems
    
    # Create sample problems
    problems_path = "sample_problems.jsonl"  
    create_sample_problems(problems_path)
    
    # Load problems
    with open(problems_path, 'r') as f:
        problems = [json.loads(line) for line in f]
    
    # Run evaluation
    model_args = {"model": "deepseek-coder", "api_key": "your-key"}  
    results = run_multi_turn_evaluation(
        problems=problems,
        model_name="deepseek", 
        model_args=model_args,
        output_path="multi_turn_results.json"
    )