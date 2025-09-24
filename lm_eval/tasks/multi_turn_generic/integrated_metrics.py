"""
Integrated Multi-turn Metrics for lm-eval framework

These metrics are designed to work directly with the lm-eval evaluation pipeline.
They analyze the complete multi-turn response and extract phase-specific scores.
"""

import re
import sys
import os
from typing import List, Dict, Any

# Add the current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from metrics import (
    prd_completeness_score, prd_clarity_score,
    design_coherence_score, design_feasibility_score,
    code_functionality_score, code_structure_score, code_test_coverage_score,
    phase_consistency_score, information_flow_score
)
from utils import extract_phase_responses_single


def integrated_prd_completeness_score(predictions: List[str], references: List[str]) -> List[float]:
    """Extract PRD phase and evaluate completeness."""
    scores = []
    
    for pred in predictions:
        phases = extract_phase_responses_single(pred)
        prd_content = phases.get('prd', '')
        
        if prd_content:
            score = prd_completeness_score([prd_content], [""])[0]
        else:
            score = 0.0
            
        scores.append(score)
    
    return scores


def integrated_prd_clarity_score(predictions: List[str], references: List[str]) -> List[float]:
    """Extract PRD phase and evaluate clarity."""
    scores = []
    
    for pred in predictions:
        phases = extract_phase_responses_single(pred)
        prd_content = phases.get('prd', '')
        
        if prd_content:
            score = prd_clarity_score([prd_content], [""])[0]
        else:
            score = 0.0
            
        scores.append(score)
    
    return scores


def integrated_design_coherence_score(predictions: List[str], references: List[str]) -> List[float]:
    """Extract design phase and evaluate coherence.""" 
    scores = []
    
    for pred in predictions:
        phases = extract_phase_responses_single(pred)
        design_content = phases.get('design', '')
        
        if design_content:
            score = design_coherence_score([design_content], [""])[0]
        else:
            score = 0.0
            
        scores.append(score)
    
    return scores


def integrated_design_feasibility_score(predictions: List[str], references: List[str]) -> List[float]:
    """Extract design phase and evaluate feasibility."""
    scores = []
    
    for pred in predictions:
        phases = extract_phase_responses_single(pred)
        design_content = phases.get('design', '')
        
        if design_content:
            score = design_feasibility_score([design_content], [""])[0]
        else:
            score = 0.0
            
        scores.append(score)
    
    return scores


def integrated_code_functionality_score(predictions: List[str], references: List[str]) -> List[float]:
    """Extract implementation phase and evaluate functionality."""
    scores = []
    
    for pred in predictions:
        phases = extract_phase_responses_single(pred)
        impl_content = phases.get('implementation', '')
        
        if impl_content:
            score = code_functionality_score([impl_content], [""])[0]
        else:
            score = 0.0
            
        scores.append(score)
    
    return scores


def integrated_code_structure_score(predictions: List[str], references: List[str]) -> List[float]:
    """Extract implementation phase and evaluate structure."""
    scores = []
    
    for pred in predictions:
        phases = extract_phase_responses_single(pred)
        impl_content = phases.get('implementation', '')
        
        if impl_content:
            score = code_structure_score([impl_content], [""])[0]
        else:
            score = 0.0
            
        scores.append(score)
    
    return scores


def integrated_code_test_coverage_score(predictions: List[str], references: List[str]) -> List[float]:
    """Extract implementation phase and evaluate test coverage."""
    scores = []
    
    for pred in predictions:
        phases = extract_phase_responses_single(pred)
        impl_content = phases.get('implementation', '')
        
        if impl_content:
            score = code_test_coverage_score([impl_content], [""])[0]
        else:
            score = 0.0
            
        scores.append(score)
    
    return scores


def integrated_phase_consistency_score(predictions: List[str], references: List[str]) -> List[float]:
    """Evaluate consistency across all phases."""
    return phase_consistency_score(predictions, references)


def integrated_information_flow_score(predictions: List[str], references: List[str]) -> List[float]:
    """Evaluate information flow between phases."""
    return information_flow_score(predictions, references)


def phase_completion_score(predictions: List[str], references: List[str]) -> List[float]:
    """Score based on how many phases were completed."""
    scores = []
    
    for pred in predictions:
        phases = extract_phase_responses_single(pred)
        
        expected_phases = ['prd', 'design', 'implementation']
        completed_phases = sum(1 for phase in expected_phases if phase in phases and phases[phase].strip())
        
        completion_score = completed_phases / len(expected_phases)
        scores.append(completion_score)
    
    return scores


def overall_multi_turn_score(predictions: List[str], references: List[str]) -> List[float]:
    """Calculate overall multi-turn score combining all metrics."""
    scores = []
    
    # Get all individual metric scores
    prd_completeness = integrated_prd_completeness_score(predictions, references)
    prd_clarity = integrated_prd_clarity_score(predictions, references)
    design_coherence = integrated_design_coherence_score(predictions, references)
    design_feasibility = integrated_design_feasibility_score(predictions, references)
    code_functionality = integrated_code_functionality_score(predictions, references)
    code_structure = integrated_code_structure_score(predictions, references)
    code_tests = integrated_code_test_coverage_score(predictions, references)
    phase_consistency = integrated_phase_consistency_score(predictions, references)
    information_flow = integrated_information_flow_score(predictions, references)
    phase_completion = phase_completion_score(predictions, references)
    
    # Combine with weights
    weights = {
        'prd_completeness': 0.12,
        'prd_clarity': 0.08, 
        'design_coherence': 0.12,
        'design_feasibility': 0.08,
        'code_functionality': 0.20,  # Highest weight for working code
        'code_structure': 0.12,
        'code_tests': 0.08,
        'phase_consistency': 0.10,
        'information_flow': 0.05,
        'phase_completion': 0.05
    }
    
    for i in range(len(predictions)):
        overall_score = (
            prd_completeness[i] * weights['prd_completeness'] +
            prd_clarity[i] * weights['prd_clarity'] +
            design_coherence[i] * weights['design_coherence'] +
            design_feasibility[i] * weights['design_feasibility'] +
            code_functionality[i] * weights['code_functionality'] +
            code_structure[i] * weights['code_structure'] +
            code_tests[i] * weights['code_tests'] +
            phase_consistency[i] * weights['phase_consistency'] +
            information_flow[i] * weights['information_flow'] +
            phase_completion[i] * weights['phase_completion']
        )
        
        scores.append(overall_score)
    
    return scores
