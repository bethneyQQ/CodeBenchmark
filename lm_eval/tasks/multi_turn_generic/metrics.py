"""
Multi-turn Generic Evaluation Metrics

This module provides evaluation metrics for each phase of the multi-turn
software engineering workflow.
"""

import re
import json
from typing import List, Dict, Any, Union
import ast


def prd_completeness_score(predictions: List[str], references: List[str]) -> List[float]:
    """Evaluate PRD completeness based on required sections."""
    scores = []
    
    required_sections = [
        'problem statement',
        'objectives', 
        'user stories',
        'functional requirements',
        'non-functional requirements',
        'success metrics'
    ]
    
    for pred in predictions:
        if not pred:
            scores.append(0.0)
            continue
            
        pred_lower = pred.lower()
        sections_found = 0
        
        for section in required_sections:
            # Check for section headers or content
            if (section in pred_lower or 
                section.replace(' ', '_') in pred_lower or
                section.replace(' ', '') in pred_lower):
                sections_found += 1
        
        completeness = sections_found / len(required_sections)
        scores.append(completeness)
    
    return scores


def prd_clarity_score(predictions: List[str], references: List[str]) -> List[float]:
    """Evaluate PRD clarity and specificity."""
    scores = []
    
    clarity_indicators = [
        # Positive indicators
        ('specific', 1.0),
        ('measurable', 1.0),
        ('actionable', 1.0),
        ('criteria', 0.8),
        ('requirement', 0.6),
        ('should', 0.4),
        ('must', 0.6),
        ('will', 0.5),
        
        # Negative indicators  
        ('maybe', -0.3),
        ('probably', -0.3),
        ('might', -0.3),
        ('unclear', -0.5),
        ('tbd', -0.8),
        ('todo', -0.6)
    ]
    
    for pred in predictions:
        if not pred:
            scores.append(0.0)
            continue
            
        pred_lower = pred.lower()
        clarity_score = 0.5  # Base score
        word_count = len(pred.split())
        
        for indicator, weight in clarity_indicators:
            count = pred_lower.count(indicator)
            clarity_score += (count / word_count) * weight * 10  # Scale factor
        
        # Normalize to 0-1 range
        clarity_score = max(0.0, min(1.0, clarity_score))
        scores.append(clarity_score)
    
    return scores


def design_coherence_score(predictions: List[str], references: List[str]) -> List[float]:
    """Evaluate technical design coherence and completeness."""
    scores = []
    
    design_components = [
        'architecture',
        'api', 
        'database',
        'data model',
        'technology',
        'security',
        'scalability',
        'component',
        'service',
        'endpoint'
    ]
    
    for pred in predictions:
        if not pred:
            scores.append(0.0)
            continue
            
        pred_lower = pred.lower()
        components_found = 0
        
        for component in design_components:
            if component in pred_lower:
                components_found += 1
        
        # Bonus for technical depth
        technical_terms = [
            'rest', 'api', 'http', 'json', 'database', 'sql',
            'authentication', 'authorization', 'middleware',
            'microservice', 'container', 'docker', 'kubernetes'
        ]
        
        technical_depth = sum(1 for term in technical_terms if term in pred_lower)
        technical_bonus = min(0.3, technical_depth * 0.05)
        
        base_score = components_found / len(design_components)
        coherence = min(1.0, base_score + technical_bonus)
        scores.append(coherence)
    
    return scores


def design_feasibility_score(predictions: List[str], references: List[str]) -> List[float]:
    """Evaluate design feasibility and practicality."""
    scores = []
    
    feasibility_indicators = [
        # Positive indicators (realistic approach)
        ('proven', 0.8),
        ('standard', 0.7),
        ('well-established', 0.8),
        ('mature', 0.6),
        ('widely-used', 0.7),
        ('simple', 0.5),
        ('straightforward', 0.6),
        
        # Negative indicators (overengineering)
        ('cutting-edge', -0.2),
        ('experimental', -0.4), 
        ('bleeding-edge', -0.5),
        ('revolutionary', -0.3),
        ('complex', -0.2),
        ('sophisticated', -0.1)
    ]
    
    for pred in predictions:
        if not pred:
            scores.append(0.0)
            continue
            
        pred_lower = pred.lower()
        feasibility_score = 0.6  # Base score
        word_count = len(pred.split())
        
        for indicator, weight in feasibility_indicators:
            count = pred_lower.count(indicator)
            feasibility_score += (count / word_count) * weight * 10
        
        # Check for balanced complexity
        if 'scalable' in pred_lower and 'maintainable' in pred_lower:
            feasibility_score += 0.1
        
        feasibility_score = max(0.0, min(1.0, feasibility_score))
        scores.append(feasibility_score)
    
    return scores


def code_functionality_score(predictions: List[str], references: List[str]) -> List[float]:
    """Evaluate code functionality and completeness."""
    scores = []
    
    for pred in predictions:
        if not pred:
            scores.append(0.0)
            continue
            
        score = 0.0
        
        # Check for Python code presence
        python_code_blocks = re.findall(r'```python(.*?)```', pred, re.DOTALL)
        if python_code_blocks:
            score += 0.3
            
            # Analyze code content
            all_code = '\n'.join(python_code_blocks)
            
            # Check for key programming constructs
            constructs = [
                ('def ', 0.2),      # Functions
                ('class ', 0.2),    # Classes  
                ('import ', 0.1),   # Imports
                ('if ', 0.1),       # Conditionals
                ('for ', 0.1),      # Loops
                ('try:', 0.1),      # Error handling
                ('return', 0.1),    # Return statements
                ('__init__', 0.1)   # Constructors
            ]
            
            for construct, weight in constructs:
                if construct in all_code:
                    score += weight
                    
            # Check for test code
            if 'test_' in all_code or 'assert' in all_code:
                score += 0.2
                
            # Syntax validation bonus
            try:
                ast.parse(all_code)
                score += 0.1  # Bonus for syntactically correct code
            except:
                score -= 0.1  # Penalty for syntax errors
        
        scores.append(min(1.0, score))
    
    return scores


def code_structure_score(predictions: List[str], references: List[str]) -> List[float]:
    """Evaluate code structure and organization."""
    scores = []
    
    for pred in predictions:
        if not pred:
            scores.append(0.0)
            continue
            
        score = 0.0
        
        # Extract Python code
        python_code_blocks = re.findall(r'```python(.*?)```', pred, re.DOTALL)
        if not python_code_blocks:
            scores.append(0.0)
            continue
            
        all_code = '\n'.join(python_code_blocks)
        
        # Check for good structure indicators
        structure_indicators = [
            ('# ', 0.1),           # Comments
            ('"""', 0.15),         # Docstrings
            ('def __init__', 0.1), # Proper init methods
            ('if __name__', 0.1),  # Main guard
            ('class ', 0.15),      # Object-oriented design
            ('import ', 0.05),     # Proper imports
            ('\n\n', 0.05),        # Good spacing
        ]
        
        for indicator, weight in structure_indicators:
            count = all_code.count(indicator)
            score += min(weight, count * weight * 0.5)  # Diminishing returns
        
        # Check for modular design
        function_count = all_code.count('def ')
        if function_count >= 3:
            score += 0.1
        if function_count >= 5:
            score += 0.1
            
        # Check for separation of concerns
        if 'class ' in all_code and 'def ' in all_code:
            score += 0.1
            
        scores.append(min(1.0, score))
    
    return scores


def code_test_coverage_score(predictions: List[str], references: List[str]) -> List[float]:
    """Evaluate test coverage and quality."""
    scores = []
    
    for pred in predictions:
        if not pred:
            scores.append(0.0)
            continue
            
        score = 0.0
        pred_lower = pred.lower()
        
        # Check for test presence
        test_indicators = [
            'test_',
            'assert',
            'unittest',
            'pytest',
            'test case',
            'unit test'
        ]
        
        test_mentions = sum(1 for indicator in test_indicators if indicator in pred_lower)
        if test_mentions > 0:
            score += 0.4
            
        # Check for specific test patterns
        python_code_blocks = re.findall(r'```python(.*?)```', pred, re.DOTALL)
        if python_code_blocks:
            all_code = '\n'.join(python_code_blocks)
            
            # Count assert statements
            assert_count = all_code.count('assert')
            score += min(0.3, assert_count * 0.1)
            
            # Check for test functions
            test_function_count = len(re.findall(r'def test_\w+', all_code))
            score += min(0.3, test_function_count * 0.1)
            
            # Bonus for comprehensive testing
            if 'setUp' in all_code or 'fixture' in all_code:
                score += 0.1
                
        scores.append(min(1.0, score))
    
    return scores


def phase_consistency_score(predictions: List[str], references: List[str]) -> List[float]:
    """Evaluate consistency between different phases."""
    scores = []
    
    for pred in predictions:
        if not pred:
            scores.append(0.0)
            continue
            
        # Extract phase responses
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        from utils import extract_phase_responses_single
        phases = extract_phase_responses_single(pred)
        
        if len(phases) < 2:
            scores.append(0.0)
            continue
            
        score = 0.5  # Base score
        
        # Check for consistency between PRD and Design
        if 'prd' in phases and 'design' in phases:
            prd_text = phases['prd'].lower()
            design_text = phases['design'].lower()
            
            # Look for shared concepts
            prd_words = set(re.findall(r'\b\w{4,}\b', prd_text))
            design_words = set(re.findall(r'\b\w{4,}\b', design_text))
            
            common_words = prd_words.intersection(design_words)
            if len(prd_words) > 0:
                consistency_ratio = len(common_words) / len(prd_words)
                score += consistency_ratio * 0.3
                
        # Check for consistency between Design and Implementation
        if 'design' in phases and 'implementation' in phases:
            design_text = phases['design'].lower()
            impl_text = phases['implementation'].lower()
            
            # Check if design concepts appear in implementation
            design_concepts = ['api', 'database', 'service', 'component', 'endpoint']
            consistency_bonus = 0.0
            
            for concept in design_concepts:
                if concept in design_text and concept in impl_text:
                    consistency_bonus += 0.04  # 0.2 total possible
                    
            score += consistency_bonus
            
        scores.append(min(1.0, score))
    
    return scores


def information_flow_score(predictions: List[str], references: List[str]) -> List[float]:
    """Evaluate how well information flows between phases."""
    scores = []
    
    for pred in predictions:
        if not pred:
            scores.append(0.0)
            continue
            
        # This metric would ideally analyze how each phase builds upon previous ones
        # For now, implement a basic version
        
        phase_transitions = [
            'based on the prd',
            'from the requirements',
            'according to the design',
            'implementing the',
            'following the specification',
            'as specified in'
        ]
        
        pred_lower = pred.lower()
        transition_count = sum(1 for transition in phase_transitions if transition in pred_lower)
        
        # Normalize based on presence of transitions
        score = min(1.0, transition_count * 0.25)
        scores.append(score)
    
    return scores