#!/usr/bin/env python3
"""
Metadata filtering example for single_turn_scenarios task.

This script demonstrates how to use metadata filtering to run evaluations
on specific subsets of the dataset.
"""

import sys
import os
from pathlib import Path

# Add lm_eval to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def demonstrate_filtering():
    """Demonstrate various metadata filtering options."""
    print("Single Turn Scenarios - Metadata Filtering Example")
    print("=" * 55)
    
    try:
        # Import task utilities
        task_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(task_dir))
        
        from utils import load_dataset, filter_by_metadata
        
        # Load full dataset
        print("Loading full dataset...")
        full_dataset = load_dataset()
        print(f"Total problems: {len(full_dataset)}")
        
        # Analyze dataset composition
        print("\nDataset Composition:")
        print("-" * 30)
        
        scenarios = {}
        languages = {}
        difficulties = {}
        context_modes = {}
        
        for item in full_dataset:
            # Count scenarios
            scenario = item.get('scenario', 'unknown')
            scenarios[scenario] = scenarios.get(scenario, 0) + 1
            
            # Count languages
            language = item.get('language', 'unknown')
            languages[language] = languages.get(language, 0) + 1
            
            # Count difficulties
            difficulty = item.get('difficulty', 'unknown')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
            
            # Count context modes
            context_mode = item.get('context_mode', 'unknown')
            context_modes[context_mode] = context_modes.get(context_mode, 0) + 1
        
        print("Scenarios:")
        for scenario, count in sorted(scenarios.items()):
            print(f"  {scenario}: {count}")
        
        print("\nLanguages:")
        for language, count in sorted(languages.items()):
            print(f"  {language}: {count}")
        
        print("\nDifficulties:")
        for difficulty, count in sorted(difficulties.items()):
            print(f"  {difficulty}: {count}")
        
        print("\nContext Modes:")
        for context_mode, count in sorted(context_modes.items()):
            print(f"  {context_mode}: {count}")
        
        # Demonstrate filtering examples
        print("\n" + "=" * 55)
        print("Filtering Examples:")
        print("-" * 30)
        
        # Example 1: Filter by scenario
        if 'code_completion' in scenarios:
            filtered = filter_by_metadata(full_dataset, {'scenario': 'code_completion'})
            print(f"Code completion only: {len(filtered)} problems")
        
        # Example 2: Filter by language
        if 'python' in languages:
            filtered = filter_by_metadata(full_dataset, {'language': 'python'})
            print(f"Python only: {len(filtered)} problems")
        
        # Example 3: Filter by difficulty
        if 'intermediate' in difficulties:
            filtered = filter_by_metadata(full_dataset, {'difficulty': 'intermediate'})
            print(f"Intermediate difficulty: {len(filtered)} problems")
        
        # Example 4: Multiple filters
        if 'python' in languages and 'intermediate' in difficulties:
            filtered = filter_by_metadata(full_dataset, {
                'language': 'python',
                'difficulty': 'intermediate'
            })
            print(f"Python + Intermediate: {len(filtered)} problems")
        
        # Example 5: Multiple values for same filter
        if 'python' in languages and 'javascript' in languages:
            filtered = filter_by_metadata(full_dataset, {
                'language': ['python', 'javascript']
            })
            print(f"Python or JavaScript: {len(filtered)} problems")
        
        # Example 6: Complex filtering
        complex_filters = {
            'scenario': ['code_completion', 'bug_fix'],
            'language': 'python',
            'difficulty': ['simple', 'intermediate']
        }
        
        try:
            filtered = filter_by_metadata(full_dataset, complex_filters)
            print(f"Complex filter (code_completion/bug_fix + python + simple/intermediate): {len(filtered)} problems")
        except Exception as e:
            print(f"Complex filter failed: {e}")
        
        # Show CLI equivalents
        print("\n" + "=" * 55)
        print("CLI Command Equivalents:")
        print("-" * 30)
        
        print("# Run code completion tasks only:")
        print("lm_eval --tasks single_turn_scenarios_code_completion")
        
        print("\n# Run Python tasks only:")
        print("lm_eval --tasks single_turn_scenarios_python")
        
        print("\n# Run intermediate difficulty tasks:")
        print("lm_eval --tasks single_turn_scenarios_intermediate")
        
        print("\n# Custom filtering with task config:")
        print('lm_eval --tasks single_turn_scenarios_suite --task_config \'{"dataset_kwargs": {"metadata": {"language": "python", "difficulty": "intermediate"}}}\'')
        
        print("\n# Multiple scenarios with filtering:")
        print('lm_eval --tasks single_turn_scenarios_code_completion,single_turn_scenarios_bug_fix --task_config \'{"dataset_kwargs": {"metadata": {"language": ["python", "javascript"]}}}\'')
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run the metadata filtering demonstration."""
    success = demonstrate_filtering()
    
    if success:
        print("\n✓ Metadata filtering demonstration completed successfully!")
    else:
        print("\n✗ Metadata filtering demonstration failed.")
        print("Check that the dataset and utilities are properly configured.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)