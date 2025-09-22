#!/usr/bin/env python3
"""
Example usage of the Python coding evaluation framework.
"""

import json
import subprocess
import sys
from pathlib import Path
import os

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)

def show_sample_problems():
    """Display sample problems from each category."""
    print("📋 Sample Problems from Each Category\n")
    
    categories_shown = set()
    
    with open('problems.jsonl', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            data = json.loads(line)
            category = data['category']
            
            if category not in categories_shown:
                print(f"🔹 {category.replace('_', ' ').title()}")
                
                if category == 'code_completion':
                    print(f"   Context: {data.get('context', 'None')}")
                    print(f"   Code: {data['incomplete_code'][:50]}...")
                    
                elif category == 'code_repair':
                    print(f"   Error: {data['error_description'][:50]}...")
                    print(f"   Buggy code: {data['buggy_code'][:50]}...")
                    
                elif category == 'code_translation':
                    print(f"   From: {data['source_language']}")
                    print(f"   Code: {data['source_code'][:50]}...")
                    
                elif category == 'docstring_generation':
                    print(f"   Function: {data['function_code'][:50]}...")
                    
                elif category == 'function_generation':
                    print(f"   Task: {data['function_description'][:50]}...")
                
                print()
                categories_shown.add(category)
                
                if len(categories_shown) >= 5:
                    break

def run_sample_evaluation():
    """Run a sample evaluation with a small model."""
    print("🚀 Running Sample Evaluation\n")
    
    # Check if lm_eval is installed
    try:
        result = subprocess.run(['lm_eval', '--help'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ lm_eval not found. Please install with: pip install lm-eval")
            return False
    except FileNotFoundError:
        print("❌ lm_eval not found. Please install with: pip install lm-eval")
        return False
    
    print("✅ lm_eval found")
    
    # Run a quick test with a small model (limit to 2 samples)
    cmd = [
        'lm_eval',
        '--model', 'hf',
        '--model_args', 'pretrained=microsoft/DialoGPT-small',
        '--tasks', 'python_coding_suite',
        '--limit', '2',
        '--log_samples'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("This may take a few minutes...\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Sample evaluation completed successfully!")
            print("\nResults preview:")
            # Show last few lines of output
            lines = result.stdout.split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"  {line}")
        else:
            print("❌ Evaluation failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Evaluation timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False
    
    return True

def show_metrics_info():
    """Show information about available metrics."""
    print("📊 Available Metrics\n")
    
    metrics_info = {
        "exact_match": "Binary match between prediction and reference",
        "bleu_score": "BLEU score for text similarity (0-1)",
        "rouge_score": "ROUGE scores for text overlap (ROUGE-1, ROUGE-2, ROUGE-L)",
        "meteor_score": "METEOR score for semantic similarity (0-1)",
        "code_bleu": "Specialized BLEU for code translation (0-1)",
        "edit_distance": "Normalized Levenshtein distance (0-1, lower is better)",
        "pass_at_k": "Percentage of code that passes test cases (0-1)"
    }
    
    for metric, description in metrics_info.items():
        print(f"🔹 {metric}: {description}")
    
    print("\nNote: Some metrics require additional setup (e.g., HF_ALLOW_CODE_EVAL=1 for pass_at_k)")

def main():
    """Main example script."""
    print("🐍 Python Coding Evaluation Framework - Example Usage\n")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "problems":
            show_sample_problems()
        elif command == "metrics":
            show_metrics_info()
        elif command == "eval":
            run_sample_evaluation()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: problems, metrics, eval")
    else:
        print("Available commands:")
        print("  python example_usage.py problems  - Show sample problems")
        print("  python example_usage.py metrics   - Show available metrics")
        print("  python example_usage.py eval      - Run sample evaluation")
        print()
        
        # Show a quick overview
        show_sample_problems()

if __name__ == "__main__":
    main()