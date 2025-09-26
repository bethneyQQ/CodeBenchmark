#!/usr/bin/env python3
"""
Minimal evaluation example for single_turn_scenarios task.

This script demonstrates the simplest way to run a single scenario evaluation
for testing and validation purposes.
"""

import sys
import os
from pathlib import Path

# Add lm_eval to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lm_eval import evaluator
from lm_eval.models import get_model

def main():
    """Run a minimal evaluation example."""
    print("Single Turn Scenarios - Minimal Evaluation Example")
    print("=" * 50)
    
    try:
        # Initialize a simple model (using HuggingFace dummy model for testing)
        print("Initializing model...")
        model = get_model("hf").create_from_arg_string(
            "pretrained=gpt2,device=cpu"  # Using small model for testing
        )
        
        # Run evaluation on a single scenario with limit
        print("Running evaluation...")
        results = evaluator.simple_evaluate(
            model=model,
            tasks=["single_turn_scenarios_code_completion"],
            limit=2,  # Only run 2 samples for quick testing
            batch_size=1
        )
        
        # Print results
        print("\nResults:")
        print("-" * 30)
        for task_name, task_results in results["results"].items():
            print(f"Task: {task_name}")
            for metric_name, metric_value in task_results.items():
                print(f"  {metric_name}: {metric_value:.4f}")
        
        print("\nEvaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("This is expected if you don't have the required dependencies.")
        print("To run a full evaluation, ensure you have:")
        print("- A proper model configured")
        print("- All dependencies installed")
        print("- Docker for sandbox execution (optional)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)