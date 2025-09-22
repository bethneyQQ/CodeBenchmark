#!/usr/bin/env python3
"""
Demo script to showcase the configurable context system for Python coding evaluation.

This script demonstrates:
1. How context can be configured (Full/Minimal/None)
2. Company-specific contexts vs generic contexts
3. Different context types (style_guide, security_policy, etc.)
4. How to run evaluations with different context modes
"""

import sys
import os
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))
import utils

def load_sample_problems():
    """Load sample problems to demonstrate context variations."""
    problems = []
    with open('problems.jsonl', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems

def demonstrate_context_modes():
    """Demonstrate different context modes with sample problems."""
    
    print("🐍 Python Coding Evaluation - Context System Demo")
    print("=" * 60)
    print()
    
    problems = load_sample_problems()
    
    # Show different context types
    context_types = set(p.get('context_type', 'general') for p in problems)
    print(f"📋 Available Context Types: {', '.join(sorted(context_types))}")
    print()
    
    # Demonstrate with one problem from each category
    categories = {}
    for problem in problems:
        category = problem['category']
        if category not in categories:
            categories[category] = problem
    
    for category, problem in categories.items():
        print(f"📝 {category.replace('_', ' ').title()} Example")
        print("-" * 40)
        print(f"Context Type: {problem.get('context_type', 'general')}")
        print()
        
        # Full Context
        print("🔧 Full Context Mode:")
        full_context = utils.format_context(problem, {
            'enable_context': True, 
            'context_mode': 'full'
        })
        print(f"   {full_context}")
        print()
        
        # Minimal Context  
        print("📝 Minimal Context Mode:")
        minimal_context = utils.format_context(problem, {
            'enable_context': True, 
            'context_mode': 'minimal'
        })
        print(f"   {minimal_context}")
        print()
        
        # No Context
        print("🚫 No Context Mode:")
        no_context = utils.format_context(problem, {
            'enable_context': False, 
            'context_mode': 'none'
        })
        print(f"   {no_context}")
        print()
        print()

def demonstrate_prompt_generation():
    """Show how different context modes affect actual prompts."""
    
    print("🎯 Prompt Generation Examples")
    print("=" * 60)
    print()
    
    # Load a code completion problem
    problems = load_sample_problems()
    code_completion_problem = next(p for p in problems if p['category'] == 'code_completion')
    
    print("📋 Sample Code Completion Problem:")
    print(f"Context Type: {code_completion_problem.get('context_type')}")
    print(f"Code: {code_completion_problem['incomplete_code'][:50]}...")
    print()
    
    # Generate prompts with different context modes
    modes = [
        ('Full Context', utils.doc_to_text),
        ('Minimal Context', utils.doc_to_text_minimal_context), 
        ('No Context', utils.doc_to_text_no_context)
    ]
    
    for mode_name, prompt_func in modes:
        print(f"🔧 {mode_name} Prompt:")
        print("-" * 30)
        prompt = prompt_func(code_completion_problem)
        print(prompt)
        print()
        print()

def demonstrate_environment_configuration():
    """Show how to configure context via environment variables."""
    
    print("⚙️  Environment Configuration Examples")
    print("=" * 60)
    print()
    
    print("You can configure context behavior using environment variables:")
    print()
    
    print("🔧 Enable/Disable Context:")
    print("   export PYTHON_CODING_ENABLE_CONTEXT=true   # Enable context")
    print("   export PYTHON_CODING_ENABLE_CONTEXT=false  # Disable context")
    print()
    
    print("📝 Set Context Mode:")
    print("   export PYTHON_CODING_CONTEXT_MODE=full     # Full company contexts")
    print("   export PYTHON_CODING_CONTEXT_MODE=minimal  # Key requirements only")
    print("   export PYTHON_CODING_CONTEXT_MODE=none     # No context")
    print()
    
    print("🏷️  Filter by Context Types:")
    print("   export PYTHON_CODING_CONTEXT_TYPES=style_guide,security_policy")
    print("   # Only use style guide and security policy contexts")
    print()
    
    print("🚀 Example Evaluation Commands:")
    print()
    print("# Run with full context (default)")
    print("lm_eval --model anthropic-chat --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion")
    print()
    print("# Run without context")
    print("PYTHON_CODING_ENABLE_CONTEXT=false lm_eval --model anthropic-chat --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion")
    print()
    print("# Run with minimal context")
    print("PYTHON_CODING_CONTEXT_MODE=minimal lm_eval --model anthropic-chat --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion")
    print()
    print("# Run with specific context types only")
    print("PYTHON_CODING_CONTEXT_TYPES=security_policy,performance_requirements lm_eval --model anthropic-chat --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion")
    print()

def demonstrate_context_analysis():
    """Show how to run context impact analysis."""
    
    print("📊 Context Impact Analysis")
    print("=" * 60)
    print()
    
    print("The framework includes tools to analyze context impact:")
    print()
    
    print("🔍 Available Analysis Tools:")
    print("   1. analyze_context_impact.py - Compare different context modes")
    print("   2. run_context_comparison.sh - Automated comparison script")
    print()
    
    print("🚀 Quick Analysis Commands:")
    print()
    print("# Run comprehensive context comparison")
    print("./run_context_comparison.sh")
    print()
    print("# Run custom analysis")
    print("python analyze_context_impact.py --output_dir results/my_analysis")
    print()
    
    print("📈 Generated Outputs:")
    print("   - Detailed analysis report (Markdown)")
    print("   - Performance comparison charts (PNG)")
    print("   - Context impact heatmaps")
    print("   - Individual task results (JSON)")
    print()

def main():
    """Main demo function."""
    
    # Change to the script directory
    os.chdir(Path(__file__).parent)
    
    try:
        demonstrate_context_modes()
        demonstrate_prompt_generation()
        demonstrate_environment_configuration()
        demonstrate_context_analysis()
        
        print("✅ Context System Demo Complete!")
        print()
        print("🎯 Key Features Demonstrated:")
        print("   ✓ Configurable context modes (Full/Minimal/None)")
        print("   ✓ Company-specific contexts vs generic prompts")
        print("   ✓ Environment variable configuration")
        print("   ✓ Context impact analysis tools")
        print("   ✓ Multiple task variants for comparison")
        print()
        print("🚀 Next Steps:")
        print("   1. Run evaluations with different context modes")
        print("   2. Use analysis tools to measure context impact")
        print("   3. Optimize context based on results")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())