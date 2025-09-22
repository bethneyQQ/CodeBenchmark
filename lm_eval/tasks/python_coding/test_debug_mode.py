#!/usr/bin/env python3
"""
Test script to demonstrate debug mode functionality.

This script shows how debug mode provides detailed information about:
- Problem details and context
- Model input prompts
- Model responses and analysis
"""

import os
import sys
import json
from pathlib import Path

def test_debug_logging():
    """Test debug logging functionality."""
    
    # Change to script directory and add to path
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    sys.path.append(str(script_dir))
    
    # Enable debug mode
    os.environ['PYTHON_CODING_DEBUG'] = 'true'
    
    import utils
    
    print("🐛 Debug Mode Test - Python Coding Evaluation")
    print("=" * 60)
    print()
    
    # Load sample problems
    problems = []
    with open('problems.jsonl', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    
    # Test one problem from each category
    categories_tested = set()
    
    for problem in problems:
        category = problem['category']
        
        if category in categories_tested:
            continue
            
        categories_tested.add(category)
        
        print(f"\n🎯 Testing Category: {category.replace('_', ' ').title()}")
        print("=" * 60)
        
        # Generate prompt based on category
        if category == 'code_completion':
            prompt = utils.doc_to_text(problem)
            expected = problem.get('expected_completion')
        elif category == 'code_repair':
            prompt = utils.doc_to_text_repair(problem)
            expected = problem.get('fixed_code')
        elif category == 'code_translation':
            prompt = utils.doc_to_text_translation(problem)
            expected = problem.get('target_code')
        elif category == 'docstring_generation':
            prompt = utils.doc_to_text_docstring(problem)
            expected = problem.get('expected_docstring')
        elif category == 'function_generation':
            prompt = utils.doc_to_text_function(problem)
            expected = '\n'.join(problem.get('test_cases', []))
        else:
            continue
        
        # Simulate a model response for debug logging
        mock_response = f"# Mock response for {category}\n{expected[:100]}..."
        utils.debug_log_response(mock_response, expected, category)
        
        if len(categories_tested) >= 3:  # Test first 3 categories
            break
    
    print("\n✅ Debug Mode Test Complete!")
    print()
    print("💡 Debug mode provides:")
    print("   ✓ Detailed problem information")
    print("   ✓ Context type and formatting details")
    print("   ✓ Complete model input prompts")
    print("   ✓ Model response analysis")
    print("   ✓ Expected vs actual comparison")

def show_debug_usage():
    """Show debug mode usage examples."""
    
    print("\n🔧 Debug Mode Usage Guide")
    print("=" * 60)
    print()
    
    print("🐛 Enable Debug Mode:")
    print("   export PYTHON_CODING_DEBUG=true")
    print()
    
    print("🚀 Debug Evaluation Examples:")
    print()
    
    examples = [
        ("Single Task Debug", "PYTHON_CODING_DEBUG=true lm_eval --model anthropic-chat --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion --limit 1"),
        ("Context Comparison Debug", "PYTHON_CODING_DEBUG=true ./run_context_comparison.sh"),
        ("Debug Script", "python debug_evaluation.py --task python_code_completion --limit 1"),
        ("Simple Test Debug", "PYTHON_CODING_DEBUG=true python simple_context_test.py")
    ]
    
    for name, command in examples:
        print(f"📝 {name}:")
        print(f"   {command}")
        print()
    
    print("📊 Debug Output Features:")
    print("   🔍 Problem category and context type identification")
    print("   📋 Raw problem data (code snippets, descriptions, requirements)")
    print("   🔧 Context information (both raw and formatted)")
    print("   📝 Complete model input prompt with character count")
    print("   🤖 Model response with length and content preview")
    print("   🎯 Expected output comparison and match analysis")
    print("   ✅ Exact match, partial match, or no match indicators")

def main():
    """Main function."""
    
    if len(sys.argv) > 1 and sys.argv[1] == '--usage':
        show_debug_usage()
    else:
        test_debug_logging()
        show_debug_usage()

if __name__ == "__main__":
    main()