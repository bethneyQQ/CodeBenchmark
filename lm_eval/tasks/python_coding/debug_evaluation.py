#!/usr/bin/env python3
"""
Debug evaluation script for Python coding tasks.

This script runs evaluations with debug mode enabled to show detailed
problem information, model inputs, and model responses.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def run_debug_evaluation(task_name, limit=2, model_type="claude-local", model_name="claude-3-5-haiku-20241022"):
    """Run evaluation with debug mode enabled."""
    
    print(f"🐛 Debug Evaluation: {task_name}")
    print("=" * 60)
    print()
    
    # Set debug environment variable (for backward compatibility)
    env = os.environ.copy()
    env['PYTHON_CODING_DEBUG'] = 'true'
    
    # Use model_args to pass debug=true to the model
    model_args = f'model={model_name},debug=true'
    
    cmd = [
        'python', '-m', 'lm_eval',
        '--model', model_type,
        '--model_args', model_args,
        '--tasks', task_name,
        '--limit', str(limit),
        '--verbosity', 'INFO'
    ]
    
    print(f"🔄 Running: {' '.join(cmd)}")
    print(f"🐛 Debug mode: ENABLED (via model_args: debug=true)")
    print(f"🤖 Model: {model_type} ({model_name})")
    print(f"📊 Sample limit: {limit}")
    print()
    
    try:
        result = subprocess.run(cmd, env=env, text=True, timeout=300)
        
        if result.returncode == 0:
            print("\n✅ Debug evaluation completed successfully!")
        else:
            print(f"\n❌ Evaluation failed with return code: {result.returncode}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("\n⏰ Evaluation timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"\n❌ Error running evaluation: {e}")
        return False

def run_context_comparison_debug(limit=2):
    """Run context comparison with debug mode."""
    
    print("🔍 Debug Context Comparison")
    print("=" * 60)
    print()
    
    tasks = [
        ("python_code_completion", "Full Context"),
        ("python_code_completion_minimal_context", "Minimal Context"),
        ("python_code_completion_no_context", "No Context")
    ]
    
    results = []
    
    for task_name, description in tasks:
        print(f"\n🎯 Testing: {description}")
        print("-" * 40)
        
        success = run_debug_evaluation(task_name, limit)
        results.append((task_name, description, success))
        
        print(f"\nResult: {'✅ Success' if success else '❌ Failed'}")
        print("\n" + "="*60)
    
    # Summary
    print(f"\n📊 Debug Comparison Summary:")
    print("-" * 30)
    
    passed = 0
    for task_name, description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {description}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} evaluations completed")
    
    return passed == len(results)

def show_debug_usage():
    """Show debug mode usage examples."""
    
    print("\n🔧 Debug Mode Usage Examples")
    print("=" * 60)
    print()
    
    print("🐛 Environment Variable:")
    print("   export PYTHON_CODING_DEBUG=true")
    print()
    
    print("🚀 Debug Evaluation Commands:")
    print()
    
    examples = [
        ("Single Task Debug (Environment)", "PYTHON_CODING_DEBUG=true lm_eval --model claude-local --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion --limit 1"),
        ("Single Task Debug (Model Args)", "lm_eval --model claude-local --model_args model=claude-3-5-haiku-20241022,debug=true --tasks python_code_completion --limit 1"),
        ("Context Comparison Debug", "./run_context_comparison.sh --debug"),
        ("Debug Script", "python debug_evaluation.py --task python_code_completion --limit 2"),
        ("Context Comparison Script", "python debug_evaluation.py --context-comparison --limit 1")
    ]
    
    for name, command in examples:
        print(f"📝 {name}:")
        print(f"   {command}")
        print()
    
    print("💡 Debug Output Includes:")
    print("   - Problem category and context type")
    print("   - Raw problem data (code, descriptions, etc.)")
    print("   - Context information (raw and formatted)")
    print("   - Complete model input prompt")
    print("   - Model response with length and preview")
    print("   - Expected output comparison")
    print("   - Match analysis (exact, partial, none)")
    print("")
    print("🤖 Model Debug Support:")
    print("   - claude-local: Supports debug mode via 'debug=True' parameter")
    print("   - claude-code-local: Supports debug mode via 'debug=True' parameter")
    print("   - Both models also respect PYTHON_CODING_DEBUG environment variable")
    print("")
    print("⚙️  Generation Configuration:")
    print("   - until: [] (no stop sequences - prevents truncated responses)")
    print("   - max_gen_toks: 256-1024 (increased for complete responses)")
    print("   - temperature: 0.0 (deterministic output)")
    print("   - do_sample: false (consistent results)")
    print("")
    print("🔧 Code Extraction Features:")
    print("   - Robust extraction from ```python code blocks")
    print("   - Prioritizes function definitions over example usage")
    print("   - Removes explanatory text and test cases")
    print("   - Handles multiple code blocks correctly")
    print("   - Validates Python syntax before returning")

def main():
    """Main debug evaluation function."""
    
    parser = argparse.ArgumentParser(description='Debug evaluation for Python coding tasks')
    parser.add_argument('--task', help='Specific task to debug evaluate')
    parser.add_argument('--context-comparison', action='store_true', 
                       help='Run context comparison debug evaluation')
    parser.add_argument('--limit', type=int, default=2, 
                       help='Number of samples to evaluate (default: 2)')
    parser.add_argument('--model-type', default='claude-local',
                       help='Model type (default: claude-local)')
    parser.add_argument('--model-name', default='claude-3-5-haiku-20241022',
                       help='Model name (default: claude-3-5-haiku-20241022)')
    
    args = parser.parse_args()
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    print("🐛 Python Coding Debug Evaluation")
    print("=" * 60)
    print()
    
    if args.context_comparison:
        success = run_context_comparison_debug(args.limit)
    elif args.task:
        success = run_debug_evaluation(args.task, args.limit, args.model_type, args.model_name)
    else:
        show_debug_usage()
        return 0
    
    if success:
        print("\n🎉 Debug evaluation completed successfully!")
        print("\n💡 Review the debug output above to understand:")
        print("   - How problems are processed and formatted")
        print("   - What context information is provided to the model")
        print("   - How the model responds to different prompts")
        print("   - Differences between context modes")
    else:
        print("\n⚠️  Some evaluations failed - check the output above")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())