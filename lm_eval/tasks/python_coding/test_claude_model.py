#!/usr/bin/env python3
"""
Quick test to verify Claude model integration works with the Python coding tasks.
"""

import subprocess
import sys
import os
from pathlib import Path

def test_claude_model():
    """Test a simple evaluation with Claude model."""
    
    print("🧪 Testing Claude Model Integration")
    print("=" * 40)
    print()
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    print("🤖 Model: anthropic-chat (claude-3-5-haiku-20241022)")
    print("📝 Task: python_code_completion")
    print("📊 Limit: 1 sample for quick test")
    print()
    
    cmd = [
        'python', '-m', 'lm_eval',
        '--model', 'anthropic-chat',
        '--model_args', 'model=claude-3-5-haiku-20241022',
        '--tasks', 'python_code_completion',
        '--limit', '1',
        '--verbosity', 'INFO'
    ]
    
    print(f"🔄 Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            print("✅ Claude model integration successful!")
            print()
            print("📊 Sample Output:")
            # Show relevant lines from output
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['exact_match', 'bleu', 'python_code_completion']):
                    print(f"   {line.strip()}")
            
            print()
            print("🎉 Ready to run full evaluations with Claude!")
            return True
            
        else:
            print("❌ Claude model test failed")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (3 minutes)")
        print("This might indicate API issues or slow response times")
        return False
        
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def show_usage_examples():
    """Show usage examples with Claude model."""
    
    print("\n🚀 Usage Examples with Claude Model")
    print("=" * 40)
    print()
    
    examples = [
        ("Single Task", "lm_eval --model anthropic-chat --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion"),
        ("Context Comparison", "lm_eval --model anthropic-chat --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion,python_code_completion_minimal_context,python_code_completion_no_context"),
        ("Complete Suite", "lm_eval --model anthropic-chat --model_args model=claude-3-5-haiku-20241022 --tasks python_coding_suite"),
        ("No Context Mode", "PYTHON_CODING_ENABLE_CONTEXT=false lm_eval --model anthropic-chat --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion"),
        ("Context Analysis", "./run_context_comparison.sh")
    ]
    
    for name, command in examples:
        print(f"📝 {name}:")
        print(f"   {command}")
        print()

def main():
    """Main test function."""
    
    print("Claude Model Integration Test for Python Coding Tasks")
    print()
    
    # Test the model
    success = test_claude_model()
    
    # Show usage examples
    show_usage_examples()
    
    if success:
        print("✅ All systems ready for Claude model evaluation!")
    else:
        print("⚠️  Please check Claude API configuration and try again")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())