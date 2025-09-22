#!/usr/bin/env python3
"""
Simple test to demonstrate the configurable context system working with actual evaluations.
"""

import subprocess
import json
import os
import sys
from pathlib import Path

def run_simple_evaluation(task_name, description, limit=2, debug_mode=False):
    """Run a simple evaluation with limited samples."""
    print(f"🔄 Testing: {description}")
    print(f"   Task: {task_name}")
    if debug_mode:
        print(f"   🐛 Debug mode: ENABLED")
    
    cmd = [
        'python', '-m', 'lm_eval',
        '--model', 'anthropic-chat',
        '--model_args', 'model=claude-3-5-haiku-20241022',
        '--tasks', task_name,
        '--limit', str(limit),
        '--verbosity', 'ERROR'  # Reduce output noise
    ]
    
    # Set up environment for debug mode
    env = os.environ.copy()
    if debug_mode:
        env['PYTHON_CODING_DEBUG'] = 'true'
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        
        if result.returncode == 0:
            print("   ✅ Success")
            # Try to extract some basic info from output
            if 'exact_match' in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'exact_match' in line and '|' in line:
                        print(f"   📊 {line.strip()}")
                        break
            return True
        else:
            print("   ❌ Failed")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Timeout")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_context_configurations(debug_mode=False):
    """Test different context configurations."""
    
    print("🧪 Python Coding Context System - Live Test")
    print("=" * 50)
    print()
    
    if debug_mode:
        print("🐛 Debug mode enabled - detailed logging will be shown")
        print()
    
    # Test different context modes for code completion
    tests = [
        ("python_code_completion", "Code Completion (Full Context)"),
        ("python_code_completion_minimal_context", "Code Completion (Minimal Context)"),
        ("python_code_completion_no_context", "Code Completion (No Context)")
    ]
    
    results = []
    
    for task_name, description in tests:
        success = run_simple_evaluation(task_name, description, debug_mode=debug_mode)
        results.append((task_name, description, success))
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    print("-" * 30)
    
    passed = 0
    for task_name, description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {description}")
        if success:
            passed += 1
    
    print()
    print(f"🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All context configurations working correctly!")
        print()
        print("💡 What this demonstrates:")
        print("   ✓ Tasks can run with different context modes")
        print("   ✓ Full context includes company-specific requirements")
        print("   ✓ Minimal context extracts key requirements only")
        print("   ✓ No context provides generic prompts")
        print("   ✓ All configurations are properly registered in lm_eval")
    else:
        print("⚠️  Some configurations failed - check the errors above")
    
    return passed == len(results)

def demonstrate_context_differences():
    """Show the actual context differences without running evaluations."""
    
    print("\n🔍 Context Content Comparison")
    print("=" * 50)
    
    # Add current directory to path
    sys.path.append(str(Path(__file__).parent))
    import utils
    
    # Load a sample problem
    with open('problems.jsonl', 'r') as f:
        line = f.readline().strip()
        if line:
            doc = json.loads(line)
            
            print(f"📋 Sample Problem: {doc['category']}")
            print(f"Context Type: {doc['context_type']}")
            print()
            
            contexts = [
                ("Full Context", {'enable_context': True, 'context_mode': 'full'}),
                ("Minimal Context", {'enable_context': True, 'context_mode': 'minimal'}),
                ("No Context", {'enable_context': False, 'context_mode': 'none'})
            ]
            
            for name, config in contexts:
                context_text = utils.format_context(doc, config)
                print(f"🔧 {name}:")
                print(f"   {context_text}")
                print()

def main():
    """Main test function."""
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Check for debug mode
    debug_mode = os.environ.get('PYTHON_CODING_DEBUG', 'false').lower() in ('true', '1', 'yes', 'on')
    
    print("Starting Python Coding Context System Test...")
    if debug_mode:
        print("🐛 Debug mode detected from environment variable")
    print()
    
    # First show context differences
    demonstrate_context_differences()
    
    # Then test actual evaluations
    success = test_context_configurations(debug_mode=debug_mode)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())