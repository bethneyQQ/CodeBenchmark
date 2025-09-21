#!/usr/bin/env python3
"""
Simple test script to verify the Python Code Quality task works correctly.
This runs a basic test without requiring API keys.
"""

import sys
import os
import subprocess
import tempfile
import shutil

def test_task_loading():
    """Test that the task can be loaded by lm-eval."""
    print("🔍 Testing task loading...")
    
    try:
        # Test that the task is registered
        result = subprocess.run(
            ["lm_eval", "--tasks", "list"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        if "python_code_quality" in result.stdout:
            print("✅ Task is properly registered with lm-eval")
            return True
        else:
            print("❌ Task not found in lm-eval task list")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running lm_eval: {e}")
        return False
    except FileNotFoundError:
        print("❌ lm_eval command not found. Please install lm-eval.")
        return False

def test_task_with_dummy_model():
    """Test the task with dummy model to verify it loads correctly."""
    print("🔍 Testing task with dummy model...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            result = subprocess.run([
                "lm_eval",
                "--model", "dummy",
                "--tasks", "python_code_quality",
                "--limit", "1",
                "--predict_only",
                "--output_path", os.path.join(temp_dir, "test_output")
            ], capture_output=True, text=True, check=True)
            
            print("✅ Task runs successfully with dummy model")
            
            # Check if output files were created
            output_files = os.listdir(temp_dir)
            if any("test_output" in f for f in output_files):
                print("✅ Output files created successfully")
                return True
            else:
                print("⚠️  Task ran but no output files found")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running task with dummy model:")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
            return False

def test_task_components():
    """Test the task components directly."""
    print("🔍 Testing task components...")
    
    try:
        # Add the task directory to Python path
        task_dir = os.path.join("lm_eval", "tasks", "code_check")
        if not os.path.exists(task_dir):
            print(f"❌ Task directory not found: {task_dir}")
            return False
        
        sys.path.insert(0, task_dir)
        
        # Import and test the utils
        import utils
        
        # Test dataset loading
        try:
            # Test raw JSON loading
            import json
            dataset_path = os.path.join(task_dir, "python_code_quality_dataset.json")
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            if len(data) == 20:
                print("✅ Dataset loads correctly (20 problems)")
            else:
                print(f"⚠️  Dataset has {len(data)} problems, expected 20")
            
            # Test code extraction
            test_code = "```python\ndef test():\n    return 42\n```"
            extracted = utils.extract_python_code(test_code)
            if "def test" in extracted:
                print("✅ Code extraction works")
            else:
                print("❌ Code extraction failed")
                return False
            
            # Test safe execution
            simple_code = "def add(a, b):\n    return a + b"
            test_cases = ["assert add(2, 3) == 5"]
            result = utils.safe_execute(simple_code, test_cases)
            
            if result['syntax_valid'] and result['execution_successful'] and result['tests_passed'] == 1:
                print("✅ Safe code execution works")
            else:
                print("❌ Safe code execution failed")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing task components: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Error importing task utils: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Python Code Quality Task - Simple Test")
    print("=" * 50)
    
    tests = [
        ("Task Loading", test_task_loading),
        ("Task Components", test_task_components),
        ("Dummy Model Run", test_task_with_dummy_model),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The task is ready to use.")
        print("\n📝 Next steps:")
        print("1. Set up your Anthropic API key: export ANTHROPIC_API_KEY='your_key'")
        print("2. Run the test with Claude:")
        print("   lm_eval --model anthropic-completions --model_args model=claude-3-haiku-20240307 --tasks python_code_quality --limit 2 --output_path test_results.json --log_samples")
        print("   Or use the test script: python test_with_claude.py")
        print("3. Then run with your local model")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)