#!/usr/bin/env python3
"""
Simple test script to run a single evaluation and verify it works.
This should be run from the lm-evaluation-harness root directory.
"""

import subprocess
import sys
import json
from pathlib import Path

def test_single_evaluation():
    """Test a single evaluation with limit 1."""
    
    print("🧪 Testing Single Python Code Completion Evaluation")
    print("=" * 60)
    
    # Check if we're in the right directory (should be in python_coding task directory)
    current_dir = Path.cwd()
    if current_dir.name != "python_coding" or not (current_dir / "problems.jsonl").exists():
        print("❌ Error: This script should be run from the python_coding task directory")
        print(f"   Current directory: {current_dir}")
        print("   Expected: .../lm_eval/tasks/python_coding/")
        print("   Please run: cd lm_eval/tasks/python_coding && python test_single_evaluation.py")
        return False
    
    print(f"✅ Running from correct directory: {current_dir}")
    
    # Create test output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "single_test.json"
    
    # Remove existing file
    if output_file.exists():
        output_file.unlink()
    
    # Build command
    cmd = [
        'lm_eval',
        '--model', 'claude-local',
        '--model_args', 'model=claude-3-5-haiku-20241022',
        '--tasks', 'python_code_completion',
        '--output_path', str(output_file),
        '--limit', '1',
        '--verbosity', 'INFO'
    ]
    
    print(f"🚀 Running command:")
    print(f"   {' '.join(cmd)}")
    print("")
    
    try:
        print("⏳ Running evaluation (this may take a minute)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        print(f"📊 Exit code: {result.returncode}")
        
        if result.returncode == 0:
            # lm_eval adds timestamps to filenames, so look for files that start with our expected name
            output_dir = output_file.parent
            base_name = output_file.stem  # filename without .json
            actual_files = list(output_dir.glob(f"{base_name}*.json"))
            
            if actual_files:
                actual_file = actual_files[0]  # Take the first match
                print(f"✅ SUCCESS! Evaluation completed")
                print(f"📁 Output file: {actual_file}")
                print(f"📊 File size: {actual_file.stat().st_size} bytes")
                
                # Check file content
                try:
                    with open(actual_file, 'r') as f:
                        data = json.load(f)
                        
                    print(f"📋 Results summary:")
                    results = data.get('results', {})
                    for task_name, task_results in results.items():
                        print(f"   Task: {task_name}")
                        for metric, value in task_results.items():
                            if isinstance(value, (int, float)):
                                print(f"     {metric}: {value:.4f}")
                            else:
                                print(f"     {metric}: {value}")
                    
                    print(f"\n🎉 Test completed successfully!")
                    return True
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️  Output file exists but contains invalid JSON: {e}")
                    return False
                    
            else:
                print(f"⚠️  Command succeeded but no output file found")
                print(f"🔍 Expected pattern: {output_dir}/{base_name}*.json")
                print(f"🔍 Files in output directory:")
                for f in output_dir.glob("*.json"):
                    print(f"      {f.name}")
                print("📥 STDERR:")
                print(result.stderr[-1000:] if result.stderr else "No stderr")
                return False
        else:
            print(f"❌ Evaluation failed with exit code {result.returncode}")
            print("📥 STDERR:")
            print(result.stderr[-1000:] if result.stderr else "No stderr")
            print("📤 STDOUT:")
            print(result.stdout[-1000:] if result.stdout else "No stdout")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Evaluation timed out after 3 minutes")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def main():
    """Main function."""
    success = test_single_evaluation()
    
    if success:
        print("\n💡 Next steps:")
        print("   1. Run the full context comparison:")
        print("      ./lm_eval/tasks/python_coding/run_from_root.sh")
        print("   2. Or with debug mode:")
        print("      ./lm_eval/tasks/python_coding/run_from_root.sh --debug")
    else:
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you have the ANTHROPIC_API_KEY environment variable set")
        print("   2. Install the anthropic package: pip install anthropic")
        print("   3. Make sure you're running from the lm-evaluation-harness root directory")
        print("   4. Check that the claude-local model is properly configured")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)