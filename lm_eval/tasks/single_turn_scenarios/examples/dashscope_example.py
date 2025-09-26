#!/usr/bin/env python3
"""
DashScope Usage Example

This example demonstrates how to use DashScope (Qwen models) with the 
single_turn_scenarios evaluation framework.

Requirements:
- DashScope API key set in environment variable DASHSCOPE_API_KEY
- lm-eval installed with single_turn_scenarios support
"""

import os
import sys
import json
from pathlib import Path

# Add the task directory to Python path
task_dir = Path(__file__).parent.parent
sys.path.insert(0, str(task_dir))

def check_dashscope_setup():
    """Check if DashScope is properly configured."""
    print("🔍 Checking DashScope setup...")
    
    # Check API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ DASHSCOPE_API_KEY environment variable not set")
        print("   Please set your DashScope API key:")
        print("   export DASHSCOPE_API_KEY='your-dashscope-api-key'")
        return False
    
    print(f"✅ DashScope API key found: {api_key[:8]}...")
    
    # Test API connectivity
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Simple test request
        test_data = {
            "model": "qwen-turbo",
            "input": {
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            },
            "parameters": {
                "max_tokens": 10
            }
        }
        
        response = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers=headers,
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ DashScope API connectivity verified")
            return True
        else:
            print(f"❌ DashScope API test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except ImportError:
        print("⚠️  requests library not available, skipping connectivity test")
        return True
    except Exception as e:
        print(f"❌ DashScope API test failed: {e}")
        return False

def run_basic_evaluation():
    """Run a basic evaluation with DashScope."""
    print("\n🚀 Running basic DashScope evaluation...")
    
    import subprocess
    
    cmd = [
        "lm_eval",
        "--model", "dashscope",
        "--model_args", "model=qwen-coder-plus,temperature=0.0",
        "--tasks", "single_turn_scenarios_code_completion",
        "--limit", "3",
        "--output_path", "dashscope_basic_results.json",
        "--log_samples"
    ]
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Basic evaluation completed successfully")
            
            # Load and display results
            if os.path.exists("dashscope_basic_results.json"):
                with open("dashscope_basic_results.json", 'r') as f:
                    results = json.load(f)
                
                print("\n📊 Results Summary:")
                task_results = results.get("results", {}).get("single_turn_scenarios_code_completion", {})
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        print(f"   {metric}: {value:.4f}")
            
            return True
        else:
            print("❌ Basic evaluation failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Evaluation timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def run_model_comparison():
    """Compare different Qwen models."""
    print("\n🔄 Running model comparison...")
    
    import subprocess
    
    models = [
        ("qwen-coder-plus", "Latest code-optimized model"),
        ("qwen-coder", "Specialized code generation model"),
        ("qwen-turbo", "Fast and cost-effective model")
    ]
    
    results_summary = {}
    
    for model, description in models:
        print(f"\n📝 Evaluating {model} ({description})...")
        
        cmd = [
            "lm_eval",
            "--model", "dashscope",
            "--model_args", f"model={model},temperature=0.0",
            "--tasks", "single_turn_scenarios_function_generation",
            "--limit", "5",
            "--output_path", f"dashscope_{model}_results.json"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                print(f"   ✅ {model} evaluation completed")
                
                # Load results
                if os.path.exists(f"dashscope_{model}_results.json"):
                    with open(f"dashscope_{model}_results.json", 'r') as f:
                        data = json.load(f)
                    
                    task_results = data.get("results", {}).get("single_turn_scenarios_function_generation", {})
                    results_summary[model] = {
                        "exact_match": task_results.get("exact_match", 0),
                        "codebleu": task_results.get("codebleu", 0),
                        "pass_at_1": task_results.get("pass_at_1", 0)
                    }
            else:
                print(f"   ❌ {model} evaluation failed: {result.stderr}")
                results_summary[model] = {"error": "Evaluation failed"}
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ {model} evaluation timed out")
            results_summary[model] = {"error": "Timeout"}
        except Exception as e:
            print(f"   ❌ {model} evaluation error: {e}")
            results_summary[model] = {"error": str(e)}
    
    # Display comparison
    print("\n📊 Model Comparison Results:")
    print("=" * 60)
    print(f"{'Model':<20} {'Exact Match':<12} {'CodeBLEU':<10} {'Pass@1':<8}")
    print("-" * 60)
    
    for model, results in results_summary.items():
        if "error" in results:
            print(f"{model:<20} {results['error']}")
        else:
            exact_match = results.get("exact_match", 0)
            codebleu = results.get("codebleu", 0)
            pass_at_1 = results.get("pass_at_1", 0)
            print(f"{model:<20} {exact_match:<12.4f} {codebleu:<10.4f} {pass_at_1:<8.4f}")

def run_advanced_evaluation():
    """Run advanced evaluation with metadata filtering."""
    print("\n🎯 Running advanced evaluation with metadata filtering...")
    
    import subprocess
    
    # Python-specific complex problems
    cmd = [
        "lm_eval",
        "--model", "dashscope",
        "--model_args", "model=qwen-coder-plus,temperature=0.1",
        "--tasks", "single_turn_scenarios_algorithm_implementation",
        "--metadata", '{"language":"python","difficulty":"complex"}',
        "--limit", "8",
        "--output_path", "dashscope_advanced_results.json",
        "--log_samples"
    ]
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Advanced evaluation completed successfully")
            
            # Load and analyze results
            if os.path.exists("dashscope_advanced_results.json"):
                with open("dashscope_advanced_results.json", 'r') as f:
                    results = json.load(f)
                
                print("\n📊 Advanced Results Analysis:")
                task_results = results.get("results", {}).get("single_turn_scenarios_algorithm_implementation", {})
                
                # Key metrics
                key_metrics = ["exact_match", "codebleu", "pass_at_1", "syntax_valid", "security_score"]
                for metric in key_metrics:
                    if metric in task_results:
                        value = task_results[metric]
                        print(f"   {metric}: {value:.4f}")
                
                # Sample analysis
                samples = results.get("samples", {}).get("single_turn_scenarios_algorithm_implementation", [])
                if samples:
                    print(f"\n📝 Analyzed {len(samples)} samples:")
                    success_count = sum(1 for sample in samples if sample.get("exact_match", 0) > 0)
                    print(f"   Successful implementations: {success_count}/{len(samples)}")
            
            return True
        else:
            print("❌ Advanced evaluation failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Advanced evaluation timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Advanced evaluation failed: {e}")
        return False

def cleanup_results():
    """Clean up result files."""
    print("\n🧹 Cleaning up result files...")
    
    result_files = [
        "dashscope_basic_results.json",
        "dashscope_qwen-coder-plus_results.json",
        "dashscope_qwen-coder_results.json", 
        "dashscope_qwen-turbo_results.json",
        "dashscope_advanced_results.json"
    ]
    
    for file in result_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   Removed {file}")

def main():
    """Main function to run DashScope examples."""
    print("🌟 DashScope Single Turn Scenarios Example")
    print("=" * 50)
    
    # Check setup
    if not check_dashscope_setup():
        print("\n❌ DashScope setup incomplete. Please fix the issues above and try again.")
        return 1
    
    try:
        # Run basic evaluation
        if not run_basic_evaluation():
            print("\n❌ Basic evaluation failed. Check your setup and try again.")
            return 1
        
        # Run model comparison
        run_model_comparison()
        
        # Run advanced evaluation
        run_advanced_evaluation()
        
        print("\n🎉 All DashScope examples completed successfully!")
        print("\nNext steps:")
        print("- Try different Qwen models (qwen-max, qwen-plus)")
        print("- Experiment with different scenarios and difficulty levels")
        print("- Use metadata filtering for specific use cases")
        print("- Check the generated result files for detailed analysis")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    finally:
        # Ask user if they want to clean up
        try:
            response = input("\n🗑️  Clean up result files? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                cleanup_results()
        except (KeyboardInterrupt, EOFError):
            pass

if __name__ == "__main__":
    exit(main())