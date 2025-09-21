#!/usr/bin/env python3

import json
import os
import sys
import glob
from pathlib import Path

def load_latest_results(results_dir):
    """Load the most recent full context and no context results."""
    
    # Find latest files
    full_context_files = glob.glob(os.path.join(results_dir, "full_context_test*.json"))
    no_context_files = glob.glob(os.path.join(results_dir, "no_context_test*.json"))
    
    results = {}
    
    if full_context_files:
        latest_full = max(full_context_files, key=os.path.getmtime)
        with open(latest_full, 'r') as f:
            results['Full Context'] = json.load(f)
            
    if no_context_files:
        latest_no = max(no_context_files, key=os.path.getmtime)
        with open(latest_no, 'r') as f:
            results['No Context'] = json.load(f)
    
    return results

def extract_metrics(result_data):
    """Extract metrics from result data."""
    if 'results' not in result_data:
        return {}
    
    task_results = result_data['results'].get('multi_turn_coding_eval_claude_code', {})
    
    metrics = {}
    for key, value in task_results.items():
        if key.endswith('_stderr,extract_responses') or 'alias' in key:
            continue
        if isinstance(value, (int, float)):
            # Clean up metric names
            clean_name = key.replace(',extract_responses', '').replace('_', ' ').title()
            metrics[clean_name] = value
    
    return metrics

def main():
    if len(sys.argv) != 2:
        print("Usage: python quick_results.py <results_directory>")
        print("Example: python quick_results.py results/simple_test_easy")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("🧪 Multi-Turn Coding Evaluation Results")
    print("=" * 60)
    
    results = load_latest_results(results_dir)
    
    if not results:
        print("❌ No result files found")
        sys.exit(1)
    
    # Display results
    for scenario, data in results.items():
        print(f"\n📊 {scenario}")
        print("-" * 40)
        
        metrics = extract_metrics(data)
        
        if not metrics:
            print("   No metrics found")
            continue
            
        for metric, value in metrics.items():
            if value >= 0.9:
                status = "🟢"
            elif value >= 0.7:
                status = "🟡"
            elif value >= 0.5:
                status = "🟠"
            else:
                status = "🔴"
            
            print(f"   {status} {metric:<30} {value:.3f}")
    
    # Compare if both available
    if len(results) == 2:
        print(f"\n📈 Context Impact Analysis")
        print("-" * 40)
        
        full_metrics = extract_metrics(results['Full Context'])
        no_metrics = extract_metrics(results['No Context'])
        
        for metric in full_metrics:
            if metric in no_metrics:
                full_val = full_metrics[metric]
                no_val = no_metrics[metric]
                
                if no_val > 0:
                    improvement = ((full_val - no_val) / no_val) * 100
                    if abs(improvement) < 1:
                        status = "➡️"
                    elif improvement > 0:
                        status = "⬆️"
                    else:
                        status = "⬇️"
                    
                    print(f"   {status} {metric:<30} {improvement:+.1f}%")
                else:
                    print(f"   ➡️ {metric:<30} N/A (baseline 0)")
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results directory: {results_dir}")
    
    # Show generated artifacts
    output_dir = "./output"
    if os.path.exists(output_dir):
        dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if dirs:
            print(f"📂 Generated artifacts: {len(dirs)} problems")
            print(f"   Latest: {sorted(dirs)[-1] if dirs else 'None'}")

if __name__ == "__main__":
    main()