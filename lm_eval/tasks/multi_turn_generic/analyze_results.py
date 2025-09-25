#!/usr/bin/env python3

import json
import os
import sys
import glob
from pathlib import Path
import argparse

def load_results(results_dir):
    """Load multi-turn generic results from directory."""
    result_files = glob.glob(os.path.join(results_dir, "*multi_turn_generic*.json"))
    
    if not result_files:
        return None
    
    latest_result = max(result_files, key=os.path.getmtime)
    with open(latest_result, 'r') as f:
        return json.load(f)

def extract_metrics(result_data):
    """Extract and organize metrics from result data."""
    if 'results' not in result_data:
        return {}
    
    # Look for multi_turn_generic task results
    task_results = None
    for task_name, metrics in result_data['results'].items():
        if 'multi_turn_generic' in task_name:
            task_results = metrics
            break
    
    if not task_results:
        return {}
    
    organized_metrics = {}
    for key, value in task_results.items():
        if isinstance(value, (int, float)) and not key.endswith('_stderr'):
            # Clean up metric names
            clean_name = key.replace('_', ' ').title()
            organized_metrics[clean_name] = value
    
    return organized_metrics

def analyze_phase_performance(result_data):
    """Analyze performance across different phases."""
    metrics = extract_metrics(result_data)
    
    phase_metrics = {
        'Analysis Phase': [],
        'Design Phase': [],
        'Implementation Phase': [],
        'Overall': []
    }
    
    for metric_name, value in metrics.items():
        if 'analysis' in metric_name.lower():
            phase_metrics['Analysis Phase'].append((metric_name, value))
        elif 'design' in metric_name.lower():
            phase_metrics['Design Phase'].append((metric_name, value))
        elif 'implementation' in metric_name.lower() or 'code' in metric_name.lower():
            phase_metrics['Implementation Phase'].append((metric_name, value))
        else:
            phase_metrics['Overall'].append((metric_name, value))
    
    return phase_metrics

def print_results(results_dir, result_data):
    """Print formatted analysis results."""
    print("🧪 Multi-Turn Generic Evaluation Results")
    print("=" * 60)
    print(f"📁 Results directory: {results_dir}")
    print()
    
    # Overall metrics
    metrics = extract_metrics(result_data)
    if metrics:
        print("📊 Overall Performance")
        print("-" * 40)
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
        print()
    
    # Phase-specific analysis
    phase_metrics = analyze_phase_performance(result_data)
    
    for phase_name, phase_data in phase_metrics.items():
        if phase_data:
            print(f"📋 {phase_name}")
            print("-" * 40)
            for metric_name, value in phase_data:
                if value >= 0.8:
                    status = "🟢"
                elif value >= 0.6:
                    status = "🟡"
                elif value >= 0.4:
                    status = "🟠"
                else:
                    status = "🔴"
                
                print(f"   {status} {metric_name:<25} {value:.3f}")
            print()
    
    # Performance insights
    print("💡 Performance Insights")
    print("-" * 40)
    
    if metrics:
        avg_score = sum(metrics.values()) / len(metrics)
        print(f"   📈 Average Score: {avg_score:.3f}")
        
        best_metric = max(metrics.items(), key=lambda x: x[1])
        worst_metric = min(metrics.items(), key=lambda x: x[1])
        
        print(f"   🏆 Best Performance: {best_metric[0]} ({best_metric[1]:.3f})")
        print(f"   ⚠️  Needs Improvement: {worst_metric[0]} ({worst_metric[1]:.3f})")
        
        # Phase analysis
        phase_averages = {}
        for phase_name, phase_data in phase_metrics.items():
            if phase_data:
                phase_avg = sum(value for _, value in phase_data) / len(phase_data)
                phase_averages[phase_name] = phase_avg
        
        if phase_averages:
            print()
            print("   📊 Phase Performance Ranking:")
            sorted_phases = sorted(phase_averages.items(), key=lambda x: x[1], reverse=True)
            for i, (phase, avg) in enumerate(sorted_phases, 1):
                print(f"   {i}. {phase}: {avg:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze multi-turn generic evaluation results')
    parser.add_argument('results_dir', help='Directory containing result files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    result_data = load_results(args.results_dir)
    if not result_data:
        print(f"❌ No multi_turn_generic result files found in {args.results_dir}")
        print("Looking for files matching: *multi_turn_generic*.json")
        sys.exit(1)
    
    print_results(args.results_dir, result_data)
    
    if args.verbose:
        print("\n🔍 Raw Results Data")
        print("-" * 40)
        print(json.dumps(result_data.get('results', {}), indent=2))
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()