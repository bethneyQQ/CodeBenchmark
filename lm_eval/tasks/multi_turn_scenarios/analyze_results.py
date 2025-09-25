#!/usr/bin/env python3

import json
import os
import sys
import glob
from pathlib import Path
import argparse
from collections import defaultdict

def load_results(results_dir):
    """Load multi-turn scenarios results from directory."""
    result_files = glob.glob(os.path.join(results_dir, "*multi_turn_scenarios*.json"))
    
    if not result_files:
        return None
    
    latest_result = max(result_files, key=os.path.getmtime)
    with open(latest_result, 'r') as f:
        return json.load(f)

def extract_scenario_metrics(result_data):
    """Extract and organize metrics by scenario type."""
    if 'results' not in result_data:
        return {}
    
    scenario_metrics = defaultdict(dict)
    
    for task_name, metrics in result_data['results'].items():
        if 'multi_turn_scenarios' in task_name:
            # Extract scenario type from task name
            scenario_type = task_name.replace('multi_turn_scenarios.', '').replace('_', ' ').title()
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not key.endswith('_stderr'):
                    clean_name = key.replace('_', ' ').title()
                    scenario_metrics[scenario_type][clean_name] = value
    
    return dict(scenario_metrics)

def analyze_scenario_performance(scenario_metrics):
    """Analyze performance across different scenarios."""
    scenario_averages = {}
    
    for scenario, metrics in scenario_metrics.items():
        if metrics:
            avg_score = sum(metrics.values()) / len(metrics)
            scenario_averages[scenario] = avg_score
    
    return scenario_averages

def get_performance_status(value):
    """Get performance status emoji based on value."""
    if value >= 0.9:
        return "🟢"
    elif value >= 0.7:
        return "🟡"
    elif value >= 0.5:
        return "🟠"
    else:
        return "🔴"

def print_results(results_dir, result_data):
    """Print formatted analysis results."""
    print("🧪 Multi-Turn Scenarios Evaluation Results")
    print("=" * 60)
    print(f"📁 Results directory: {results_dir}")
    print()
    
    scenario_metrics = extract_scenario_metrics(result_data)
    
    if not scenario_metrics:
        print("❌ No scenario metrics found")
        return
    
    # Overall summary
    all_scores = []
    for metrics in scenario_metrics.values():
        all_scores.extend(metrics.values())
    
    if all_scores:
        overall_avg = sum(all_scores) / len(all_scores)
        print(f"📊 Overall Average Score: {overall_avg:.3f} {get_performance_status(overall_avg)}")
        print()
    
    # Scenario-specific results
    scenario_averages = analyze_scenario_performance(scenario_metrics)
    
    print("📋 Performance by Scenario Type")
    print("-" * 50)
    
    for scenario, metrics in scenario_metrics.items():
        scenario_avg = scenario_averages.get(scenario, 0)
        print(f"\n🎯 {scenario} (Avg: {scenario_avg:.3f} {get_performance_status(scenario_avg)})")
        print("   " + "-" * 45)
        
        for metric_name, value in metrics.items():
            status = get_performance_status(value)
            print(f"   {status} {metric_name:<30} {value:.3f}")
    
    # Scenario ranking
    if scenario_averages:
        print("\n🏆 Scenario Performance Ranking")
        print("-" * 40)
        
        sorted_scenarios = sorted(scenario_averages.items(), key=lambda x: x[1], reverse=True)
        for i, (scenario, avg) in enumerate(sorted_scenarios, 1):
            status = get_performance_status(avg)
            print(f"   {i}. {scenario:<25} {avg:.3f} {status}")
    
    # Performance insights
    print("\n💡 Performance Insights")
    print("-" * 40)
    
    if scenario_averages:
        best_scenario = max(scenario_averages.items(), key=lambda x: x[1])
        worst_scenario = min(scenario_averages.items(), key=lambda x: x[1])
        
        print(f"   🏆 Best Scenario: {best_scenario[0]} ({best_scenario[1]:.3f})")
        print(f"   ⚠️  Challenging Scenario: {worst_scenario[0]} ({worst_scenario[1]:.3f})")
        
        # Performance distribution
        excellent = sum(1 for avg in scenario_averages.values() if avg >= 0.9)
        good = sum(1 for avg in scenario_averages.values() if 0.7 <= avg < 0.9)
        fair = sum(1 for avg in scenario_averages.values() if 0.5 <= avg < 0.7)
        poor = sum(1 for avg in scenario_averages.values() if avg < 0.5)
        
        total = len(scenario_averages)
        print(f"\n   📈 Performance Distribution:")
        print(f"   🟢 Excellent (≥0.9): {excellent}/{total} ({excellent/total*100:.1f}%)")
        print(f"   🟡 Good (0.7-0.9): {good}/{total} ({good/total*100:.1f}%)")
        print(f"   🟠 Fair (0.5-0.7): {fair}/{total} ({fair/total*100:.1f}%)")
        print(f"   🔴 Poor (<0.5): {poor}/{total} ({poor/total*100:.1f}%)")

def print_scenario_recommendations(scenario_metrics):
    """Print recommendations based on scenario performance."""
    print("\n🎯 Recommendations")
    print("-" * 40)
    
    scenario_averages = analyze_scenario_performance(scenario_metrics)
    
    if not scenario_averages:
        return
    
    # Find scenarios that need improvement
    weak_scenarios = [scenario for scenario, avg in scenario_averages.items() if avg < 0.6]
    strong_scenarios = [scenario for scenario, avg in scenario_averages.items() if avg >= 0.8]
    
    if weak_scenarios:
        print("   ⚠️  Focus Areas for Improvement:")
        for scenario in weak_scenarios:
            avg = scenario_averages[scenario]
            print(f"   • {scenario}: {avg:.3f} - Consider additional training on this scenario type")
    
    if strong_scenarios:
        print("   ✅ Strong Performance Areas:")
        for scenario in strong_scenarios:
            avg = scenario_averages[scenario]
            print(f"   • {scenario}: {avg:.3f} - Model performs well in this scenario")
    
    # General recommendations
    overall_avg = sum(scenario_averages.values()) / len(scenario_averages)
    if overall_avg < 0.7:
        print("   💡 General Recommendations:")
        print("   • Consider fine-tuning on multi-turn conversation data")
        print("   • Review chat template configuration for better instruction following")
        print("   • Increase context window if conversations are being truncated")

def main():
    parser = argparse.ArgumentParser(description='Analyze multi-turn scenarios evaluation results')
    parser.add_argument('results_dir', help='Directory containing result files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed analysis')
    parser.add_argument('--recommendations', '-r', action='store_true', help='Show improvement recommendations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    result_data = load_results(args.results_dir)
    if not result_data:
        print(f"❌ No multi_turn_scenarios result files found in {args.results_dir}")
        print("Looking for files matching: *multi_turn_scenarios*.json")
        sys.exit(1)
    
    print_results(args.results_dir, result_data)
    
    if args.recommendations:
        scenario_metrics = extract_scenario_metrics(result_data)
        print_scenario_recommendations(scenario_metrics)
    
    if args.verbose:
        print("\n🔍 Raw Results Data")
        print("-" * 40)
        print(json.dumps(result_data.get('results', {}), indent=2))
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()