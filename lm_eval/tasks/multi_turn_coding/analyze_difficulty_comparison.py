#!/usr/bin/env python3

"""
Analyze results from run_difficulty_comparison.sh
Provides quick analysis and visualization of context impact across difficulty levels.
"""

import json
import os
import sys
from pathlib import Path

def load_comparison_results(base_dir="results/difficulty_comparison"):
    """Load results from difficulty comparison evaluation."""
    
    difficulties = ["easy", "simple", "medium", "complex"]
    contexts = ["full_context", "no_context"]
    
    results = {}
    
    print(f"📊 Loading results from {base_dir}")
    print("=" * 50)
    
    for difficulty in difficulties:
        difficulty_results = {}
        
        for context in contexts:
            file_path = os.path.join(base_dir, difficulty, f"{context}_results.json")
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract metrics
                    task_results = data.get('results', {}).get('multi_turn_coding_eval_claude_code', {})
                    metrics = {}
                    
                    for metric, value in task_results.items():
                        if isinstance(value, (int, float)) and 'stderr' not in metric:
                            clean_metric = metric.replace(',extract_responses', '').replace('_', ' ').title()
                            metrics[clean_metric] = value
                    
                    difficulty_results[context] = metrics
                    print(f"✅ {difficulty.capitalize()} - {context.replace('_', ' ').title()}: {len(metrics)} metrics")
                    
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
            else:
                print(f"⚠️  Missing: {file_path}")
        
        if difficulty_results:
            results[difficulty] = difficulty_results
    
    return results


def calculate_improvements(results):
    """Calculate percentage improvements from no context to full context."""
    
    improvements = {}
    
    for difficulty, contexts in results.items():
        if 'full_context' in contexts and 'no_context' in contexts:
            full_metrics = contexts['full_context']
            no_metrics = contexts['no_context']
            
            difficulty_improvements = {}
            
            for metric in full_metrics:
                if metric in no_metrics:
                    baseline = no_metrics[metric]
                    full_value = full_metrics[metric]
                    
                    if baseline > 0:
                        improvement = ((full_value - baseline) / baseline) * 100
                    else:
                        improvement = 0 if full_value == 0 else float('inf')
                    
                    difficulty_improvements[metric] = {
                        'improvement_pct': improvement,
                        'baseline': baseline,
                        'full_context': full_value
                    }
            
            improvements[difficulty] = difficulty_improvements
    
    return improvements


def print_summary_table(results):
    """Print a summary table of all results."""
    
    print(f"\n📋 Results Summary")
    print("=" * 80)
    
    # Get all metrics
    all_metrics = set()
    for difficulty_data in results.values():
        for context_data in difficulty_data.values():
            all_metrics.update(context_data.keys())
    
    all_metrics = sorted(all_metrics)
    
    # Print header
    print(f"{'Difficulty':<12} {'Context':<12}", end="")
    for metric in all_metrics:
        print(f"{metric[:12]:<14}", end="")
    print()
    
    print("-" * (24 + len(all_metrics) * 14))
    
    # Print data
    for difficulty in ["easy", "simple", "medium", "complex"]:
        if difficulty in results:
            for context in ["full_context", "no_context"]:
                if context in results[difficulty]:
                    context_display = context.replace('_', ' ').title()
                    print(f"{difficulty.capitalize():<12} {context_display:<12}", end="")
                    
                    for metric in all_metrics:
                        value = results[difficulty][context].get(metric, 0)
                        print(f"{value:<14.3f}", end="")
                    print()
            print()


def print_improvement_analysis(improvements):
    """Print improvement analysis."""
    
    print(f"\n📈 Context Impact Analysis")
    print("=" * 60)
    
    for difficulty in ["easy", "simple", "medium", "complex"]:
        if difficulty in improvements:
            print(f"\n🎯 {difficulty.capitalize()} Difficulty:")
            print("-" * 30)
            
            difficulty_improvements = improvements[difficulty]
            
            # Sort by improvement percentage
            sorted_metrics = sorted(difficulty_improvements.items(), 
                                  key=lambda x: x[1]['improvement_pct'], reverse=True)
            
            for metric, data in sorted_metrics:
                improvement = data['improvement_pct']
                baseline = data['baseline']
                full_value = data['full_context']
                
                if improvement == float('inf'):
                    status = "🚀"
                    improvement_str = "∞% (from 0)"
                elif improvement > 10:
                    status = "🟢"
                    improvement_str = f"+{improvement:.1f}%"
                elif improvement > 0:
                    status = "🟡"
                    improvement_str = f"+{improvement:.1f}%"
                elif improvement == 0:
                    status = "➡️"
                    improvement_str = "0.0%"
                else:
                    status = "🔴"
                    improvement_str = f"{improvement:.1f}%"
                
                print(f"  {status} {metric:<25} {improvement_str:<10} ({full_value:.3f} vs {baseline:.3f})")


def generate_quick_insights(results, improvements):
    """Generate quick insights from the analysis."""
    
    print(f"\n💡 Quick Insights")
    print("=" * 40)
    
    # Find difficulty with highest average improvement
    avg_improvements = {}
    for difficulty, metrics in improvements.items():
        if metrics:
            valid_improvements = [data['improvement_pct'] for data in metrics.values() 
                                if data['improvement_pct'] != float('inf')]
            if valid_improvements:
                avg_improvements[difficulty] = sum(valid_improvements) / len(valid_improvements)
    
    if avg_improvements:
        best_difficulty = max(avg_improvements, key=avg_improvements.get)
        worst_difficulty = min(avg_improvements, key=avg_improvements.get)
        
        print(f"🏆 Highest context benefit: {best_difficulty.capitalize()} ({avg_improvements[best_difficulty]:.1f}% avg improvement)")
        print(f"📊 Lowest context benefit: {worst_difficulty.capitalize()} ({avg_improvements[worst_difficulty]:.1f}% avg improvement)")
    
    # Find most context-sensitive metric
    metric_improvements = {}
    for difficulty_data in improvements.values():
        for metric, data in difficulty_data.items():
            if data['improvement_pct'] != float('inf'):
                if metric not in metric_improvements:
                    metric_improvements[metric] = []
                metric_improvements[metric].append(data['improvement_pct'])
    
    if metric_improvements:
        avg_metric_improvements = {metric: sum(values) / len(values) 
                                 for metric, values in metric_improvements.items()}
        
        most_sensitive = max(avg_metric_improvements, key=avg_metric_improvements.get)
        least_sensitive = min(avg_metric_improvements, key=avg_metric_improvements.get)
        
        print(f"🎯 Most context-sensitive metric: {most_sensitive} ({avg_metric_improvements[most_sensitive]:.1f}% avg)")
        print(f"📈 Least context-sensitive metric: {least_sensitive} ({avg_metric_improvements[least_sensitive]:.1f}% avg)")
    
    # Count perfect scores
    perfect_scores = {}
    for difficulty, contexts in results.items():
        perfect_count = 0
        total_metrics = 0
        
        for context, metrics in contexts.items():
            for metric, value in metrics.items():
                total_metrics += 1
                if value >= 0.99:  # Consider >= 0.99 as perfect
                    perfect_count += 1
        
        if total_metrics > 0:
            perfect_scores[difficulty] = (perfect_count / total_metrics) * 100
    
    if perfect_scores:
        print(f"\n🎯 Perfect Score Rates (≥0.99):")
        for difficulty in ["easy", "simple", "medium", "complex"]:
            if difficulty in perfect_scores:
                print(f"  {difficulty.capitalize()}: {perfect_scores[difficulty]:.1f}%")


def main():
    """Main analysis function."""
    
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("🎯 Difficulty Comparison Analysis")
        print("=" * 50)
        print("Usage: python analyze_difficulty_comparison.py [results_directory]")
        print("")
        print("Arguments:")
        print("  results_directory    Directory containing difficulty comparison results")
        print("                      (default: results/difficulty_comparison)")
        print("")
        print("Examples:")
        print("  python analyze_difficulty_comparison.py")
        print("  python analyze_difficulty_comparison.py results/my_comparison")
        print("")
        print("This script analyzes results from run_difficulty_comparison.sh")
        print("and provides insights into context impact across difficulty levels.")
        return 0
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "results/difficulty_comparison"
    
    print("🎯 Difficulty Comparison Analysis")
    print("=" * 50)
    print(f"Analyzing results from: {base_dir}")
    
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        print("Run the evaluation first:")
        print("./run_difficulty_comparison.sh")
        return 1
    
    # Load results
    results = load_comparison_results(base_dir)
    
    if not results:
        print("❌ No results found to analyze")
        return 1
    
    # Calculate improvements
    improvements = calculate_improvements(results)
    
    # Print analysis
    print_summary_table(results)
    print_improvement_analysis(improvements)
    generate_quick_insights(results, improvements)
    
    # Suggest next steps
    print(f"\n🚀 Next Steps:")
    print(f"1. Generate visualizations:")
    print(f"   python analyze_context_impact.py --difficulty_dirs {base_dir}/easy {base_dir}/simple {base_dir}/medium {base_dir}/complex")
    print(f"2. View individual difficulty results:")
    for difficulty in ["easy", "simple", "medium", "complex"]:
        if difficulty in results:
            print(f"   python quick_results.py {base_dir}/{difficulty}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())