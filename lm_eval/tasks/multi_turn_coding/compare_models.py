#!/usr/bin/env python3

import json
import os
import sys
import glob
import argparse
from pathlib import Path
from collections import defaultdict

def load_results(file_path):
    """Load evaluation results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def extract_metrics(result_data, model_name="Unknown"):
    """Extract metrics from result data."""
    if not result_data or 'results' not in result_data:
        return {}
    
    metrics = {}
    for task_name, task_results in result_data['results'].items():
        for key, value in task_results.items():
            if key.endswith('_stderr') or 'alias' in key:
                continue
            if isinstance(value, (int, float)):
                # Clean up metric names
                clean_name = key.replace(',extract_responses', '').replace('_', ' ').title()
                metrics[clean_name] = value
    
    return metrics

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

def compare_models(result_files):
    """Compare multiple model results."""
    model_results = {}
    
    for file_path in result_files:
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue
            
        # Extract model name from filename
        filename = os.path.basename(file_path)
        if 'claude' in filename.lower():
            model_name = "Claude"
        elif 'deepseek' in filename.lower():
            model_name = "DeepSeek"
        elif 'openai' in filename.lower() or 'gpt' in filename.lower():
            model_name = "OpenAI"
        elif 'qwen' in filename.lower() or 'dashscope' in filename.lower():
            model_name = "Qwen/DashScope"
        elif 'anthropic' in filename.lower():
            model_name = "Anthropic"
        else:
            model_name = filename.replace('.json', '').replace('_comparison', '').title()
        
        result_data = load_results(file_path)
        if result_data:
            metrics = extract_metrics(result_data, model_name)
            if metrics:
                model_results[model_name] = metrics
                print(f"✅ Loaded results for {model_name}: {len(metrics)} metrics")
            else:
                print(f"⚠️  No metrics found in {file_path}")
        else:
            print(f"❌ Failed to load {file_path}")
    
    return model_results

def print_comparison_table(model_results):
    """Print a comparison table of all models."""
    if not model_results:
        print("❌ No model results to compare")
        return
    
    print("\n🏆 Multi-Model Performance Comparison")
    print("=" * 80)
    
    # Get all unique metrics
    all_metrics = set()
    for metrics in model_results.values():
        all_metrics.update(metrics.keys())
    
    all_metrics = sorted(all_metrics)
    
    # Print header
    print(f"{'Metric':<35}", end="")
    for model_name in model_results.keys():
        print(f"{model_name:<12}", end="")
    print()
    print("-" * 80)
    
    # Print metrics
    for metric in all_metrics:
        print(f"{metric:<35}", end="")
        for model_name, metrics in model_results.items():
            if metric in metrics:
                value = metrics[metric]
                status = get_performance_status(value)
                print(f"{status} {value:<8.3f}", end="")
            else:
                print(f"{'N/A':<10}", end="")
        print()

def print_model_rankings(model_results):
    """Print model rankings by average performance."""
    if not model_results:
        return
    
    print("\n📊 Model Rankings (by Average Performance)")
    print("-" * 50)
    
    model_averages = {}
    for model_name, metrics in model_results.items():
        if metrics:
            avg_score = sum(metrics.values()) / len(metrics)
            model_averages[model_name] = avg_score
    
    # Sort by average score
    sorted_models = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model_name, avg_score) in enumerate(sorted_models, 1):
        status = get_performance_status(avg_score)
        print(f"{i}. {model_name:<15} {avg_score:.3f} {status}")

def print_metric_analysis(model_results):
    """Print analysis of which model performs best on each metric."""
    if not model_results:
        return
    
    print("\n🎯 Best Performance by Metric")
    print("-" * 50)
    
    # Get all unique metrics
    all_metrics = set()
    for metrics in model_results.values():
        all_metrics.update(metrics.keys())
    
    for metric in sorted(all_metrics):
        best_model = None
        best_score = -1
        
        for model_name, metrics in model_results.items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name
        
        if best_model:
            status = get_performance_status(best_score)
            print(f"{metric:<35} {best_model:<12} {best_score:.3f} {status}")

def print_summary_insights(model_results):
    """Print summary insights and recommendations."""
    if not model_results:
        return
    
    print("\n💡 Summary Insights")
    print("-" * 40)
    
    # Calculate overall statistics
    model_averages = {}
    for model_name, metrics in model_results.items():
        if metrics:
            avg_score = sum(metrics.values()) / len(metrics)
            model_averages[model_name] = avg_score
    
    if model_averages:
        best_overall = max(model_averages.items(), key=lambda x: x[1])
        worst_overall = min(model_averages.items(), key=lambda x: x[1])
        
        print(f"🏆 Best Overall Performance: {best_overall[0]} ({best_overall[1]:.3f})")
        print(f"⚠️  Needs Improvement: {worst_overall[0]} ({worst_overall[1]:.3f})")
        
        # Performance distribution
        excellent = sum(1 for avg in model_averages.values() if avg >= 0.8)
        good = sum(1 for avg in model_averages.values() if 0.6 <= avg < 0.8)
        fair = sum(1 for avg in model_averages.values() if 0.4 <= avg < 0.6)
        poor = sum(1 for avg in model_averages.values() if avg < 0.4)
        
        total = len(model_averages)
        print(f"\n📈 Performance Distribution:")
        print(f"🟢 Excellent (≥0.8): {excellent}/{total}")
        print(f"🟡 Good (0.6-0.8): {good}/{total}")
        print(f"🟠 Fair (0.4-0.6): {fair}/{total}")
        print(f"🔴 Poor (<0.4): {poor}/{total}")

def main():
    parser = argparse.ArgumentParser(description='Compare multiple model evaluation results')
    parser.add_argument('result_files', nargs='+', help='JSON result files to compare')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    print("🔍 Multi-Model Comparison Analysis")
    print("=" * 50)
    
    # Load and compare results
    model_results = compare_models(args.result_files)
    
    if not model_results:
        print("❌ No valid model results found to compare")
        sys.exit(1)
    
    print(f"\n📋 Comparing {len(model_results)} models")
    
    # Print comparison table
    print_comparison_table(model_results)
    
    # Print rankings
    print_model_rankings(model_results)
    
    # Print metric analysis
    print_metric_analysis(model_results)
    
    # Print summary insights
    print_summary_insights(model_results)
    
    if args.verbose:
        print("\n🔍 Detailed Model Results")
        print("-" * 40)
        for model_name, metrics in model_results.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                status = get_performance_status(value)
                print(f"  {status} {metric:<30} {value:.3f}")
    
    print("\n✅ Comparison analysis complete!")

if __name__ == "__main__":
    main()