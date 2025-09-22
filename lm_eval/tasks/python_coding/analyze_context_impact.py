#!/usr/bin/env python3
"""
Context Impact Analysis for Python Coding Evaluation Tasks

This script analyzes the results from run_context_comparison.sh to compare
Full Context vs No Context performance on Python coding tasks.

Usage:
    python analyze_context_impact.py [--results_dir results/context_comparison]
"""

import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import glob
import sys


def find_result_files(results_dir):
    """Find the latest result files with timestamped names."""
    result_files = {}
    
    # Task categories to look for
    task_categories = [
        'code_completion',
        'code_repair', 
        'function_generation',
        'docstring_generation',
        'code_translation'
    ]
    
    # Context modes to look for
    context_modes = ['full_context', 'no_context', 'minimal_context']
    
    for category in task_categories:
        for context_mode in context_modes:
            pattern = os.path.join(results_dir, f"{category}_{context_mode}*.json")
            files = glob.glob(pattern)
            if files:
                result_files[f'{category}_{context_mode}'] = max(files, key=os.path.getmtime)
    
    return result_files


def load_results(results_file):
    """Load evaluation results from JSON file."""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', {})
        
        # Clean up metric names by removing filter suffixes like ",extract_code"
        cleaned_results = {}
        for task_name, task_results in results.items():
            cleaned_task_results = {}
            for metric_name, value in task_results.items():
                # Remove filter suffixes from metric names
                if ',' in metric_name and not metric_name.endswith('_stderr'):
                    clean_metric_name = metric_name.split(',')[0]
                    cleaned_task_results[clean_metric_name] = value
                elif not metric_name.endswith('_stderr') and ',' not in metric_name:
                    # Keep metrics without suffixes as-is (except stderr)
                    cleaned_task_results[metric_name] = value
                # Keep stderr metrics as-is for reference
                elif metric_name.endswith('_stderr'):
                    cleaned_task_results[metric_name] = value
            
            cleaned_results[task_name] = cleaned_task_results
        
        return cleaned_results
    except Exception as e:
        print(f"Error loading results from {results_file}: {e}")
        return {}


def analyze_context_impact(results_dir):
    """Analyze context impact from existing result files."""
    
    print(f"🔍 Looking for result files in: {results_dir}")
    
    # Find result files
    result_files = find_result_files(results_dir)
    
    if not result_files:
        print("❌ No result files found!")
        print(f"   Expected files matching patterns:")
        print(f"   - {results_dir}/<category>_full_context*.json")
        print(f"   - {results_dir}/<category>_no_context*.json")
        print(f"   - {results_dir}/<category>_minimal_context*.json")
        print(f"   Where <category> is one of: code_completion, code_repair, function_generation, docstring_generation, code_translation")
        print(f"   Please run ./run_context_comparison.sh first")
        return False
    
    print(f"✅ Found {len(result_files)} result files:")
    for context_type, file_path in result_files.items():
        print(f"   {context_type}: {os.path.basename(file_path)}")
    
    # Load results
    all_results = {}
    
    for context_type, file_path in result_files.items():
        print(f"\n📊 Loading {context_type} results...")
        results = load_results(file_path)
        if results:
            all_results[context_type] = results
            print(f"   ✅ Loaded results for {len(results)} tasks")
        else:
            print(f"   ❌ Failed to load results from {file_path}")
    
    if not all_results:
        print("❌ No valid results loaded!")
        return False
    
    # Create output directory for analysis
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate analysis report
    generate_context_report(all_results, result_files, analysis_dir)
    
    # Generate visualizations
    generate_context_visualizations(all_results, analysis_dir)
    
    print(f"\n🎉 Context impact analysis complete!")
    print(f"📁 Analysis saved to: {analysis_dir}")
    print(f"📊 Report: {analysis_dir}/context_impact_report.md")
    print(f"📈 Charts: {analysis_dir}/context_impact_charts.png")
    
    return True


def generate_context_report(results, result_files, output_dir):
    """Generate comprehensive context impact analysis report."""
    
    report_path = os.path.join(output_dir, "context_impact_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Python Coding Evaluation - Context Impact Analysis\n\n")
        f.write("## Table of Contents\n\n")
        f.write("1. [Summary](#summary)\n")
        f.write("2. [Data Sources](#data-sources)\n")
        f.write("3. [Context Configurations](#context-configurations)\n") 
        f.write("4. [Results Comparison](#results-comparison)\n")
        f.write("5. [Context Impact Analysis](#context-impact-analysis)\n")
        f.write("6. [Visualizations](#visualizations)\n")
        f.write("7. [Key Insights](#key-insights)\n")
        f.write("8. [Recommendations](#recommendations)\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report analyzes how company-specific context affects model performance ")
        f.write("on Python code completion tasks. We compare **Full Context** (detailed company ")
        f.write("policies and standards) vs **No Context** (generic prompts).\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Analysis Directory:** `{output_dir}`\n\n")
        
        f.write("## Data Sources\n\n")
        f.write("Analysis based on the following result files:\n\n")
        for context_type, file_path in result_files.items():
            f.write(f"- **{context_type.replace('_', ' ').title()}**: `{os.path.basename(file_path)}`\n")
        f.write("\n")
        
        f.write("## Context Configurations\n\n")
        f.write("### Full Context\n")
        f.write("- **Style Guide**: Google Python Style Guide with specific formatting requirements\n")
        f.write("- **Security Policy**: Enterprise security requirements and validation rules\n")
        f.write("- **Performance Requirements**: Specific performance targets and optimization guidelines\n")
        f.write("- **Architecture Standards**: Enterprise architecture patterns and best practices\n")
        f.write("- **Documentation Standards**: Comprehensive documentation requirements\n\n")
        
        f.write("### No Context\n")
        f.write("- Generic prompts without specific requirements\n")
        f.write("- Tests baseline model capabilities\n")
        f.write("- Represents typical coding scenarios without company-specific guidance\n\n")
        
        # Add detailed results analysis
        f.write("## Results Comparison\n\n")
        
        # Create comparison table for all tasks
        f.write("### Task Performance Overview\n\n")
        
        # Group results by task category
        task_results = {}
        for result_key, result_data in results.items():
            # Extract task name and context mode
            for context_mode in ['_full_context', '_no_context', '_minimal_context']:
                if context_mode in result_key:
                    task_name = result_key.replace(context_mode, '')
                    context_key = context_mode[1:]  # Remove leading underscore
                    if task_name not in task_results:
                        task_results[task_name] = {}
                    task_results[task_name][context_key] = result_data
                    break
        
        # Create comprehensive comparison table
        f.write("| Task Category | Metric | Full Context | No Context | Minimal Context | Full vs No | Full vs Minimal |\n")
        f.write("|---------------|--------|--------------|------------|-----------------|------------|------------------|\n")
        
        for task_name, task_data in task_results.items():
            if 'full_context' in task_data:
                full_results = task_data['full_context']
                full_task_key = list(full_results.keys())[0] if full_results else None
                
                if full_task_key:
                    full_metrics = full_results[full_task_key]
                    
                    # Show key metrics for each task
                    key_metrics = ['exact_match', 'context_adherence_score', 'style_compliance_score']
                    
                    for metric in key_metrics:
                        if metric in full_metrics:
                            full_val = full_metrics[metric]
                            
                            # Get no context value
                            no_val = "N/A"
                            no_improvement = "N/A"
                            if 'no_context' in task_data:
                                no_results = task_data['no_context']
                                no_task_key = list(no_results.keys())[0] if no_results else None
                                if no_task_key and metric in no_results[no_task_key]:
                                    no_val = no_results[no_task_key][metric]
                                    if isinstance(full_val, (int, float)) and isinstance(no_val, (int, float)) and no_val > 0:
                                        improvement = ((full_val - no_val) / no_val) * 100
                                        no_improvement = f"{improvement:+.1f}%"
                                        no_val = f"{no_val:.3f}"
                            
                            # Get minimal context value
                            minimal_val = "N/A"
                            minimal_improvement = "N/A"
                            if 'minimal_context' in task_data:
                                minimal_results = task_data['minimal_context']
                                minimal_task_key = list(minimal_results.keys())[0] if minimal_results else None
                                if minimal_task_key and metric in minimal_results[minimal_task_key]:
                                    minimal_val_raw = minimal_results[minimal_task_key][metric]
                                    if isinstance(full_val, (int, float)) and isinstance(minimal_val_raw, (int, float)) and minimal_val_raw > 0:
                                        improvement = ((full_val - minimal_val_raw) / minimal_val_raw) * 100
                                        minimal_improvement = f"{improvement:+.1f}%"
                                        minimal_val = f"{minimal_val_raw:.3f}"
                            
                            f.write(f"| {task_name.replace('_', ' ').title()} | {metric.replace('_', ' ').title()} | {full_val:.3f} | {no_val} | {minimal_val} | {no_improvement} | {minimal_improvement} |\n")
        
        f.write("\n")
        
        f.write("## Context Impact Analysis\n\n")
        
        # Analyze impact across all task categories
        f.write("### Performance Impact by Task Category\n\n")
        
        for task_name, task_data in task_results.items():
            f.write(f"#### {task_name.replace('_', ' ').title()}\n\n")
            
            if 'full_context' in task_data:
                full_results = task_data['full_context']
                full_task_key = list(full_results.keys())[0] if full_results else None
                
                if full_task_key:
                    full_metrics = full_results[full_task_key]
                    
                    # Compare with no context
                    if 'no_context' in task_data:
                        no_results = task_data['no_context']
                        no_task_key = list(no_results.keys())[0] if no_results else None
                        
                        if no_task_key:
                            no_metrics = no_results[no_task_key]
                            
                            # Context Adherence Analysis
                            full_context_score = full_metrics.get('context_adherence_score', 0)
                            no_context_score = no_metrics.get('context_adherence_score', 0)
                            
                            if isinstance(full_context_score, (int, float)) and isinstance(no_context_score, (int, float)):
                                f.write(f"**Context Adherence:**\n")
                                f.write(f"- Full Context: {full_context_score:.3f}\n")
                                f.write(f"- No Context: {no_context_score:.3f}\n")
                                if no_context_score > 0:
                                    improvement = ((full_context_score - no_context_score) / no_context_score) * 100
                                    f.write(f"- **Improvement: {improvement:+.1f}%**\n\n")
                                else:
                                    f.write(f"- **Improvement: N/A (baseline is 0)**\n\n")
                            
                            # Style Compliance Analysis
                            full_style_score = full_metrics.get('style_compliance_score', 0)
                            no_style_score = no_metrics.get('style_compliance_score', 0)
                            
                            if isinstance(full_style_score, (int, float)) and isinstance(no_style_score, (int, float)):
                                f.write(f"**Style Compliance:**\n")
                                f.write(f"- Full Context: {full_style_score:.3f}\n")
                                f.write(f"- No Context: {no_style_score:.3f}\n")
                                if no_style_score > 0:
                                    improvement = ((full_style_score - no_style_score) / no_style_score) * 100
                                    f.write(f"- **Improvement: {improvement:+.1f}%**\n\n")
                                else:
                                    f.write(f"- **Improvement: N/A (baseline is 0)**\n\n")
                            
                            # BLEU Score Analysis (if available)
                            full_bleu = full_metrics.get('bleu', 0)
                            no_bleu = no_metrics.get('bleu', 0)
                            
                            if isinstance(full_bleu, (int, float)) and isinstance(no_bleu, (int, float)) and full_bleu > 0 and no_bleu > 0:
                                f.write(f"**BLEU Score:**\n")
                                f.write(f"- Full Context: {full_bleu:.3f}\n")
                                f.write(f"- No Context: {no_bleu:.3f}\n")
                                improvement = ((full_bleu - no_bleu) / no_bleu) * 100
                                f.write(f"- **Improvement: {improvement:+.1f}%**\n\n")
                            
                            # Determine overall impact
                            context_improvement = 0
                            style_improvement = 0
                            
                            if no_context_score > 0:
                                context_improvement = ((full_context_score - no_context_score) / no_context_score) * 100
                            if no_style_score > 0:
                                style_improvement = ((full_style_score - no_style_score) / no_style_score) * 100
                            
                            avg_improvement = (context_improvement + style_improvement) / 2
                            
                            if avg_improvement > 20:
                                f.write("🟢 **High positive impact** - Context significantly improves performance\n\n")
                            elif avg_improvement > 5:
                                f.write("🟡 **Moderate positive impact** - Context provides noticeable improvement\n\n")
                            elif avg_improvement > -5:
                                f.write("🟡 **Minimal impact** - Context has little effect on performance\n\n")
                            else:
                                f.write("🔴 **Negative impact** - Context may be hindering performance\n\n")
        
        # Overall analysis
        f.write("### Overall Context Impact Summary\n\n")
        
        # Calculate average improvements across all tasks
        total_improvements = []
        task_count = 0
        
        for task_name, task_data in task_results.items():
            if 'full_context' in task_data and 'no_context' in task_data:
                full_results = task_data['full_context']
                no_results = task_data['no_context']
                
                full_task_key = list(full_results.keys())[0] if full_results else None
                no_task_key = list(no_results.keys())[0] if no_results else None
                
                if full_task_key and no_task_key:
                    full_metrics = full_results[full_task_key]
                    no_metrics = no_results[no_task_key]
                    
                    full_exact = full_metrics.get('exact_match', 0)
                    no_exact = no_metrics.get('exact_match', 0)
                    
                    if isinstance(full_exact, (int, float)) and isinstance(no_exact, (int, float)) and no_exact > 0:
                        improvement = ((full_exact - no_exact) / no_exact) * 100
                        total_improvements.append(improvement)
                        task_count += 1
        
        if total_improvements:
            avg_improvement = sum(total_improvements) / len(total_improvements)
            f.write(f"**Average Improvement Across {task_count} Tasks:** {avg_improvement:+.1f}%\n\n")
            
            if avg_improvement > 10:
                f.write("🟢 **Overall Assessment: Significant positive impact** - Context consistently improves performance\n\n")
            elif avg_improvement > 0:
                f.write("🟡 **Overall Assessment: Moderate positive impact** - Context generally helps performance\n\n")
            elif avg_improvement > -5:
                f.write("🟡 **Overall Assessment: Minimal impact** - Context has little overall effect\n\n")
            else:
                f.write("🔴 **Overall Assessment: Negative impact** - Context may be hindering performance\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("The following charts provide comprehensive visual representations of the context impact analysis:\n\n")
        
        f.write("### Core Metrics Comparison\n")
        f.write("![Core Metrics](context_impact_core_metrics.png)\n\n")
        f.write("*Figure 1: Context Adherence and Style Compliance comparison across all tasks*\n\n")
        
        f.write("### BLEU Scores and Improvements\n")
        f.write("![BLEU and Improvements](context_impact_bleu_improvements.png)\n\n")
        f.write("*Figure 2: BLEU score comparison and context adherence improvement percentages*\n\n")
        
        f.write("### Impact Heatmap\n")
        f.write("![Impact Heatmap](context_impact_heatmap.png)\n\n")
        f.write("*Figure 3: Comprehensive heatmap showing percentage improvements across all metrics and tasks*\n\n")
        
        f.write("### Summary Dashboard\n")
        f.write("![Summary Dashboard](context_impact_dashboard.png)\n\n")
        f.write("*Figure 4: Overall summary dashboard with average improvements and impact distribution*\n\n")
        
        f.write("### Chart Descriptions\n\n")
        f.write("**Core Metrics Comparison:**\n")
        f.write("- Side-by-side comparison of Context Adherence and Style Compliance scores\n")
        f.write("- Shows absolute performance values for each task category\n")
        f.write("- Enables direct comparison between Full Context and No Context scenarios\n\n")
        
        f.write("**BLEU Scores and Improvements:**\n")
        f.write("- Left panel: BLEU score comparison showing semantic similarity improvements\n")
        f.write("- Right panel: Context adherence improvement percentages\n")
        f.write("- Green bars indicate positive impact, red bars indicate negative impact\n\n")
        
        f.write("**Impact Heatmap:**\n")
        f.write("- Color-coded matrix showing percentage improvements across all metrics\n")
        f.write("- Green indicates positive impact, red indicates negative impact\n")
        f.write("- Provides quick overview of which tasks benefit most from context\n\n")
        
        f.write("**Summary Dashboard:**\n")
        f.write("- Top Left: Average improvement by metric type\n")
        f.write("- Top Right: Average performance comparison by task\n")
        f.write("- Bottom Left: Security and performance specific metrics\n")
        f.write("- Bottom Right: Overall impact distribution pie chart\n\n")
        
        f.write("## Key Insights\n\n")
        f.write("1. **Context Sensitivity**: Different task categories show varying sensitivity to context\n")
        f.write("2. **Performance Patterns**: Identify which contexts provide the most benefit\n")
        f.write("3. **Task-Specific Impact**: Some tasks benefit more from specific context types\n")
        f.write("4. **Diminishing Returns**: Assess whether full context provides proportional benefits\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the context impact analysis:\n\n")
        f.write("- **High-Impact Contexts**: Focus on context types that show significant improvements\n")
        f.write("- **Task-Specific Optimization**: Tailor context based on task category requirements\n")
        f.write("- **Context Efficiency**: Use minimal context where full context shows diminishing returns\n")
        f.write("- **Evaluation Strategy**: Include context variations in comprehensive evaluations\n\n")


def generate_context_visualizations(results, output_dir):
    """Generate comprehensive visualizations for context impact analysis."""
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Group results by task category
        task_results = {}
        for result_key, result_data in results.items():
            # Extract task name and context mode
            for context_mode in ['_full_context', '_no_context', '_minimal_context']:
                if context_mode in result_key:
                    task_name = result_key.replace(context_mode, '')
                    context_key = context_mode[1:]  # Remove leading underscore
                    if task_name not in task_results:
                        task_results[task_name] = {}
                    task_results[task_name][context_key] = result_data
                    break
        
        # Prepare comprehensive data for plotting
        all_metrics_data = []
        improvement_data = []
        bleu_data = []
        
        for task_name, task_data in task_results.items():
            if 'full_context' in task_data and 'no_context' in task_data:
                full_results = task_data['full_context']
                no_results = task_data['no_context']
                
                full_task_key = list(full_results.keys())[0] if full_results else None
                no_task_key = list(no_results.keys())[0] if no_results else None
                
                if full_task_key and no_task_key:
                    full_metrics = full_results[full_task_key]
                    no_metrics = no_results[no_task_key]
                    
                    # Collect all metrics for comparison
                    metrics_to_plot = ['context_adherence_score', 'style_compliance_score', 'security_compliance_score', 'performance_awareness_score']
                    
                    for metric in metrics_to_plot:
                        full_val = full_metrics.get(metric, 0)
                        no_val = no_metrics.get(metric, 0)
                        
                        if isinstance(full_val, (int, float)) and isinstance(no_val, (int, float)):
                            # Add data for comparison charts
                            all_metrics_data.append({
                                'Task': task_name.replace('_', ' ').title(),
                                'Context': 'Full Context',
                                'Metric': metric.replace('_', ' ').title(),
                                'Score': full_val
                            })
                            all_metrics_data.append({
                                'Task': task_name.replace('_', ' ').title(),
                                'Context': 'No Context',
                                'Metric': metric.replace('_', ' ').title(),
                                'Score': no_val
                            })
                            
                            # Calculate improvement
                            if no_val > 0:
                                improvement = ((full_val - no_val) / no_val) * 100
                                improvement_data.append({
                                    'Task': task_name.replace('_', ' ').title(),
                                    'Metric': metric.replace('_', ' ').title(),
                                    'Improvement': improvement
                                })
                    
                    # Handle BLEU scores separately
                    full_bleu = full_metrics.get('bleu', 0)
                    no_bleu = no_metrics.get('bleu', 0)
                    
                    if isinstance(full_bleu, (int, float)) and isinstance(no_bleu, (int, float)) and (full_bleu > 0 or no_bleu > 0):
                        bleu_data.append({
                            'Task': task_name.replace('_', ' ').title(),
                            'Context': 'Full Context',
                            'BLEU': full_bleu
                        })
                        bleu_data.append({
                            'Task': task_name.replace('_', ' ').title(),
                            'Context': 'No Context',
                            'BLEU': no_bleu
                        })
        
        # Create multiple comprehensive charts
        
        # Chart 1: Context Adherence and Style Compliance Comparison
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig1.suptitle('Context Impact Analysis - Core Metrics Comparison', fontsize=16, fontweight='bold')
        
        if all_metrics_data:
            df_all = pd.DataFrame(all_metrics_data)
            
            # Context Adherence
            df_context = df_all[df_all['Metric'] == 'Context Adherence Score']
            if not df_context.empty:
                sns.barplot(data=df_context, x='Task', y='Score', hue='Context', ax=ax1)
                ax1.set_title('Context Adherence Score by Task')
                ax1.set_xlabel('Task Category')
                ax1.set_ylabel('Context Adherence Score')
                ax1.tick_params(axis='x', rotation=45)
                ax1.set_ylim(0, 1.1)
                
                # Add value labels
                for container in ax1.containers:
                    ax1.bar_label(container, fmt='%.2f', fontsize=9)
            
            # Style Compliance
            df_style = df_all[df_all['Metric'] == 'Style Compliance Score']
            if not df_style.empty:
                sns.barplot(data=df_style, x='Task', y='Score', hue='Context', ax=ax2)
                ax2.set_title('Style Compliance Score by Task')
                ax2.set_xlabel('Task Category')
                ax2.set_ylabel('Style Compliance Score')
                ax2.tick_params(axis='x', rotation=45)
                ax2.set_ylim(0, 1.1)
                
                # Add value labels
                for container in ax2.containers:
                    ax2.bar_label(container, fmt='%.2f', fontsize=9)
        
        plt.tight_layout()
        chart1_path = os.path.join(output_dir, "context_impact_core_metrics.png")
        plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Chart 2: BLEU Score Comparison and Improvement Analysis
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig2.suptitle('Context Impact Analysis - BLEU Scores and Improvements', fontsize=16, fontweight='bold')
        
        # BLEU Scores
        if bleu_data:
            df_bleu = pd.DataFrame(bleu_data)
            sns.barplot(data=df_bleu, x='Task', y='BLEU', hue='Context', ax=ax1)
            ax1.set_title('BLEU Score Comparison')
            ax1.set_xlabel('Task Category')
            ax1.set_ylabel('BLEU Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for container in ax1.containers:
                ax1.bar_label(container, fmt='%.3f', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No BLEU data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('BLEU Score Comparison')
        
        # Improvement percentages
        if improvement_data:
            df_imp = pd.DataFrame(improvement_data)
            # Focus on context adherence improvements
            df_context_imp = df_imp[df_imp['Metric'] == 'Context Adherence Score']
            if not df_context_imp.empty:
                bars = ax2.bar(df_context_imp['Task'], df_context_imp['Improvement'])
                ax2.set_title('Context Adherence Improvement (%)')
                ax2.set_xlabel('Task Category')
                ax2.set_ylabel('Improvement %')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
                
                # Color bars based on improvement
                for bar, improvement in zip(bars, df_context_imp['Improvement']):
                    if improvement > 0:
                        bar.set_color('green')
                    elif improvement < 0:
                        bar.set_color('red')
                    else:
                        bar.set_color('gray')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%', ha='center', 
                            va='bottom' if height > 0 else 'top', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No improvement data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Context Adherence Improvement (%)')
        
        plt.tight_layout()
        chart2_path = os.path.join(output_dir, "context_impact_bleu_improvements.png")
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Chart 3: Comprehensive Metrics Heatmap
        fig3, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig3.suptitle('Context Impact Analysis - All Metrics Heatmap', fontsize=16, fontweight='bold')
        
        if all_metrics_data:
            df_all = pd.DataFrame(all_metrics_data)
            # Create pivot table for heatmap
            pivot_full = df_all[df_all['Context'] == 'Full Context'].pivot(index='Task', columns='Metric', values='Score')
            pivot_no = df_all[df_all['Context'] == 'No Context'].pivot(index='Task', columns='Metric', values='Score')
            
            # Calculate improvement matrix
            improvement_matrix = ((pivot_full - pivot_no) / pivot_no * 100).fillna(0)
            
            # Create heatmap
            sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                       ax=ax, cbar_kws={'label': 'Improvement %'})
            ax.set_title('Context Impact Heatmap (% Improvement: Full vs No Context)')
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Task Categories')
        else:
            ax.text(0.5, 0.5, 'No data available for heatmap', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Context Impact Heatmap')
        
        plt.tight_layout()
        chart3_path = os.path.join(output_dir, "context_impact_heatmap.png")
        plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Chart 4: Overall Summary Dashboard
        fig4, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig4.suptitle('Context Impact Analysis - Summary Dashboard', fontsize=16, fontweight='bold')
        
        # Overall average improvements by metric
        if improvement_data:
            df_imp = pd.DataFrame(improvement_data)
            avg_improvements = df_imp.groupby('Metric')['Improvement'].mean().reset_index()
            
            bars = ax1.bar(avg_improvements['Metric'], avg_improvements['Improvement'])
            ax1.set_title('Average Improvement by Metric')
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Average Improvement %')
            ax1.tick_params(axis='x', rotation=45)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Color bars
            for bar, improvement in zip(bars, avg_improvements['Improvement']):
                if improvement > 0:
                    bar.set_color('green')
                elif improvement < 0:
                    bar.set_color('red')
                else:
                    bar.set_color('gray')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', 
                        va='bottom' if height > 0 else 'top', fontsize=9)
        
        # Task-wise average performance
        if all_metrics_data:
            df_all = pd.DataFrame(all_metrics_data)
            task_avg = df_all.groupby(['Task', 'Context'])['Score'].mean().reset_index()
            
            sns.barplot(data=task_avg, x='Task', y='Score', hue='Context', ax=ax2)
            ax2.set_title('Average Performance by Task')
            ax2.set_xlabel('Task Categories')
            ax2.set_ylabel('Average Score')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for container in ax2.containers:
                ax2.bar_label(container, fmt='%.2f', fontsize=8)
        
        # Security and Performance metrics
        if all_metrics_data:
            df_all = pd.DataFrame(all_metrics_data)
            df_sec_perf = df_all[df_all['Metric'].isin(['Security Compliance Score', 'Performance Awareness Score'])]
            
            if not df_sec_perf.empty:
                sns.barplot(data=df_sec_perf, x='Task', y='Score', hue='Context', ax=ax3)
                ax3.set_title('Security & Performance Metrics')
                ax3.set_xlabel('Task Categories')
                ax3.set_ylabel('Score')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for container in ax3.containers:
                    ax3.bar_label(container, fmt='%.2f', fontsize=8)
        
        # Impact summary pie chart
        if improvement_data:
            df_imp = pd.DataFrame(improvement_data)
            
            # Categorize improvements
            positive_count = len(df_imp[df_imp['Improvement'] > 5])
            neutral_count = len(df_imp[(df_imp['Improvement'] >= -5) & (df_imp['Improvement'] <= 5)])
            negative_count = len(df_imp[df_imp['Improvement'] < -5])
            
            if positive_count + neutral_count + negative_count > 0:
                labels = ['Positive Impact\n(>5%)', 'Neutral Impact\n(-5% to 5%)', 'Negative Impact\n(<-5%)']
                sizes = [positive_count, neutral_count, negative_count]
                colors = ['green', 'yellow', 'red']
                
                # Filter out zero values
                non_zero_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
                if non_zero_data:
                    labels, sizes, colors = zip(*non_zero_data)
                    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax4.set_title('Context Impact Distribution')
        
        plt.tight_layout()
        chart4_path = os.path.join(output_dir, "context_impact_dashboard.png")
        plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a combined overview chart
        plot_path = os.path.join(output_dir, "context_impact_charts.png")
        
        # Copy the dashboard as the main chart
        import shutil
        shutil.copy2(chart4_path, plot_path)
        
        print(f"📈 Context impact visualizations saved:")
        print(f"   - Core Metrics: {os.path.basename(chart1_path)}")
        print(f"   - BLEU & Improvements: {os.path.basename(chart2_path)}")
        print(f"   - Heatmap: {os.path.basename(chart3_path)}")
        print(f"   - Dashboard: {os.path.basename(chart4_path)}")
        print(f"   - Main Chart: {os.path.basename(plot_path)}")
        
    except Exception as e:
        print(f"⚠️  Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing without charts...")


def main():
    """Main function to run context impact analysis."""
    parser = argparse.ArgumentParser(description='Analyze context impact on Python coding evaluation')
    parser.add_argument('--results_dir', default='results/context_comparison',
                       help='Directory containing result files from run_context_comparison.sh')
    
    args = parser.parse_args()
    
    print("🐍 Python Coding Evaluation - Context Impact Analysis")
    print("=" * 60)
    print()
    
    success = analyze_context_impact(args.results_dir)
    
    if not success:
        print("\n💡 To generate results, run:")
        print("   ./run_context_comparison.sh --limit 5")
        print("   Then run this analysis script again.")
        sys.exit(1)


if __name__ == "__main__":
    main()