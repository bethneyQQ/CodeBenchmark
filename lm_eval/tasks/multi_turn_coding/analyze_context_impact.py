#!/usr/bin/env python3

import json
import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  seaborn not available - using matplotlib only for visualizations")


def load_results(file_path):
    """Load evaluation results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_metrics(results):
    """Extract key metrics from results."""
    if not results or 'results' not in results:
        return {}
    
    metrics = {}
    for task_name, task_results in results['results'].items():
        for metric_name, metric_value in task_results.items():
            if isinstance(metric_value, (int, float)):
                metrics[metric_name] = metric_value
    
    return metrics


def find_latest_file(results_dir, pattern):
    """Find the most recent file matching a pattern."""
    import glob
    files = glob.glob(os.path.join(results_dir, pattern))
    if files:
        # Sort by modification time, return most recent
        return max(files, key=os.path.getmtime)
    return None


def detect_difficulty_from_path(results_dir):
    """Detect difficulty level from results directory path."""
    dir_name = os.path.basename(results_dir.rstrip('/'))
    
    # Check for difficulty keywords in directory name
    difficulty_keywords = ['easy', 'simple', 'medium', 'complex']
    for keyword in difficulty_keywords:
        if keyword in dir_name.lower():
            return keyword
    
    return 'unknown'


def analyze_difficulty_impact(results_dirs):
    """Analyze context impact across different difficulty levels."""
    
    difficulty_results = {}
    
    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            print(f"⚠️  Directory not found: {results_dir}")
            continue
            
        difficulty = detect_difficulty_from_path(results_dir)
        print(f"\n🎯 Analyzing {difficulty.capitalize()} Difficulty: {results_dir}")
        print("-" * 60)
        
        # Analyze this difficulty level
        scenarios = {
            'Full Context': 'full_context_results*.json',
            'No Context': 'no_context_results*.json'
        }
        
        difficulty_data = {}
        for scenario_name, pattern in scenarios.items():
            # Try to find file matching pattern
            exact_file = os.path.join(results_dir, pattern.replace('*', ''))
            if os.path.exists(exact_file):
                file_path = exact_file
            else:
                file_path = find_latest_file(results_dir, pattern)
            
            if file_path and os.path.exists(file_path):
                results = load_results(file_path)
                if results:
                    difficulty_data[scenario_name] = extract_metrics(results)
                    print(f"  ✅ {scenario_name}: {os.path.basename(file_path)}")
            else:
                print(f"  ⚠️  Missing {scenario_name}: {pattern}")
        
        if difficulty_data:
            difficulty_results[difficulty] = difficulty_data
    
    return difficulty_results


def create_difficulty_visualizations(difficulty_results, output_dir):
    """Create visualizations comparing context impact across difficulties."""
    
    if not difficulty_results:
        print("⚠️  No difficulty data available for visualization")
        return
    
    # Prepare data for visualization
    plot_data = []
    for difficulty, scenarios in difficulty_results.items():
        for scenario, metrics in scenarios.items():
            for metric, value in metrics.items():
                clean_metric = clean_metric_name(metric)
                plot_data.append({
                    'Difficulty': difficulty.capitalize(),
                    'Scenario': scenario,
                    'Metric': clean_metric,
                    'Value': value
                })
    
    if not plot_data:
        print("⚠️  No data available for difficulty visualization")
        return
    
    df = pd.DataFrame(plot_data)
    
    # Set up the plotting style with better colors
    plt.style.use('default')
    
    # Define distinct colors for scenarios
    scenario_colors = {
        'Full Context': '#2E86AB',    # Blue
        'No Context': '#A23B72',      # Dark Pink/Purple
        'PRD Only': '#F18F01',        # Orange
        'Technical Only': '#4CAF50',  # Green
        'Design+Code Only': '#9C27B0' # Purple
    }
    
    if HAS_SEABORN:
        # Create custom palette for seaborn
        unique_scenarios = df['Scenario'].unique()
        palette = [scenario_colors.get(scenario, '#607D8B') for scenario in unique_scenarios]
        sns.set_palette(palette)
    
    # Get unique metrics and create optimal layout
    unique_metrics = df['Metric'].unique()
    n_metrics = len(unique_metrics)
    
    # Calculate optimal subplot layout
    if n_metrics <= 4:
        rows, cols = 2, 2
        figsize = (12, 10)
    elif n_metrics <= 6:
        rows, cols = 2, 3
        figsize = (18, 10)
    elif n_metrics <= 9:
        rows, cols = 3, 3
        figsize = (18, 15)
    else:
        rows, cols = 4, 3
        figsize = (18, 20)
    
    # Create difficulty comparison chart
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('Context Impact Analysis Across Difficulty Levels', fontsize=16, fontweight='bold')
    
    # Handle single subplot case
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(unique_metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        metric_data = df[df['Metric'] == metric]
        
        if HAS_SEABORN:
            sns.barplot(data=metric_data, x='Difficulty', y='Value', hue='Scenario', ax=ax)
        else:
            # Fallback to matplotlib with custom colors
            difficulties = metric_data['Difficulty'].unique()
            scenarios = metric_data['Scenario'].unique()
            
            x = range(len(difficulties))
            width = 0.35
            
            for j, scenario in enumerate(scenarios):
                scenario_data = metric_data[metric_data['Scenario'] == scenario]
                values = [scenario_data[scenario_data['Difficulty'] == d]['Value'].iloc[0] 
                         if len(scenario_data[scenario_data['Difficulty'] == d]) > 0 else 0 
                         for d in difficulties]
                
                # Use distinct color for each scenario
                color = scenario_colors.get(scenario, '#607D8B')
                ax.bar([xi + j*width for xi in x], values, width, label=scenario, color=color, alpha=0.8)
            
            ax.set_xlabel('Difficulty')
            ax.set_ylabel('Score')
            ax.set_xticks([xi + width/2 for xi in x])
            ax.set_xticklabels(difficulties)
            ax.legend()
        
        ax.set_title(f'{metric}', fontweight='bold', fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8)
    
    # Remove empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    difficulty_chart_path = os.path.join(output_dir, 'difficulty_context_analysis.png')
    plt.savefig(difficulty_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Difficulty analysis chart saved: {difficulty_chart_path}")
    
    # Create difficulty heatmap
    create_difficulty_heatmap(difficulty_results, output_dir)


def create_difficulty_heatmap(difficulty_results, output_dir):
    """Create a heatmap showing context improvement across difficulties."""
    
    # Calculate improvement percentages
    improvement_data = []
    
    for difficulty, scenarios in difficulty_results.items():
        if 'Full Context' in scenarios and 'No Context' in scenarios:
            full_context = scenarios['Full Context']
            no_context = scenarios['No Context']
            
            for metric in full_context:
                if metric in no_context:
                    baseline = no_context[metric]
                    full_value = full_context[metric]
                    
                    if baseline > 0:
                        improvement = ((full_value - baseline) / baseline) * 100
                    else:
                        improvement = 0
                    
                    clean_metric = clean_metric_name(metric)
                    improvement_data.append({
                        'Difficulty': difficulty.capitalize(),
                        'Metric': clean_metric,
                        'Improvement': improvement
                    })
    
    if not improvement_data:
        print("⚠️  No improvement data available for heatmap")
        return
    
    df = pd.DataFrame(improvement_data)
    
    # Pivot for heatmap
    heatmap_data = df.pivot(index='Difficulty', columns='Metric', values='Improvement')
    
    # Create heatmap with better sizing
    plt.figure(figsize=(max(14, len(heatmap_data.columns) * 1.2), 8))
    
    if HAS_SEABORN:
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Improvement (%)'}, square=False)
    else:
        # Fallback to matplotlib imshow
        im = plt.imshow(heatmap_data.values, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, label='Improvement (%)')
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                plt.text(j, i, f'{heatmap_data.iloc[i, j]:.1f}%', 
                        ha='center', va='center', fontsize=9)
        
        # Set ticks and labels
        plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45)
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    
    plt.title('Context Impact by Difficulty Level\n(% Improvement: Full Context vs No Context)', 
              fontweight='bold', pad=20, fontsize=14)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Difficulty Levels', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the heatmap
    difficulty_heatmap_path = os.path.join(output_dir, 'difficulty_context_heatmap.png')
    plt.savefig(difficulty_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🔥 Difficulty heatmap saved: {difficulty_heatmap_path}")
    
    return heatmap_data


def analyze_context_impact(results_dir):
    """Analyze the impact of different context configurations."""
    
    # Define scenarios and their file patterns (for simple_test.sh and run_difficulty_comparison.sh output)
    scenarios = {
        'Full Context': ['full_context_test*.json', 'full_context_results*.json'],
        'No Context': ['no_context_test*.json', 'no_context_results*.json'], 
        'PRD Only': 'prd_only_results.json',
        'Technical Only': 'technical_only_results.json',
        'Design+Code Only': 'design_code_only_results.json'
    }
    
    # Load all results
    all_results = {}
    for scenario_name, patterns in scenarios.items():
        # Handle both single pattern and list of patterns
        if isinstance(patterns, str):
            patterns = [patterns]
        
        file_path = None
        for pattern in patterns:
            # First try exact filename (for full context comparison runs)
            exact_file = os.path.join(results_dir, pattern.replace('*', ''))
            if os.path.exists(exact_file):
                file_path = exact_file
                break
            else:
                # Try to find latest file matching pattern (for simple_test.sh runs)
                file_path = find_latest_file(results_dir, pattern)
                if file_path:
                    break
        
        if file_path and os.path.exists(file_path):
            results = load_results(file_path)
            if results:
                all_results[scenario_name] = extract_metrics(results)
                print(f"✅ Loaded {scenario_name}: {os.path.basename(file_path)}")
        else:
            print(f"⚠️  Missing {scenario_name}: {patterns}")
    
    if not all_results:
        print("❌ No results found to analyze")
        return
    
    # Create comparison DataFrame
    df_data = []
    for scenario, metrics in all_results.items():
        row = {'Scenario': scenario}
        row.update(metrics)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Print summary table
    print("\n📊 Context Impact Analysis")
    print("=" * 80)
    
    metric_columns = [col for col in df.columns if col != 'Scenario']
    
    if metric_columns:
        print(f"{'Scenario':<20}", end="")
        for metric in metric_columns:
            print(f"{metric:<20}", end="")
        print()
        print("-" * (20 + len(metric_columns) * 20))
        
        for _, row in df.iterrows():
            print(f"{row['Scenario']:<20}", end="")
            for metric in metric_columns:
                value = row.get(metric, 0)
                print(f"{value:<20.3f}", end="")
            print()
    
    # Calculate improvements over baseline
    if 'No Context' in all_results:
        baseline = all_results['No Context']
        print(f"\n📈 Improvement over No Context Baseline")
        print("=" * 60)
        
        for scenario, metrics in all_results.items():
            if scenario == 'No Context':
                continue
                
            print(f"\n{scenario}:")
            for metric, value in metrics.items():
                if metric in baseline:
                    baseline_value = baseline[metric]
                    if baseline_value > 0:
                        improvement = ((value - baseline_value) / baseline_value) * 100
                        print(f"  {metric}: {improvement:+.1f}% ({value:.3f} vs {baseline_value:.3f})")
    
    # Generate visualizations if matplotlib is available
    try:
        create_visualizations(df, results_dir)
    except ImportError:
        print("\n📊 Install matplotlib and seaborn for visualizations:")
        print("pip install matplotlib seaborn")
    except Exception as e:
        print(f"\n⚠️  Error creating visualizations: {e}")
    
    # Save summary report
    save_summary_report(df, all_results, results_dir)


def clean_metric_name(metric_name):
    """Clean metric names by removing suffixes and making them readable."""
    # Remove common suffixes
    clean_name = metric_name.replace(',extract_responses', '').replace('_', ' ')
    
    # Convert to title case and handle special cases
    clean_name = clean_name.title()
    
    # Fix specific metric names for better readability
    name_mappings = {
        'File Existence Check': 'File Existence',
        'Prd Quality From File': 'PRD Quality',
        'Design Coherence From File': 'Design Quality',
        'Code Execution Test': 'Code Quality',
        'Project Structure Validation': 'Project Structure',
        'Integration Test': 'Integration',
        'Architecture Quality Assessment': 'Architecture',
        'Policy Utilization Score': 'Policy Utilization',
        'Policy Adherence Score': 'Policy Adherence',
        'Technical Constraint Adherence': 'Technical Constraints',
        'Performance Requirement Coverage': 'Performance Coverage',
        'Security Compliance Check': 'Security Compliance',
        'Execution Time Efficiency': 'Time Efficiency',
        'Token Cost Estimation': 'Cost Estimation'
    }
    
    return name_mappings.get(clean_name, clean_name)


def create_visualizations(df, results_dir):
    """Create visualization charts with improved layout and readability."""
    
    metric_columns = [col for col in df.columns if col != 'Scenario']
    
    if not metric_columns:
        return
    
    # Clean metric names for better display
    clean_metrics = {col: clean_metric_name(col) for col in metric_columns}
    
    # Set up the plotting style with better colors
    plt.style.use('default')
    
    # Define a clear, distinguishable color palette
    distinct_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']
    
    if HAS_SEABORN:
        sns.set_palette(distinct_colors)
    else:
        # Set matplotlib color cycle
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=distinct_colors)
    
    # Create subplots with better layout to prevent overlapping
    n_metrics = len(metric_columns)
    
    # Calculate optimal subplot layout
    if n_metrics <= 4:
        rows, cols = 2, 2
        figsize = (12, 8)
    elif n_metrics <= 6:
        rows, cols = 2, 3
        figsize = (15, 8)
    elif n_metrics <= 9:
        rows, cols = 3, 3
        figsize = (15, 12)
    else:
        rows, cols = 4, 3
        figsize = (18, 16)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('Context Impact on Multi-Turn Coding Evaluation', fontsize=16, fontweight='bold')
    
    # Handle single subplot case
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Create bar plots for each metric
    for i, metric in enumerate(metric_columns):
        if i >= len(axes):
            break
            
        ax = axes[i]
        clean_name = clean_metrics[metric]
        
        # Create bar plot with distinct colors
        scenarios = df['Scenario'].unique()
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#9C27B0'][:len(scenarios)]
        
        bars = ax.bar(df['Scenario'], df[metric], alpha=0.8, color=colors)
        ax.set_title(clean_name, fontsize=11, fontweight='bold', pad=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Set y-axis limits for better visualization
        ax.set_ylim(0, max(df[metric]) * 1.15)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(results_dir, 'context_impact_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved: {plot_path}")
    
    # Create a heatmap if we have multiple metrics
    if len(metric_columns) > 1:
        plt.figure(figsize=(max(12, len(metric_columns) * 1.2), 6))
        
        # Prepare data for heatmap with clean names
        heatmap_df = df.set_index('Scenario')[metric_columns].copy()
        heatmap_df.columns = [clean_metrics[col] for col in heatmap_df.columns]
        heatmap_data = heatmap_df.T
        
        # Create heatmap with better color scheme
        if HAS_SEABORN:
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', 
                       cbar_kws={'label': 'Score'}, square=False)
        else:
            # Fallback to matplotlib imshow
            im = plt.imshow(heatmap_data.values, cmap='viridis', aspect='auto')
            plt.colorbar(im, label='Score')
            
            # Add text annotations
            for i in range(len(heatmap_data.index)):
                for j in range(len(heatmap_data.columns)):
                    plt.text(j, i, f'{heatmap_data.iloc[i, j]:.3f}', 
                            ha='center', va='center', fontsize=9)
            
            # Set ticks and labels
            plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45)
            plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
            
        plt.title('Context Configuration Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Metrics', fontsize=12)
        plt.xlabel('Context Scenarios', fontsize=12)
        plt.xticks(rotation=45)
        
        # Save heatmap
        heatmap_path = os.path.join(results_dir, 'context_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🔥 Heatmap saved: {heatmap_path}")


def get_metric_definitions():
    """Return definitions for all metrics."""
    return {
        'file_existence_check': 'Verifies that all required files (PRD, design, source code) are created during evaluation.',
        'prd_quality_from_file': 'Assesses Product Requirements Document quality including completeness, specific metrics, and business value.',
        'design_coherence_from_file': 'Evaluates technical design quality including architecture patterns, scalability considerations, and documentation.',
        'code_execution_test': 'Multi-dimensional code quality assessment covering syntax, structure, testing, and organization.',
        'project_structure_validation': 'Validates Python project organization including configuration, documentation, and module structure.',
        'integration_test': 'Measures consistency and alignment between PRD, design, and implementation phases.',
        'architecture_quality_assessment': 'Evaluates architectural decisions, design patterns, and implementation alignment.',
        'policy_utilization_score': 'Measures how effectively the provided context information is utilized in the solution.',
        'policy_adherence_score': 'Checks adherence to specific requirements and policies mentioned in the context.',
        'technical_constraint_adherence': 'Verifies that technical constraints specified in context are followed in implementation.',
        'performance_requirement_coverage': 'Assesses whether performance requirements from context are addressed in the solution.',
        'security_compliance_check': 'Verifies that security and compliance requirements are properly implemented.',
        'execution_time_efficiency': 'Measures evaluation completion time in seconds (lower is better).',
        'token_cost_estimation': 'Estimates API usage cost in USD based on token consumption (lower is better).'
    }


def clean_metric_name_for_table(metric_name):
    """Clean metric names for table display."""
    return metric_name.replace(',extract_responses', '').replace('_', ' ').title()


def save_summary_report(df, all_results, results_dir):
    """Save a detailed summary report."""
    
    report_path = os.path.join(results_dir, 'context_impact_report.md')
    metric_definitions = get_metric_definitions()
    
    with open(report_path, 'w') as f:
        f.write("# Multi-Turn Coding Evaluation - Context Impact Analysis\n\n")
        
        # Add table of contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Summary](#summary)\n")
        f.write("2. [Metric Definitions](#metric-definitions)\n")
        f.write("3. [Scenarios Tested](#scenarios-tested)\n")
        f.write("4. [Results Summary](#results-summary)\n")
        f.write("5. [Improvement Over Baseline](#improvement-over-baseline)\n")
        f.write("6. [Visualizations](#visualizations)\n")
        f.write("7. [Key Insights](#key-insights)\n")
        f.write("8. [Recommendations](#recommendations)\n")
        f.write("9. [Generated Files](#generated-files)\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report analyzes the impact of different context configurations ")
        f.write("on multi-turn coding evaluation performance. The analysis includes ")
        f.write("quantitative metrics, visual comparisons, and actionable insights ")
        f.write("for optimizing context usage in multi-turn coding tasks.\n\n")
        
        # Add generation timestamp
        from datetime import datetime
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Results Directory:** `{results_dir}`\n\n")
        
        # Add metric definitions section
        f.write("## Metric Definitions\n\n")
        f.write("This section explains what each metric measures to help interpret the results.\n\n")
        
        # Group metrics by category
        quality_metrics = ['file_existence_check', 'prd_quality_from_file', 'design_coherence_from_file', 
                          'code_execution_test', 'project_structure_validation', 'integration_test']
        context_metrics = ['policy_utilization_score', 'policy_adherence_score', 'technical_constraint_adherence', 
                          'architecture_quality_assessment']
        compliance_metrics = ['performance_requirement_coverage', 'security_compliance_check']
        efficiency_metrics = ['execution_time_efficiency', 'token_cost_estimation']
        
        f.write("### Quality Assessment Metrics\n")
        for metric in quality_metrics:
            if metric in metric_definitions:
                clean_name = clean_metric_name_for_table(metric)
                f.write(f"- **{clean_name}**: {metric_definitions[metric]}\n")
        
        f.write("\n### Context-Dependent Metrics\n")
        for metric in context_metrics:
            if metric in metric_definitions:
                clean_name = clean_metric_name_for_table(metric)
                f.write(f"- **{clean_name}**: {metric_definitions[metric]}\n")
        
        f.write("\n### Compliance Metrics\n")
        for metric in compliance_metrics:
            if metric in metric_definitions:
                clean_name = clean_metric_name_for_table(metric)
                f.write(f"- **{clean_name}**: {metric_definitions[metric]}\n")
        
        f.write("\n### Efficiency Metrics\n")
        for metric in efficiency_metrics:
            if metric in metric_definitions:
                clean_name = clean_metric_name_for_table(metric)
                f.write(f"- **{clean_name}**: {metric_definitions[metric]}\n")
        
        f.write("\n## Scenarios Tested\n\n")
        for scenario in all_results.keys():
            f.write(f"- **{scenario}**: ")
            if scenario == "Full Context":
                f.write("All context fields enabled (PRD, Design, Code, Quality)")
            elif scenario == "No Context":
                f.write("No context information provided (baseline)")
            elif scenario == "PRD Only":
                f.write("Only PRD context enabled")
            elif scenario == "Technical Only":
                f.write("Design, Code, and Quality contexts enabled")
            elif scenario == "Design+Code Only":
                f.write("Only Design and Code contexts enabled")
            f.write("\n")
        
        f.write("\n## Results Summary\n\n")
        f.write("| Scenario | ")
        
        metric_columns = [col for col in df.columns if col != 'Scenario']
        # Use clean names for table headers
        for metric in metric_columns:
            clean_name = clean_metric_name_for_table(metric)
            f.write(f"{clean_name} | ")
        f.write("\n")
        
        f.write("|----------|")
        for _ in metric_columns:
            f.write("----------|")
        f.write("\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['Scenario']} | ")
            for metric in metric_columns:
                value = row.get(metric, 0)
                f.write(f"{value:.3f} | ")
            f.write("\n")
        
        # Add improvement analysis
        if 'No Context' in all_results:
            baseline = all_results['No Context']
            f.write("\n## Improvement Over Baseline\n\n")
            
            for scenario, metrics in all_results.items():
                if scenario == 'No Context':
                    continue
                    
                f.write(f"### {scenario}\n\n")
                for metric, value in metrics.items():
                    if metric in baseline:
                        baseline_value = baseline[metric]
                        clean_name = clean_metric_name_for_table(metric)
                        if baseline_value > 0:
                            improvement = ((value - baseline_value) / baseline_value) * 100
                            f.write(f"- **{clean_name}**: {improvement:+.1f}% ")
                            f.write(f"({value:.3f} vs {baseline_value:.3f})\n")
                        else:
                            # Handle zero baseline case
                            if value > 0:
                                f.write(f"- **{clean_name}**: New value {value:.3f} ")
                                f.write(f"(baseline was 0.000)\n")
                f.write("\n")
        
        # Add visualizations section
        f.write("## Visualizations\n\n")
        
        # Check if visualization files exist and add them to the report
        bar_chart_path = os.path.join(results_dir, 'context_impact_analysis.png')
        heatmap_path = os.path.join(results_dir, 'context_heatmap.png')
        
        if os.path.exists(bar_chart_path):
            f.write("### Performance Comparison Chart\n\n")
            f.write('<img src="./context_impact_analysis.png" alt="Context Impact Analysis Bar Chart" width="800">\n\n')
            f.write("*Bar chart showing performance metrics across different context configurations. ")
            f.write("Each metric is displayed as grouped bars for easy comparison between scenarios.*\n\n")
        
        if os.path.exists(heatmap_path):
            f.write("### Performance Heatmap\n\n")
            f.write('<img src="./context_heatmap.png" alt="Context Performance Heatmap" width="600">\n\n')
            f.write("*Heatmap visualization showing the performance matrix across all scenarios and metrics. ")
            f.write("Darker colors indicate higher performance scores.*\n\n")
        
        if not os.path.exists(bar_chart_path) and not os.path.exists(heatmap_path):
            f.write("*Visualizations not available. Install matplotlib and seaborn for chart generation.*\n\n")
        else:
            f.write("### How to Interpret the Visualizations\n\n")
            f.write("**Bar Chart:**\n")
            f.write("- Each metric is shown as a group of bars\n")
            f.write("- Different colors represent different context scenarios\n")
            f.write("- Higher bars indicate better performance\n")
            f.write("- Compare bar heights within each metric group\n\n")
            
            f.write("**Heatmap:**\n")
            f.write("- Rows represent different context scenarios\n")
            f.write("- Columns represent different evaluation metrics\n")
            f.write("- Color intensity indicates performance level (darker = better)\n")
            f.write("- Use to quickly identify best-performing scenario-metric combinations\n\n")
        
        f.write("## Key Insights\n\n")
        f.write("1. **Context Impact**: Compare how different context configurations ")
        f.write("affect model performance across all evaluation metrics.\n\n")
        f.write("2. **Phase-Specific Effects**: Analyze which contexts have the ")
        f.write("strongest impact on specific phases (PRD, Design, Implementation).\n\n")
        f.write("3. **Baseline Comparison**: Understand the model's inherent ")
        f.write("capabilities without external guidance.\n\n")
        f.write("4. **Visual Analysis**: Use the charts above to identify patterns ")
        f.write("and trends in context effectiveness.\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the results, consider:\n\n")
        f.write("- Which context configurations provide the best performance\n")
        f.write("- Whether certain contexts are more critical than others\n")
        f.write("- How to optimize context information for your specific use case\n")
        f.write("- Use the heatmap to identify which metrics benefit most from context\n")
        
        # Add file references section
        f.write("\n## Generated Files\n\n")
        f.write("This analysis generated the following files:\n\n")
        
        if os.path.exists(bar_chart_path):
            f.write("- `context_impact_analysis.png` - Performance comparison bar chart\n")
        if os.path.exists(heatmap_path):
            f.write("- `context_heatmap.png` - Performance heatmap visualization\n")
        
        f.write("- `context_impact_report.md` - This detailed analysis report\n")
        f.write("\n*All files are saved in the same directory as this report.*\n")
    
    print(f"📄 Summary report saved: {report_path}")


def save_difficulty_report(difficulty_results, output_dir):
    """Save a detailed difficulty analysis report with metrics definitions."""
    
    report_path = os.path.join(output_dir, 'difficulty_context_analysis_report.md')
    metric_definitions = get_metric_definitions()
    
    with open(report_path, 'w') as f:
        f.write("# Multi-Turn Coding Evaluation - Difficulty-Based Context Impact Analysis\n\n")
        
        # Add table of contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Summary](#summary)\n")
        f.write("2. [Metric Definitions](#metric-definitions)\n")
        f.write("3. [Difficulty Levels Analyzed](#difficulty-levels-analyzed)\n")
        f.write("4. [Results by Difficulty](#results-by-difficulty)\n")
        f.write("5. [Context Impact Comparison](#context-impact-comparison)\n")
        f.write("6. [Visualizations](#visualizations)\n")
        f.write("7. [Key Insights](#key-insights)\n")
        f.write("8. [Recommendations](#recommendations)\n")
        f.write("9. [Generated Files](#generated-files)\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report analyzes how context impact varies across different problem ")
        f.write("difficulty levels in multi-turn coding evaluation. The analysis compares ")
        f.write("Full Context vs No Context performance across Easy, Simple, Medium, and ")
        f.write("Complex problem difficulties.\n\n")
        
        # Add generation timestamp
        from datetime import datetime
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Output Directory:** `{output_dir}`\n\n")
        
        # Add metric definitions section
        f.write("## Metric Definitions\n\n")
        f.write("Understanding what each metric measures is crucial for interpreting the results ")
        f.write("and understanding how context affects different aspects of multi-turn coding performance.\n\n")
        
        # Group metrics by category for better organization
        quality_metrics = ['file_existence_check', 'prd_quality_from_file', 'design_coherence_from_file', 
                          'code_execution_test', 'project_structure_validation', 'integration_test']
        context_metrics = ['policy_utilization_score', 'policy_adherence_score', 'technical_constraint_adherence', 
                          'architecture_quality_assessment']
        compliance_metrics = ['performance_requirement_coverage', 'security_compliance_check']
        efficiency_metrics = ['execution_time_efficiency', 'token_cost_estimation']
        
        f.write("### Quality Assessment Metrics\n")
        f.write("These metrics evaluate the fundamental quality of outputs across different phases:\n\n")
        for metric in quality_metrics:
            if metric in metric_definitions:
                clean_name = clean_metric_name_for_table(metric)
                f.write(f"- **{clean_name}**: {metric_definitions[metric]}\n")
        
        f.write("\n### Context-Dependent Metrics\n")
        f.write("These metrics specifically measure how well the model utilizes provided context:\n\n")
        for metric in context_metrics:
            if metric in metric_definitions:
                clean_name = clean_metric_name_for_table(metric)
                f.write(f"- **{clean_name}**: {metric_definitions[metric]}\n")
        
        f.write("\n### Compliance and Requirements Metrics\n")
        f.write("These metrics assess adherence to specific requirements and constraints:\n\n")
        for metric in compliance_metrics:
            if metric in metric_definitions:
                clean_name = clean_metric_name_for_table(metric)
                f.write(f"- **{clean_name}**: {metric_definitions[metric]}\n")
        
        f.write("\n### Efficiency Metrics\n")
        f.write("These metrics measure resource usage and performance efficiency:\n\n")
        for metric in efficiency_metrics:
            if metric in metric_definitions:
                clean_name = clean_metric_name_for_table(metric)
                f.write(f"- **{clean_name}**: {metric_definitions[metric]}\n")
        
        f.write("\n### Metric Interpretation Guidelines\n\n")
        f.write("- **Higher is Better**: Most metrics (0.0-1.0 scale) where higher scores indicate better performance\n")
        f.write("- **Lower is Better**: Execution Time Efficiency and Token Cost Estimation where lower values are preferred\n")
        f.write("- **Context Sensitivity**: Metrics like Policy Utilization and Policy Adherence are most sensitive to context availability\n")
        f.write("- **Baseline Metrics**: File Existence and basic Code Execution should remain relatively stable across contexts\n\n")
        
        # Add difficulty levels section
        f.write("## Difficulty Levels Analyzed\n\n")
        for difficulty in sorted(difficulty_results.keys()):
            scenarios = list(difficulty_results[difficulty].keys())
            f.write(f"- **{difficulty.capitalize()}**: {len(scenarios)} scenarios ({', '.join(scenarios)})\n")
        
        # Add results by difficulty
        f.write("\n## Results by Difficulty\n\n")
        
        for difficulty in ['easy', 'simple', 'medium', 'complex']:
            if difficulty in difficulty_results:
                f.write(f"### {difficulty.capitalize()} Difficulty\n\n")
                
                scenarios = difficulty_results[difficulty]
                if not scenarios:
                    f.write("No data available for this difficulty level.\n\n")
                    continue
                
                # Create table header
                f.write("| Scenario | ")
                
                # Get all metrics from first scenario
                first_scenario = list(scenarios.values())[0]
                metrics = list(first_scenario.keys())
                
                for metric in metrics:
                    clean_name = clean_metric_name_for_table(metric)
                    f.write(f"{clean_name} | {clean_name} Stderr | ")
                f.write("\n")
                
                # Create separator row
                f.write("|----------|")
                for _ in metrics:
                    f.write("----------|----------|")
                f.write("\n")
                
                # Add data rows
                for scenario_name, scenario_data in scenarios.items():
                    f.write(f"| {scenario_name} | ")
                    for metric in metrics:
                        value = scenario_data.get(metric, 0)
                        f.write(f"{value:.3f} | 0.000 | ")  # Adding stderr placeholder
                    f.write("\n")
                
                f.write("\n")
        
        # Add context impact comparison
        f.write("## Context Impact Comparison\n\n")
        f.write("This section shows the percentage improvement when using Full Context ")
        f.write("compared to No Context baseline for each difficulty level.\n\n")
        
        for difficulty in ['easy', 'simple', 'medium', 'complex']:
            if difficulty in difficulty_results:
                scenarios = difficulty_results[difficulty]
                if 'Full Context' in scenarios and 'No Context' in scenarios:
                    f.write(f"### {difficulty.capitalize()} Difficulty Improvements\n\n")
                    
                    full_context = scenarios['Full Context']
                    no_context = scenarios['No Context']
                    
                    for metric in full_context:
                        if metric in no_context:
                            baseline = no_context[metric]
                            full_value = full_context[metric]
                            clean_name = clean_metric_name_for_table(metric)
                            
                            if baseline > 0:
                                improvement = ((full_value - baseline) / baseline) * 100
                                f.write(f"- **{clean_name}**: {improvement:+.1f}% ")
                                f.write(f"({full_value:.3f} vs {baseline:.3f})\n")
                            else:
                                if full_value > 0:
                                    f.write(f"- **{clean_name}**: New value {full_value:.3f} ")
                                    f.write(f"(baseline was 0.000)\n")
                    f.write("\n")
        
        # Add visualizations section
        f.write("## Visualizations\n\n")
        
        # Check if visualization files exist and add them to the report
        difficulty_chart_path = os.path.join(output_dir, 'difficulty_context_analysis.png')
        difficulty_heatmap_path = os.path.join(output_dir, 'difficulty_context_heatmap.png')
        
        if os.path.exists(difficulty_chart_path):
            f.write("### Difficulty Comparison Chart\n\n")
            f.write('<img src="./difficulty_context_analysis.png" alt="Difficulty Context Analysis" width="900">\n\n')
            f.write("*Comprehensive comparison of context impact across all difficulty levels and metrics.*\n\n")
        
        if os.path.exists(difficulty_heatmap_path):
            f.write("### Context Impact Heatmap by Difficulty\n\n")
            f.write('<img src="./difficulty_context_heatmap.png" alt="Difficulty Context Heatmap" width="700">\n\n')
            f.write("*Heatmap showing percentage improvement from Full Context vs No Context across difficulties.*\n\n")
        
        if os.path.exists(difficulty_chart_path) or os.path.exists(difficulty_heatmap_path):
            f.write("### How to Interpret the Visualizations\n\n")
            f.write("**Difficulty Comparison Chart:**\n")
            f.write("- Each subplot shows one evaluation metric\n")
            f.write("- X-axis represents difficulty levels (Easy → Complex)\n")
            f.write("- Different colored bars represent different context scenarios\n")
            f.write("- Higher bars indicate better performance\n\n")
            
            f.write("**Context Impact Heatmap:**\n")
            f.write("- Rows represent difficulty levels\n")
            f.write("- Columns represent evaluation metrics\n")
            f.write("- Color intensity shows percentage improvement (green = positive, red = negative)\n")
            f.write("- Numbers show exact improvement percentages\n\n")
        
        f.write("## Key Insights\n\n")
        f.write("1. **Difficulty Progression**: Analyze how context effectiveness changes ")
        f.write("as problem complexity increases.\n\n")
        f.write("2. **Metric Sensitivity**: Identify which evaluation metrics are most ")
        f.write("sensitive to context across different difficulties.\n\n")
        f.write("3. **Context Necessity**: Determine at which difficulty levels context ")
        f.write("becomes most critical for performance.\n\n")
        f.write("4. **Performance Patterns**: Observe consistent patterns or anomalies ")
        f.write("across the difficulty spectrum.\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the difficulty analysis:\n\n")
        f.write("- **Easy Problems**: Context may have minimal impact - suitable for baseline testing\n")
        f.write("- **Medium/Complex Problems**: Context likely provides significant benefits\n")
        f.write("- **Metric-Specific**: Focus context optimization on metrics showing highest sensitivity\n")
        f.write("- **Progressive Training**: Use difficulty progression for model training and evaluation\n\n")
        
        # Add file references section
        f.write("## Generated Files\n\n")
        f.write("This difficulty analysis generated the following files:\n\n")
        
        if os.path.exists(difficulty_chart_path):
            f.write("- `difficulty_context_analysis.png` - Comprehensive difficulty comparison chart\n")
        if os.path.exists(difficulty_heatmap_path):
            f.write("- `difficulty_context_heatmap.png` - Context impact heatmap by difficulty\n")
        
        f.write("- `difficulty_context_analysis_report.md` - This comprehensive analysis report\n")
        f.write("\n*All files are saved in the same directory as this report.*\n")
    
    print(f"📄 Difficulty analysis report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze context impact on multi-turn coding evaluation")
    parser.add_argument("--results_dir", default="results/context_comparison", 
                       help="Directory containing evaluation results")
    parser.add_argument("--output_format", choices=["console", "report", "both"], 
                       default="both", help="Output format")
    parser.add_argument("--difficulty_dirs", nargs="+", 
                       help="Multiple results directories for difficulty comparison (e.g., results/simple_test_easy results/simple_test_medium)")
    parser.add_argument("--difficulty_analysis", action="store_true",
                       help="Enable difficulty-based analysis across multiple directories")
    
    args = parser.parse_args()
    
    # Check if difficulty analysis is requested
    if args.difficulty_analysis or args.difficulty_dirs:
        if args.difficulty_dirs:
            difficulty_dirs = args.difficulty_dirs
        else:
            # Auto-detect difficulty directories
            base_dir = os.path.dirname(args.results_dir) or "results"
            difficulty_patterns = ["*easy*", "*simple*", "*medium*", "*complex*"]
            difficulty_dirs = []
            
            for pattern in difficulty_patterns:
                import glob
                matching_dirs = glob.glob(os.path.join(base_dir, pattern))
                difficulty_dirs.extend([d for d in matching_dirs if os.path.isdir(d)])
        
        if not difficulty_dirs:
            print("❌ No difficulty directories found for analysis")
            print("Use --difficulty_dirs to specify directories or ensure they exist")
            return
        
        print("🎯 Difficulty Analysis Mode")
        print("=" * 60)
        print(f"Analyzing {len(difficulty_dirs)} difficulty levels:")
        for d in difficulty_dirs:
            print(f"  - {d}")
        
        # Perform difficulty analysis
        difficulty_results = analyze_difficulty_impact(difficulty_dirs)
        
        if difficulty_results:
            # Create difficulty-specific visualizations
            output_dir = os.path.dirname(difficulty_dirs[0]) or "results"
            create_difficulty_visualizations(difficulty_results, output_dir)
            
            # Save difficulty analysis report
            save_difficulty_report(difficulty_results, output_dir)
        
    else:
        # Standard single-directory analysis
        if not os.path.exists(args.results_dir):
            print(f"❌ Results directory not found: {args.results_dir}")
            print("Run the context comparison evaluation first:")
            print("./run_context_comparison.sh")
            return
        
        analyze_context_impact(args.results_dir)


if __name__ == "__main__":
    main()