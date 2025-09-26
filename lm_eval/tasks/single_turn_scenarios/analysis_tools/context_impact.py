"""
Context impact analysis tool for single_turn_scenarios evaluation results.

This module analyzes performance differences across context modes and provides
statistical analysis of context effects on model performance.
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class ContextReport:
    """Container for context impact analysis results."""
    context_comparison: pd.DataFrame
    statistical_analysis: Dict[str, Any]
    effect_sizes: Dict[str, Dict[str, float]]
    context_rankings: Dict[str, List[str]]
    improvement_matrix: pd.DataFrame


class ContextAnalyzer:
    """
    Context impact analysis tool.
    
    Analyzes performance differences across context modes with statistical
    analysis and visualization of context effects.
    """
    
    def __init__(self, results_data: List[Dict[str, Any]]):
        """
        Initialize the context analyzer.
        
        Args:
            results_data: List of evaluation result dictionaries
        """
        self.results_data = results_data
        self.df = self._load_results_to_dataframe()
        
    def _load_results_to_dataframe(self) -> pd.DataFrame:
        """Convert results data to pandas DataFrame for analysis."""
        rows = []
        
        for result in self.results_data:
            row = {
                'id': result.get('id'),
                'model': result.get('model'),
                'config': result.get('config'),
                'scenario': result.get('scenario'),
                'difficulty': result.get('difficulty'),
                'language': result.get('language'),
                'context_mode': result.get('context_mode')
            }
            
            # Add all metrics
            metrics = result.get('metrics', {})
            for metric_name, value in metrics.items():
                row[f'metric_{metric_name}'] = value
                
            # Add runtime information
            runtime = result.get('runtime', {})
            row['runtime_time_s'] = runtime.get('time_s')
            row['runtime_peak_memory_mb'] = runtime.get('peak_memory_mb')
            
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def analyze_context_impact(self, 
                             models: Optional[List[str]] = None,
                             metrics: Optional[List[str]] = None) -> ContextReport:
        """
        Perform comprehensive context impact analysis.
        
        Args:
            models: List of model names to analyze (None for all)
            metrics: List of metrics to analyze (None for all)
            
        Returns:
            ContextReport with analysis results
        """
        if models is None:
            models = self.df['model'].unique().tolist()
        if metrics is None:
            metrics = [col for col in self.df.columns if col.startswith('metric_')]
            
        # Generate context comparison matrix
        context_comparison = self._generate_context_comparison(models, metrics)
        
        # Perform statistical analysis
        statistical_analysis = self._perform_context_statistical_tests(models, metrics)
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(models, metrics)
        
        # Generate context rankings
        context_rankings = self._generate_context_rankings(models, metrics)
        
        # Generate improvement matrix
        improvement_matrix = self._generate_improvement_matrix(models, metrics)
        
        return ContextReport(
            context_comparison=context_comparison,
            statistical_analysis=statistical_analysis,
            effect_sizes=effect_sizes,
            context_rankings=context_rankings,
            improvement_matrix=improvement_matrix
        )
    
    def _generate_context_comparison(self, 
                                   models: List[str], 
                                   metrics: List[str]) -> pd.DataFrame:
        """Generate comparison matrix showing performance by context mode."""
        comparison_data = []
        
        context_modes = self.df['context_mode'].unique()
        
        for model in models:
            for context_mode in context_modes:
                subset = self.df[
                    (self.df['model'] == model) & 
                    (self.df['context_mode'] == context_mode)
                ]
                
                if len(subset) > 0:
                    row = {
                        'model': model,
                        'context_mode': context_mode,
                        'sample_count': len(subset)
                    }
                    
                    for metric in metrics:
                        if metric in subset.columns:
                            values = subset[metric].dropna()
                            if len(values) > 0:
                                row[f"{metric.replace('metric_', '')}_mean"] = values.mean()
                                row[f"{metric.replace('metric_', '')}_std"] = values.std()
                                row[f"{metric.replace('metric_', '')}_median"] = values.median()
                            else:
                                row[f"{metric.replace('metric_', '')}_mean"] = np.nan
                                row[f"{metric.replace('metric_', '')}_std"] = np.nan
                                row[f"{metric.replace('metric_', '')}_median"] = np.nan
                    
                    comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def _perform_context_statistical_tests(self, 
                                         models: List[str], 
                                         metrics: List[str]) -> Dict[str, Any]:
        """Perform statistical tests to assess context impact."""
        tests = {}
        
        context_modes = self.df['context_mode'].unique()
        
        for model in models:
            tests[model] = {}
            model_data = self.df[self.df['model'] == model]
            
            for metric in metrics:
                metric_name = metric.replace('metric_', '')
                tests[model][metric_name] = {}
                
                # Get data for each context mode
                context_data = {}
                for context_mode in context_modes:
                    data = model_data[model_data['context_mode'] == context_mode][metric].dropna()
                    if len(data) > 0:
                        context_data[context_mode] = data.values
                
                # Perform pairwise comparisons between context modes
                pairwise_tests = {}
                context_list = list(context_data.keys())
                
                for i, context1 in enumerate(context_list):
                    for j, context2 in enumerate(context_list[i+1:], i+1):
                        if len(context_data[context1]) > 1 and len(context_data[context2]) > 1:
                            try:
                                # Paired t-test if same problems, otherwise independent
                                t_stat, p_value = stats.ttest_ind(
                                    context_data[context1], 
                                    context_data[context2]
                                )
                                
                                # Calculate effect size (Cohen's d)
                                pooled_std = np.sqrt(
                                    ((len(context_data[context1]) - 1) * np.var(context_data[context1], ddof=1) +
                                     (len(context_data[context2]) - 1) * np.var(context_data[context2], ddof=1)) /
                                    (len(context_data[context1]) + len(context_data[context2]) - 2)
                                )
                                
                                cohens_d = (np.mean(context_data[context1]) - np.mean(context_data[context2])) / pooled_std if pooled_std > 0 else 0
                                
                                pairwise_tests[f"{context1}_vs_{context2}"] = {
                                    't_statistic': t_stat,
                                    'p_value': p_value,
                                    'cohens_d': cohens_d,
                                    'significant': p_value < 0.05,
                                    'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
                                }
                            except Exception as e:
                                pairwise_tests[f"{context1}_vs_{context2}"] = {
                                    'error': str(e)
                                }
                
                tests[model][metric_name]['pairwise_tests'] = pairwise_tests
                
                # Perform ANOVA if more than 2 context modes
                if len(context_data) > 2:
                    try:
                        f_stat, p_value = stats.f_oneway(*context_data.values())
                        tests[model][metric_name]['anova'] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except Exception as e:
                        tests[model][metric_name]['anova'] = {'error': str(e)}
        
        return tests
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_effect_sizes(self, 
                              models: List[str], 
                              metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate effect sizes for context mode differences."""
        effect_sizes = {}
        
        context_modes = self.df['context_mode'].unique()
        baseline_context = 'no_context' if 'no_context' in context_modes else context_modes[0]
        
        for model in models:
            effect_sizes[model] = {}
            model_data = self.df[self.df['model'] == model]
            
            # Get baseline data
            baseline_data = model_data[model_data['context_mode'] == baseline_context]
            
            for context_mode in context_modes:
                if context_mode == baseline_context:
                    continue
                    
                context_data = model_data[model_data['context_mode'] == context_mode]
                effect_sizes[model][f"{baseline_context}_to_{context_mode}"] = {}
                
                for metric in metrics:
                    metric_name = metric.replace('metric_', '')
                    
                    baseline_values = baseline_data[metric].dropna()
                    context_values = context_data[metric].dropna()
                    
                    if len(baseline_values) > 0 and len(context_values) > 0:
                        # Calculate relative improvement
                        baseline_mean = baseline_values.mean()
                        context_mean = context_values.mean()
                        
                        if baseline_mean != 0:
                            relative_improvement = (context_mean - baseline_mean) / baseline_mean
                        else:
                            relative_improvement = 0
                        
                        effect_sizes[model][f"{baseline_context}_to_{context_mode}"][metric_name] = relative_improvement
        
        return effect_sizes
    
    def _generate_context_rankings(self, 
                                 models: List[str], 
                                 metrics: List[str]) -> Dict[str, List[str]]:
        """Generate context mode rankings for each model-metric combination."""
        rankings = {}
        
        context_modes = self.df['context_mode'].unique()
        
        for model in models:
            rankings[model] = {}
            model_data = self.df[self.df['model'] == model]
            
            for metric in metrics:
                metric_name = metric.replace('metric_', '')
                context_scores = []
                
                for context_mode in context_modes:
                    context_data = model_data[model_data['context_mode'] == context_mode]
                    if metric in context_data.columns:
                        values = context_data[metric].dropna()
                        if len(values) > 0:
                            context_scores.append((context_mode, values.mean()))
                
                # Sort by score (descending for most metrics)
                reverse_sort = not metric_name.endswith('_distance')
                context_scores.sort(key=lambda x: x[1], reverse=reverse_sort)
                
                rankings[model][metric_name] = [context for context, _ in context_scores]
        
        return rankings
    
    def _generate_improvement_matrix(self, 
                                   models: List[str], 
                                   metrics: List[str]) -> pd.DataFrame:
        """Generate matrix showing improvement from baseline context."""
        improvement_data = []
        
        context_modes = self.df['context_mode'].unique()
        baseline_context = 'no_context' if 'no_context' in context_modes else context_modes[0]
        
        for model in models:
            model_data = self.df[self.df['model'] == model]
            baseline_data = model_data[model_data['context_mode'] == baseline_context]
            
            for context_mode in context_modes:
                if context_mode == baseline_context:
                    continue
                    
                context_data = model_data[model_data['context_mode'] == context_mode]
                
                row = {
                    'model': model,
                    'context_mode': context_mode,
                    'baseline': baseline_context
                }
                
                for metric in metrics:
                    metric_name = metric.replace('metric_', '')
                    
                    baseline_values = baseline_data[metric].dropna()
                    context_values = context_data[metric].dropna()
                    
                    if len(baseline_values) > 0 and len(context_values) > 0:
                        baseline_mean = baseline_values.mean()
                        context_mean = context_values.mean()
                        
                        # Calculate absolute and relative improvement
                        absolute_improvement = context_mean - baseline_mean
                        if baseline_mean != 0:
                            relative_improvement = (context_mean - baseline_mean) / baseline_mean * 100
                        else:
                            relative_improvement = 0
                        
                        row[f"{metric_name}_abs_improvement"] = absolute_improvement
                        row[f"{metric_name}_rel_improvement"] = relative_improvement
                    else:
                        row[f"{metric_name}_abs_improvement"] = np.nan
                        row[f"{metric_name}_rel_improvement"] = np.nan
                
                improvement_data.append(row)
        
        return pd.DataFrame(improvement_data)
    
    def create_context_heatmap(self, 
                             models: List[str], 
                             metrics: List[str],
                             save_path: Optional[str] = None) -> plt.Figure:
        """Create heatmap showing context impact across models and metrics."""
        # Generate improvement matrix
        improvement_matrix = self._generate_improvement_matrix(models, metrics)
        
        # Prepare data for heatmap
        heatmap_data = []
        context_modes = improvement_matrix['context_mode'].unique()
        
        for model in models:
            for context_mode in context_modes:
                subset = improvement_matrix[
                    (improvement_matrix['model'] == model) & 
                    (improvement_matrix['context_mode'] == context_mode)
                ]
                
                if len(subset) > 0:
                    row = {'model': model, 'context_mode': context_mode}
                    for metric in metrics:
                        metric_name = metric.replace('metric_', '')
                        col_name = f"{metric_name}_rel_improvement"
                        if col_name in subset.columns:
                            row[metric_name] = subset[col_name].iloc[0]
                        else:
                            row[metric_name] = 0
                    heatmap_data.append(row)
        
        if not heatmap_data:
            return None
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df['model_context'] = heatmap_df['model'] + '_' + heatmap_df['context_mode']
        heatmap_df = heatmap_df.set_index('model_context')
        
        # Remove non-metric columns
        metric_cols = [m.replace('metric_', '') for m in metrics]
        heatmap_df = heatmap_df[metric_cols]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Relative Improvement (%)'}
        )
        
        plt.title('Context Impact Heatmap\n(Relative Improvement from Baseline)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Model + Context Mode', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_context_comparison_plot(self, 
                                     model: str, 
                                     metric: str,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """Create box plot comparing context modes for a specific model and metric."""
        model_data = self.df[self.df['model'] == model]
        
        if metric not in model_data.columns:
            raise ValueError(f"Metric {metric} not found in data")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create box plot
        context_modes = model_data['context_mode'].unique()
        data_for_plot = []
        labels = []
        
        for context_mode in context_modes:
            context_data = model_data[model_data['context_mode'] == context_mode]
            values = context_data[metric].dropna()
            if len(values) > 0:
                data_for_plot.append(values)
                labels.append(context_mode)
        
        if data_for_plot:
            bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
        
        ax.set_title(f'Context Mode Comparison: {model}\nMetric: {metric.replace("metric_", "")}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Context Mode', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_results(self, 
                      report: ContextReport, 
                      output_dir: str) -> None:
        """Export context analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export context comparison
        report.context_comparison.to_csv(
            output_path / 'context_comparison.csv', index=False
        )
        
        # Export statistical analysis
        with open(output_path / 'context_statistical_analysis.json', 'w') as f:
            json.dump(report.statistical_analysis, f, indent=2, default=str)
        
        # Export effect sizes
        with open(output_path / 'context_effect_sizes.json', 'w') as f:
            json.dump(report.effect_sizes, f, indent=2, default=str)
        
        # Export context rankings
        with open(output_path / 'context_rankings.json', 'w') as f:
            json.dump(report.context_rankings, f, indent=2)
        
        # Export improvement matrix
        report.improvement_matrix.to_csv(
            output_path / 'context_improvement_matrix.csv', index=False
        )


def main():
    """Example usage of ContextAnalyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze context impact on model performance')
    parser.add_argument('--results', required=True, 
                       help='Path to results JSON file')
    parser.add_argument('--output', required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--models', nargs='+',
                       help='Models to analyze (default: all)')
    parser.add_argument('--metrics', nargs='+', 
                       help='Metrics to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results, 'r') as f:
        results_data = [json.loads(line) for line in f]
    
    # Create analyzer and run analysis
    analyzer = ContextAnalyzer(results_data)
    report = analyzer.analyze_context_impact(
        models=args.models,
        metrics=args.metrics
    )
    
    # Export results
    analyzer.export_results(report, args.output)
    
    # Create visualizations
    if args.models and args.metrics:
        # Create heatmap
        fig = analyzer.create_context_heatmap(
            args.models, 
            [f'metric_{m}' for m in args.metrics],
            save_path=f"{args.output}/context_heatmap.png"
        )
        
        # Create comparison plots for each model
        for model in args.models:
            for metric in args.metrics:
                try:
                    fig = analyzer.create_context_comparison_plot(
                        model, 
                        f'metric_{metric}',
                        save_path=f"{args.output}/context_comparison_{model}_{metric}.png"
                    )
                except Exception as e:
                    print(f"Could not create plot for {model}-{metric}: {e}")
    
    print(f"Context analysis complete. Results saved to {args.output}")


if __name__ == '__main__':
    main()