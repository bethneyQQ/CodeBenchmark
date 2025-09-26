"""
Model comparison analysis tool for single_turn_scenarios evaluation results.

This module provides horizontal performance comparison across all metrics,
statistical significance testing, and performance matrix generation.
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
class ComparisonReport:
    """Container for model comparison analysis results."""
    performance_matrix: pd.DataFrame
    statistical_tests: Dict[str, Any]
    rankings: Dict[str, List[str]]
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]]
    summary_stats: Dict[str, Any]


class ModelComparator:
    """
    Comprehensive model comparison analysis tool.
    
    Provides horizontal performance comparison across all metrics with
    statistical significance testing and ranking systems.
    """
    
    def __init__(self, results_data: List[Dict[str, Any]]):
        """
        Initialize the model comparator.
        
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
    
    def compare_models(self, 
                      models: Optional[List[str]] = None,
                      metrics: Optional[List[str]] = None) -> ComparisonReport:
        """
        Perform comprehensive model comparison analysis.
        
        Args:
            models: List of model names to compare (None for all)
            metrics: List of metrics to analyze (None for all)
            
        Returns:
            ComparisonReport with analysis results
        """
        if models is None:
            models = self.df['model'].unique().tolist()
        if metrics is None:
            metrics = [col for col in self.df.columns if col.startswith('metric_')]
            
        # Generate performance matrix
        performance_matrix = self._generate_performance_matrix(models, metrics)
        
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(models, metrics)
        
        # Generate rankings
        rankings = self._generate_rankings(models, metrics)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(models, metrics)
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats(models, metrics)
        
        return ComparisonReport(
            performance_matrix=performance_matrix,
            statistical_tests=statistical_tests,
            rankings=rankings,
            confidence_intervals=confidence_intervals,
            summary_stats=summary_stats
        )
    
    def _generate_performance_matrix(self, 
                                   models: List[str], 
                                   metrics: List[str]) -> pd.DataFrame:
        """Generate performance matrix with mean scores for each model-metric combination."""
        matrix_data = []
        
        for model in models:
            model_data = self.df[self.df['model'] == model]
            row = {'model': model}
            
            for metric in metrics:
                if metric in model_data.columns:
                    values = model_data[metric].dropna()
                    if len(values) > 0:
                        row[metric.replace('metric_', '')] = values.mean()
                    else:
                        row[metric.replace('metric_', '')] = np.nan
                else:
                    row[metric.replace('metric_', '')] = np.nan
                    
            matrix_data.append(row)
            
        return pd.DataFrame(matrix_data).set_index('model')
    
    def _perform_statistical_tests(self, 
                                 models: List[str], 
                                 metrics: List[str]) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        tests = {}
        
        for metric in metrics:
            metric_name = metric.replace('metric_', '')
            tests[metric_name] = {}
            
            # Get data for each model
            model_data = {}
            for model in models:
                data = self.df[self.df['model'] == model][metric].dropna()
                if len(data) > 0:
                    model_data[model] = data.values
            
            # Perform pairwise t-tests
            pairwise_tests = {}
            model_list = list(model_data.keys())
            
            for i, model1 in enumerate(model_list):
                for j, model2 in enumerate(model_list[i+1:], i+1):
                    if len(model_data[model1]) > 1 and len(model_data[model2]) > 1:
                        try:
                            t_stat, p_value = stats.ttest_ind(
                                model_data[model1], 
                                model_data[model2]
                            )
                            pairwise_tests[f"{model1}_vs_{model2}"] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                        except Exception as e:
                            pairwise_tests[f"{model1}_vs_{model2}"] = {
                                'error': str(e)
                            }
            
            tests[metric_name]['pairwise_tests'] = pairwise_tests
            
            # Perform ANOVA if more than 2 models
            if len(model_data) > 2:
                try:
                    f_stat, p_value = stats.f_oneway(*model_data.values())
                    tests[metric_name]['anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    tests[metric_name]['anova'] = {'error': str(e)}
        
        return tests
    
    def _generate_rankings(self, 
                         models: List[str], 
                         metrics: List[str]) -> Dict[str, List[str]]:
        """Generate model rankings for each metric."""
        rankings = {}
        
        for metric in metrics:
            metric_name = metric.replace('metric_', '')
            model_scores = []
            
            for model in models:
                model_data = self.df[self.df['model'] == model]
                if metric in model_data.columns:
                    values = model_data[metric].dropna()
                    if len(values) > 0:
                        model_scores.append((model, values.mean()))
            
            # Sort by score (descending for most metrics)
            # Note: Some metrics like edit_distance might need ascending order
            reverse_sort = not metric_name.endswith('_distance')
            model_scores.sort(key=lambda x: x[1], reverse=reverse_sort)
            
            rankings[metric_name] = [model for model, _ in model_scores]
        
        return rankings
    
    def _calculate_confidence_intervals(self, 
                                     models: List[str], 
                                     metrics: List[str],
                                     confidence: float = 0.95) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate confidence intervals for each model-metric combination."""
        intervals = {}
        alpha = 1 - confidence
        
        for metric in metrics:
            metric_name = metric.replace('metric_', '')
            intervals[metric_name] = {}
            
            for model in models:
                model_data = self.df[self.df['model'] == model]
                if metric in model_data.columns:
                    values = model_data[metric].dropna()
                    if len(values) > 1:
                        mean = values.mean()
                        sem = stats.sem(values)
                        h = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
                        intervals[metric_name][model] = (mean - h, mean + h)
                    elif len(values) == 1:
                        intervals[metric_name][model] = (values.iloc[0], values.iloc[0])
        
        return intervals
    
    def _generate_summary_stats(self, 
                              models: List[str], 
                              metrics: List[str]) -> Dict[str, Any]:
        """Generate summary statistics across models and metrics."""
        summary = {
            'total_evaluations': len(self.df),
            'models_analyzed': len(models),
            'metrics_analyzed': len(metrics),
            'scenarios_covered': self.df['scenario'].nunique(),
            'languages_covered': self.df['language'].nunique(),
            'difficulty_levels': self.df['difficulty'].nunique()
        }
        
        # Overall best performing model per metric
        best_models = {}
        for metric in metrics:
            metric_name = metric.replace('metric_', '')
            model_means = self.df.groupby('model')[metric].mean()
            if len(model_means) > 0:
                # Assume higher is better for most metrics
                best_models[metric_name] = model_means.idxmax()
        
        summary['best_models_per_metric'] = best_models
        
        return summary
    
    def create_radar_chart(self, 
                          models: List[str], 
                          metrics: List[str],
                          save_path: Optional[str] = None) -> plt.Figure:
        """Create radar chart comparing models across metrics."""
        # Prepare data
        performance_matrix = self._generate_performance_matrix(models, metrics)
        
        # Normalize metrics to 0-1 scale for radar chart
        normalized_matrix = performance_matrix.copy()
        for col in normalized_matrix.columns:
            col_min = normalized_matrix[col].min()
            col_max = normalized_matrix[col].max()
            if col_max > col_min:
                normalized_matrix[col] = (normalized_matrix[col] - col_min) / (col_max - col_min)
            else:
                normalized_matrix[col] = 0.5  # Default to middle if no variation
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Set up angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            if model in normalized_matrix.index:
                values = normalized_matrix.loc[model].values.tolist()
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=model, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('metric_', '') for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Model Performance Comparison\n(Normalized Metrics)', 
                 size=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_results(self, 
                      report: ComparisonReport, 
                      output_dir: str) -> None:
        """Export comparison results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export performance matrix
        report.performance_matrix.to_csv(
            output_path / 'performance_matrix.csv'
        )
        
        # Export statistical tests
        with open(output_path / 'statistical_tests.json', 'w') as f:
            json.dump(report.statistical_tests, f, indent=2, default=str)
        
        # Export rankings
        with open(output_path / 'rankings.json', 'w') as f:
            json.dump(report.rankings, f, indent=2)
        
        # Export confidence intervals
        with open(output_path / 'confidence_intervals.json', 'w') as f:
            json.dump(report.confidence_intervals, f, indent=2, default=str)
        
        # Export summary stats
        with open(output_path / 'summary_stats.json', 'w') as f:
            json.dump(report.summary_stats, f, indent=2, default=str)


def main():
    """Example usage of ModelComparator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare model performance')
    parser.add_argument('--results', required=True, 
                       help='Path to results JSON file')
    parser.add_argument('--output', required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--models', nargs='+',
                       help='Models to compare (default: all)')
    parser.add_argument('--metrics', nargs='+', 
                       help='Metrics to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results, 'r') as f:
        results_data = [json.loads(line) for line in f]
    
    # Create comparator and run analysis
    comparator = ModelComparator(results_data)
    report = comparator.compare_models(
        models=args.models,
        metrics=args.metrics
    )
    
    # Export results
    comparator.export_results(report, args.output)
    
    # Create radar chart
    if args.models and args.metrics:
        fig = comparator.create_radar_chart(
            args.models, 
            [f'metric_{m}' for m in args.metrics],
            save_path=f"{args.output}/radar_chart.png"
        )
    
    print(f"Analysis complete. Results saved to {args.output}")


if __name__ == '__main__':
    main()