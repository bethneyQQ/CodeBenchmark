"""
Scenario and difficulty analysis tool for single_turn_scenarios evaluation results.

This module provides analysis of model performance by scenario type, difficulty levels,
and programming language support with detailed adaptation analysis.
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
class ScenarioAnalysisReport:
    """Container for scenario and difficulty analysis results."""
    scenario_performance: pd.DataFrame
    difficulty_analysis: pd.DataFrame
    language_adaptation: pd.DataFrame
    scenario_rankings: Dict[str, Dict[str, List[str]]]
    difficulty_sensitivity: Dict[str, Dict[str, float]]
    language_preferences: Dict[str, Dict[str, float]]
    cross_analysis: Dict[str, Any]


class ScenarioAnalyzer:
    """
    Scenario and difficulty analysis tool.
    
    Analyzes model performance across scenario types, difficulty levels,
    and programming languages with detailed adaptation metrics.
    """
    
    def __init__(self, results_data: List[Dict[str, Any]]):
        """
        Initialize the scenario analyzer.
        
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
    
    def analyze_scenarios_and_difficulty(self, 
                                       models: Optional[List[str]] = None,
                                       metrics: Optional[List[str]] = None) -> ScenarioAnalysisReport:
        """
        Perform comprehensive scenario and difficulty analysis.
        
        Args:
            models: List of model names to analyze (None for all)
            metrics: List of metrics to analyze (None for all)
            
        Returns:
            ScenarioAnalysisReport with analysis results
        """
        if models is None:
            models = self.df['model'].unique().tolist()
        if metrics is None:
            metrics = [col for col in self.df.columns if col.startswith('metric_')]
            
        # Analyze scenario performance
        scenario_performance = self._analyze_scenario_performance(models, metrics)
        
        # Analyze difficulty sensitivity
        difficulty_analysis = self._analyze_difficulty_sensitivity(models, metrics)
        
        # Analyze language adaptation
        language_adaptation = self._analyze_language_adaptation(models, metrics)
        
        # Generate scenario rankings
        scenario_rankings = self._generate_scenario_rankings(models, metrics)
        
        # Calculate difficulty sensitivity scores
        difficulty_sensitivity = self._calculate_difficulty_sensitivity(models, metrics)
        
        # Calculate language preferences
        language_preferences = self._calculate_language_preferences(models, metrics)
        
        # Perform cross-dimensional analysis
        cross_analysis = self._perform_cross_analysis(models, metrics)
        
        return ScenarioAnalysisReport(
            scenario_performance=scenario_performance,
            difficulty_analysis=difficulty_analysis,
            language_adaptation=language_adaptation,
            scenario_rankings=scenario_rankings,
            difficulty_sensitivity=difficulty_sensitivity,
            language_preferences=language_preferences,
            cross_analysis=cross_analysis
        )
    
    def _analyze_scenario_performance(self, 
                                    models: List[str], 
                                    metrics: List[str]) -> pd.DataFrame:
        """Analyze model performance across different scenarios."""
        performance_data = []
        
        scenarios = self.df['scenario'].unique()
        
        for model in models:
            model_data = self.df[self.df['model'] == model]
            
            for scenario in scenarios:
                scenario_data = model_data[model_data['scenario'] == scenario]
                
                if len(scenario_data) > 0:
                    row = {
                        'model': model,
                        'scenario': scenario,
                        'sample_count': len(scenario_data),
                        'scenario_type': self._classify_scenario_type(scenario)
                    }
                    
                    for metric in metrics:
                        if metric in scenario_data.columns:
                            values = scenario_data[metric].dropna()
                            if len(values) > 0:
                                row[f"{metric.replace('metric_', '')}_mean"] = values.mean()
                                row[f"{metric.replace('metric_', '')}_std"] = values.std()
                                row[f"{metric.replace('metric_', '')}_median"] = values.median()
                                row[f"{metric.replace('metric_', '')}_q75"] = values.quantile(0.75)
                                row[f"{metric.replace('metric_', '')}_q25"] = values.quantile(0.25)
                            else:
                                for stat in ['mean', 'std', 'median', 'q75', 'q25']:
                                    row[f"{metric.replace('metric_', '')}_{stat}"] = np.nan
                    
                    performance_data.append(row)
        
        return pd.DataFrame(performance_data)
    
    def _classify_scenario_type(self, scenario: str) -> str:
        """Classify scenario into basic, advanced, or comprehensive categories."""
        basic_scenarios = ['code_completion', 'bug_fix', 'code_translation', 'documentation', 'function_generation']
        advanced_scenarios = ['system_design', 'algorithm_implementation', 'api_design', 'database_design', 'performance_optimization']
        comprehensive_scenarios = ['full_stack', 'testing_strategy', 'security']
        
        if scenario in basic_scenarios:
            return 'basic'
        elif scenario in advanced_scenarios:
            return 'advanced'
        elif scenario in comprehensive_scenarios:
            return 'comprehensive'
        else:
            return 'unknown'
    
    def _analyze_difficulty_sensitivity(self, 
                                      models: List[str], 
                                      metrics: List[str]) -> pd.DataFrame:
        """Analyze how model performance changes with difficulty levels."""
        difficulty_data = []
        
        difficulties = self.df['difficulty'].unique()
        difficulty_order = ['simple', 'intermediate', 'complex']
        
        # Sort difficulties by expected order
        sorted_difficulties = [d for d in difficulty_order if d in difficulties]
        sorted_difficulties.extend([d for d in difficulties if d not in difficulty_order])
        
        for model in models:
            model_data = self.df[self.df['model'] == model]
            
            for difficulty in sorted_difficulties:
                difficulty_data_subset = model_data[model_data['difficulty'] == difficulty]
                
                if len(difficulty_data_subset) > 0:
                    row = {
                        'model': model,
                        'difficulty': difficulty,
                        'difficulty_order': difficulty_order.index(difficulty) if difficulty in difficulty_order else len(difficulty_order),
                        'sample_count': len(difficulty_data_subset)
                    }
                    
                    for metric in metrics:
                        if metric in difficulty_data_subset.columns:
                            values = difficulty_data_subset[metric].dropna()
                            if len(values) > 0:
                                row[f"{metric.replace('metric_', '')}_mean"] = values.mean()
                                row[f"{metric.replace('metric_', '')}_std"] = values.std()
                                
                                # Calculate performance drop from previous difficulty
                                if difficulty_order.index(difficulty) > 0:
                                    prev_difficulty = difficulty_order[difficulty_order.index(difficulty) - 1]
                                    prev_data = model_data[model_data['difficulty'] == prev_difficulty]
                                    if len(prev_data) > 0:
                                        prev_values = prev_data[metric].dropna()
                                        if len(prev_values) > 0:
                                            prev_mean = prev_values.mean()
                                            current_mean = values.mean()
                                            drop = (prev_mean - current_mean) / prev_mean if prev_mean != 0 else 0
                                            row[f"{metric.replace('metric_', '')}_difficulty_drop"] = drop
                            else:
                                row[f"{metric.replace('metric_', '')}_mean"] = np.nan
                                row[f"{metric.replace('metric_', '')}_std"] = np.nan
                    
                    difficulty_data.append(row)
        
        return pd.DataFrame(difficulty_data)
    
    def _analyze_language_adaptation(self, 
                                   models: List[str], 
                                   metrics: List[str]) -> pd.DataFrame:
        """Analyze model performance across programming languages."""
        language_data = []
        
        languages = self.df['language'].unique()
        
        for model in models:
            model_data = self.df[self.df['model'] == model]
            
            for language in languages:
                language_subset = model_data[model_data['language'] == language]
                
                if len(language_subset) > 0:
                    row = {
                        'model': model,
                        'language': language,
                        'sample_count': len(language_subset),
                        'language_family': self._classify_language_family(language)
                    }
                    
                    for metric in metrics:
                        if metric in language_subset.columns:
                            values = language_subset[metric].dropna()
                            if len(values) > 0:
                                row[f"{metric.replace('metric_', '')}_mean"] = values.mean()
                                row[f"{metric.replace('metric_', '')}_std"] = values.std()
                                row[f"{metric.replace('metric_', '')}_median"] = values.median()
                                
                                # Calculate relative performance vs model average
                                model_avg = model_data[metric].dropna().mean()
                                if model_avg != 0:
                                    relative_perf = (values.mean() - model_avg) / model_avg
                                    row[f"{metric.replace('metric_', '')}_relative_performance"] = relative_perf
                            else:
                                for stat in ['mean', 'std', 'median', 'relative_performance']:
                                    row[f"{metric.replace('metric_', '')}_{stat}"] = np.nan
                    
                    language_data.append(row)
        
        return pd.DataFrame(language_data)
    
    def _classify_language_family(self, language: str) -> str:
        """Classify programming language into families."""
        families = {
            'python': 'interpreted',
            'javascript': 'interpreted',
            'typescript': 'interpreted',
            'java': 'compiled_vm',
            'c++': 'compiled_native',
            'cpp': 'compiled_native',
            'go': 'compiled_native',
            'rust': 'compiled_native'
        }
        return families.get(language.lower(), 'unknown')
    
    def _generate_scenario_rankings(self, 
                                  models: List[str], 
                                  metrics: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """Generate rankings for each scenario-metric combination."""
        rankings = {}
        
        scenarios = self.df['scenario'].unique()
        
        for scenario in scenarios:
            rankings[scenario] = {}
            scenario_data = self.df[self.df['scenario'] == scenario]
            
            for metric in metrics:
                metric_name = metric.replace('metric_', '')
                model_scores = []
                
                for model in models:
                    model_scenario_data = scenario_data[scenario_data['model'] == model]
                    if metric in model_scenario_data.columns:
                        values = model_scenario_data[metric].dropna()
                        if len(values) > 0:
                            model_scores.append((model, values.mean()))
                
                # Sort by score (descending for most metrics)
                reverse_sort = not metric_name.endswith('_distance')
                model_scores.sort(key=lambda x: x[1], reverse=reverse_sort)
                
                rankings[scenario][metric_name] = [model for model, _ in model_scores]
        
        return rankings
    
    def _calculate_difficulty_sensitivity(self, 
                                        models: List[str], 
                                        metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate difficulty sensitivity scores for each model."""
        sensitivity = {}
        
        difficulty_order = ['simple', 'intermediate', 'complex']
        
        for model in models:
            sensitivity[model] = {}
            model_data = self.df[self.df['model'] == model]
            
            for metric in metrics:
                metric_name = metric.replace('metric_', '')
                
                # Calculate performance across difficulties
                difficulty_scores = []
                for difficulty in difficulty_order:
                    difficulty_data = model_data[model_data['difficulty'] == difficulty]
                    if metric in difficulty_data.columns:
                        values = difficulty_data[metric].dropna()
                        if len(values) > 0:
                            difficulty_scores.append(values.mean())
                
                if len(difficulty_scores) >= 2:
                    # Calculate slope of performance decline
                    x = np.arange(len(difficulty_scores))
                    slope, _, r_value, _, _ = stats.linregress(x, difficulty_scores)
                    
                    # Negative slope indicates performance decline with difficulty
                    # Convert to positive sensitivity score
                    sensitivity_score = -slope if slope < 0 else 0
                    
                    sensitivity[model][metric_name] = {
                        'sensitivity_score': sensitivity_score,
                        'r_squared': r_value ** 2,
                        'performance_decline': difficulty_scores[0] - difficulty_scores[-1] if len(difficulty_scores) > 1 else 0
                    }
                else:
                    sensitivity[model][metric_name] = {
                        'sensitivity_score': 0,
                        'r_squared': 0,
                        'performance_decline': 0
                    }
        
        return sensitivity
    
    def _calculate_language_preferences(self, 
                                      models: List[str], 
                                      metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate language preference scores for each model."""
        preferences = {}
        
        languages = self.df['language'].unique()
        
        for model in models:
            preferences[model] = {}
            model_data = self.df[self.df['model'] == model]
            
            # Calculate overall model performance
            model_overall = {}
            for metric in metrics:
                if metric in model_data.columns:
                    values = model_data[metric].dropna()
                    if len(values) > 0:
                        model_overall[metric] = values.mean()
            
            # Calculate language-specific performance
            for language in languages:
                language_data = model_data[model_data['language'] == language]
                language_score = 0
                metric_count = 0
                
                for metric in metrics:
                    if metric in language_data.columns and metric in model_overall:
                        values = language_data[metric].dropna()
                        if len(values) > 0 and model_overall[metric] != 0:
                            relative_perf = values.mean() / model_overall[metric]
                            language_score += relative_perf
                            metric_count += 1
                
                if metric_count > 0:
                    preferences[model][language] = language_score / metric_count
                else:
                    preferences[model][language] = 1.0  # Neutral if no data
        
        return preferences
    
    def _perform_cross_analysis(self, 
                              models: List[str], 
                              metrics: List[str]) -> Dict[str, Any]:
        """Perform cross-dimensional analysis (scenario × difficulty × language)."""
        cross_analysis = {}
        
        # Scenario-Difficulty interaction
        scenario_difficulty = self.df.groupby(['model', 'scenario', 'difficulty']).agg({
            metric: 'mean' for metric in metrics if metric in self.df.columns
        }).reset_index()
        
        cross_analysis['scenario_difficulty_interaction'] = scenario_difficulty.to_dict('records')
        
        # Language-Difficulty interaction
        language_difficulty = self.df.groupby(['model', 'language', 'difficulty']).agg({
            metric: 'mean' for metric in metrics if metric in self.df.columns
        }).reset_index()
        
        cross_analysis['language_difficulty_interaction'] = language_difficulty.to_dict('records')
        
        # Scenario-Language interaction
        scenario_language = self.df.groupby(['model', 'scenario', 'language']).agg({
            metric: 'mean' for metric in metrics if metric in self.df.columns
        }).reset_index()
        
        cross_analysis['scenario_language_interaction'] = scenario_language.to_dict('records')
        
        return cross_analysis
    
    def create_scenario_performance_chart(self, 
                                        models: List[str], 
                                        metric: str,
                                        save_path: Optional[str] = None) -> plt.Figure:
        """Create chart showing performance across scenarios."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        scenarios = self.df['scenario'].unique()
        x_pos = np.arange(len(scenarios))
        width = 0.8 / len(models)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_scores = []
            model_errors = []
            
            for scenario in scenarios:
                scenario_data = self.df[
                    (self.df['model'] == model) & 
                    (self.df['scenario'] == scenario)
                ]
                
                if metric in scenario_data.columns:
                    values = scenario_data[metric].dropna()
                    if len(values) > 0:
                        model_scores.append(values.mean())
                        model_errors.append(values.std() / np.sqrt(len(values)))  # SEM
                    else:
                        model_scores.append(0)
                        model_errors.append(0)
                else:
                    model_scores.append(0)
                    model_errors.append(0)
            
            ax.bar(x_pos + i * width, model_scores, width, 
                  label=model, color=colors[i], alpha=0.8,
                  yerr=model_errors, capsize=3)
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel(f'{metric.replace("metric_", "")} Score')
        ax.set_title(f'Model Performance by Scenario: {metric.replace("metric_", "")}')
        ax.set_xticks(x_pos + width * (len(models) - 1) / 2)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_difficulty_sensitivity_chart(self, 
                                          models: List[str], 
                                          metric: str,
                                          save_path: Optional[str] = None) -> plt.Figure:
        """Create chart showing difficulty sensitivity."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        difficulty_order = ['simple', 'intermediate', 'complex']
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_data = self.df[self.df['model'] == model]
            
            difficulty_scores = []
            difficulty_errors = []
            
            for difficulty in difficulty_order:
                difficulty_data = model_data[model_data['difficulty'] == difficulty]
                if metric in difficulty_data.columns:
                    values = difficulty_data[metric].dropna()
                    if len(values) > 0:
                        difficulty_scores.append(values.mean())
                        difficulty_errors.append(values.std() / np.sqrt(len(values)))
                    else:
                        difficulty_scores.append(np.nan)
                        difficulty_errors.append(0)
                else:
                    difficulty_scores.append(np.nan)
                    difficulty_errors.append(0)
            
            # Filter out NaN values
            valid_indices = [i for i, score in enumerate(difficulty_scores) if not np.isnan(score)]
            valid_difficulties = [difficulty_order[i] for i in valid_indices]
            valid_scores = [difficulty_scores[i] for i in valid_indices]
            valid_errors = [difficulty_errors[i] for i in valid_indices]
            
            if valid_scores:
                ax.errorbar(valid_difficulties, valid_scores, yerr=valid_errors,
                           marker='o', linewidth=2, markersize=8, 
                           label=model, color=colors[i], capsize=5)
        
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel(f'{metric.replace("metric_", "")} Score')
        ax.set_title(f'Difficulty Sensitivity: {metric.replace("metric_", "")}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_language_adaptation_chart(self, 
                                       models: List[str], 
                                       metric: str,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """Create chart showing language adaptation."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        languages = self.df['language'].unique()
        x_pos = np.arange(len(languages))
        width = 0.8 / len(models)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_scores = []
            model_errors = []
            
            for language in languages:
                language_data = self.df[
                    (self.df['model'] == model) & 
                    (self.df['language'] == language)
                ]
                
                if metric in language_data.columns:
                    values = language_data[metric].dropna()
                    if len(values) > 0:
                        model_scores.append(values.mean())
                        model_errors.append(values.std() / np.sqrt(len(values)))
                    else:
                        model_scores.append(0)
                        model_errors.append(0)
                else:
                    model_scores.append(0)
                    model_errors.append(0)
            
            ax.bar(x_pos + i * width, model_scores, width, 
                  label=model, color=colors[i], alpha=0.8,
                  yerr=model_errors, capsize=3)
        
        ax.set_xlabel('Programming Language')
        ax.set_ylabel(f'{metric.replace("metric_", "")} Score')
        ax.set_title(f'Language Adaptation: {metric.replace("metric_", "")}')
        ax.set_xticks(x_pos + width * (len(models) - 1) / 2)
        ax.set_xticklabels(languages, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_results(self, 
                      report: ScenarioAnalysisReport, 
                      output_dir: str) -> None:
        """Export scenario analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export scenario performance
        report.scenario_performance.to_csv(
            output_path / 'scenario_performance.csv', index=False
        )
        
        # Export difficulty analysis
        report.difficulty_analysis.to_csv(
            output_path / 'difficulty_analysis.csv', index=False
        )
        
        # Export language adaptation
        report.language_adaptation.to_csv(
            output_path / 'language_adaptation.csv', index=False
        )
        
        # Export scenario rankings
        with open(output_path / 'scenario_rankings.json', 'w') as f:
            json.dump(report.scenario_rankings, f, indent=2)
        
        # Export difficulty sensitivity
        with open(output_path / 'difficulty_sensitivity.json', 'w') as f:
            json.dump(report.difficulty_sensitivity, f, indent=2, default=str)
        
        # Export language preferences
        with open(output_path / 'language_preferences.json', 'w') as f:
            json.dump(report.language_preferences, f, indent=2, default=str)
        
        # Export cross analysis
        with open(output_path / 'cross_analysis.json', 'w') as f:
            json.dump(report.cross_analysis, f, indent=2, default=str)


def main():
    """Example usage of ScenarioAnalyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze scenario and difficulty performance')
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
    analyzer = ScenarioAnalyzer(results_data)
    report = analyzer.analyze_scenarios_and_difficulty(
        models=args.models,
        metrics=args.metrics
    )
    
    # Export results
    analyzer.export_results(report, args.output)
    
    # Create visualizations
    if args.models and args.metrics:
        for metric in args.metrics:
            metric_name = f'metric_{metric}'
            
            # Scenario performance chart
            try:
                fig = analyzer.create_scenario_performance_chart(
                    args.models, 
                    metric_name,
                    save_path=f"{args.output}/scenario_performance_{metric}.png"
                )
                plt.close(fig)
            except Exception as e:
                print(f"Could not create scenario chart for {metric}: {e}")
            
            # Difficulty sensitivity chart
            try:
                fig = analyzer.create_difficulty_sensitivity_chart(
                    args.models, 
                    metric_name,
                    save_path=f"{args.output}/difficulty_sensitivity_{metric}.png"
                )
                plt.close(fig)
            except Exception as e:
                print(f"Could not create difficulty chart for {metric}: {e}")
            
            # Language adaptation chart
            try:
                fig = analyzer.create_language_adaptation_chart(
                    args.models, 
                    metric_name,
                    save_path=f"{args.output}/language_adaptation_{metric}.png"
                )
                plt.close(fig)
            except Exception as e:
                print(f"Could not create language chart for {metric}: {e}")
    
    print(f"Scenario analysis complete. Results saved to {args.output}")


if __name__ == '__main__':
    main()