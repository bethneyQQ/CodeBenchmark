# Analysis Tools for Single Turn Scenarios

This directory contains comprehensive analysis and visualization tools for single_turn_scenarios evaluation results. The tools provide model comparison, context impact analysis, scenario performance analysis, and comprehensive reporting capabilities.

## Overview

The analysis tools are designed to work with evaluation results in JSONL format, where each line contains a complete evaluation result with metrics, runtime information, and metadata.

### Available Tools

1. **ModelComparator** (`compare_models.py`) - Horizontal performance comparison across all metrics
2. **ContextAnalyzer** (`context_impact.py`) - Analysis of context mode impact on performance  
3. **ScenarioAnalyzer** (`scenario_analysis.py`) - Scenario and difficulty performance analysis
4. **ReportGenerator** (`generate_report.py`) - Comprehensive HTML, CSV, and SVG report generation

## Quick Start

### Run Complete Analysis

The easiest way to analyze your results is using the comprehensive analysis runner:

```bash
# Analyze all models and metrics
python -m lm_eval.tasks.single_turn_scenarios.analysis_tools.run_analysis \
  --results evaluation_results.jsonl \
  --output analysis_output/

# Analyze specific models
python -m lm_eval.tasks.single_turn_scenarios.analysis_tools.run_analysis \
  --results evaluation_results.jsonl \
  --output analysis_output/ \
  --models claude-3-5-sonnet deepseek-coder openai-gpt-4

# Analyze specific metrics
python -m lm_eval.tasks.single_turn_scenarios.analysis_tools.run_analysis \
  --results evaluation_results.jsonl \
  --output analysis_output/ \
  --metrics exact_match pass_at_1 codebleu syntax_valid
```

This will generate:
- Model comparison analysis with statistical tests
- Context impact analysis with heatmaps
- Scenario and difficulty performance analysis
- Comprehensive HTML report
- CSV data exports
- Summary dashboard with SVG visualizations

### Individual Tool Usage

Each tool can also be used independently:

```bash
# Model comparison
python -m lm_eval.tasks.single_turn_scenarios.analysis_tools.compare_models \
  --results results.jsonl --output model_analysis/

# Context impact analysis  
python -m lm_eval.tasks.single_turn_scenarios.analysis_tools.context_impact \
  --results results.jsonl --output context_analysis/

# Scenario analysis
python -m lm_eval.tasks.single_turn_scenarios.analysis_tools.scenario_analysis \
  --results results.jsonl --output scenario_analysis/

# Report generation
python -m lm_eval.tasks.single_turn_scenarios.analysis_tools.generate_report \
  --results results.jsonl --output reports/ --format html
```

## Input Format

The analysis tools expect evaluation results in JSONL format with the following structure:

```json
{
  "id": "st_0001",
  "model": "claude-3-5-sonnet",
  "config": "full_context|temperature=0",
  "scenario": "algorithm_implementation",
  "difficulty": "intermediate", 
  "language": "python",
  "context_mode": "full_context",
  "prediction": "def reverse_list(head): ...",
  "metrics": {
    "exact_match": 0,
    "codebleu": 0.73,
    "pass_at_1": 1,
    "syntax_valid": 1,
    "cyclomatic_complexity": 3,
    "security_score": 0.95,
    "performance_score": 0.88
  },
  "runtime": {
    "time_s": 0.45,
    "exit_code": 0,
    "peak_memory_mb": 12
  },
  "seed": 1234,
  "commit": "abc123",
  "requirements": "requirements.txt v1.2.3",
  "timestamp": "2025-09-25T15:00:00Z"
}
```

## Analysis Features

### Model Comparison Analysis

- **Performance Matrix**: Mean scores for each model-metric combination
- **Statistical Tests**: Pairwise t-tests and ANOVA for significance testing
- **Rankings**: Model rankings for each metric
- **Confidence Intervals**: 95% confidence intervals for performance estimates
- **Radar Charts**: Normalized performance visualization across metrics

### Context Impact Analysis

- **Context Comparison**: Performance across different context modes
- **Effect Sizes**: Cohen's d calculations for context impact
- **Improvement Matrix**: Relative improvement from baseline context
- **Statistical Analysis**: Significance testing for context effects
- **Heatmaps**: Visual representation of context impact

### Scenario and Difficulty Analysis

- **Scenario Performance**: Analysis across basic, advanced, and comprehensive scenarios
- **Difficulty Sensitivity**: Performance changes across difficulty levels
- **Language Adaptation**: Performance across programming languages
- **Cross-Analysis**: Multi-dimensional interactions (scenario × difficulty × language)
- **Trend Charts**: Performance trends and sensitivity analysis

### Report Generation

- **HTML Reports**: Comprehensive interactive reports with embedded visualizations
- **CSV Exports**: Raw data exports for further analysis
- **SVG Dashboards**: Scalable vector graphics for presentations
- **Summary Statistics**: Executive summary with key findings

## Output Structure

The analysis tools generate organized output directories:

```
analysis_output/
├── comprehensive_report.html          # Main HTML report
├── evaluation_results.csv             # Raw results export
├── model_comparison/                   # Model comparison analysis
│   ├── performance_matrix.csv
│   ├── statistical_tests.json
│   ├── rankings.json
│   └── confidence_intervals.json
├── context_analysis/                   # Context impact analysis
│   ├── context_comparison.csv
│   ├── context_statistical_analysis.json
│   ├── context_effect_sizes.json
│   └── context_improvement_matrix.csv
├── scenario_analysis/                  # Scenario analysis
│   ├── scenario_performance.csv
│   ├── difficulty_analysis.csv
│   ├── language_adaptation.csv
│   └── cross_analysis.json
├── dashboard/                          # SVG dashboard
│   ├── model_comparison_radar.svg
│   ├── context_impact_heatmap.svg
│   └── performance_trends.svg
└── *.png                              # Individual visualization files
```

## Dependencies

The analysis tools require the following Python packages:

```
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install with:
```bash
pip install pandas numpy scipy matplotlib seaborn
```

## Statistical Methods

### Model Comparison
- **Pairwise t-tests**: Compare performance between model pairs
- **ANOVA**: Multi-group comparison when >2 models
- **Confidence Intervals**: Bootstrap or t-distribution based intervals
- **Effect Sizes**: Cohen's d for practical significance

### Context Impact
- **Paired/Independent t-tests**: Context mode comparisons
- **Effect Size Calculation**: Standardized mean differences
- **Relative Improvement**: Percentage improvement from baseline
- **Statistical Significance**: p-value thresholds and corrections

### Scenario Analysis
- **Linear Regression**: Difficulty sensitivity trends
- **Correlation Analysis**: Cross-dimensional relationships
- **Variance Analysis**: Performance consistency across conditions
- **Ranking Systems**: Ordinal performance comparisons

## Customization

### Adding New Metrics

To analyze custom metrics, ensure they are included in the results JSON with the `metric_` prefix:

```json
{
  "metrics": {
    "custom_metric": 0.85,
    "another_metric": 0.92
  }
}
```

### Custom Visualizations

Each analysis tool provides methods for creating custom visualizations:

```python
from lm_eval.tasks.single_turn_scenarios.analysis_tools import ModelComparator

comparator = ModelComparator(results_data)
fig = comparator.create_radar_chart(['model1', 'model2'], ['metric_exact_match'])
```

### Filtering and Subsetting

All tools support filtering by models and metrics:

```python
# Analyze only specific models and metrics
report = comparator.compare_models(
    models=['claude-3-5-sonnet', 'deepseek-coder'],
    metrics=['metric_exact_match', 'metric_pass_at_1']
)
```

## Troubleshooting

### Common Issues

1. **Missing Data**: Tools handle missing metrics gracefully with NaN values
2. **Empty Results**: Check JSONL format and ensure results contain required fields
3. **Memory Issues**: For large datasets, consider processing in chunks
4. **Visualization Errors**: Ensure matplotlib backend is properly configured

### Performance Optimization

- Use specific model/metric filters to reduce computation time
- Process results in batches for very large datasets
- Cache intermediate results for repeated analysis
- Use SVG format for scalable visualizations

## Examples

### Basic Analysis Workflow

```python
from lm_eval.tasks.single_turn_scenarios.analysis_tools import *

# Load results
with open('results.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

# Model comparison
comparator = ModelComparator(results)
model_report = comparator.compare_models()

# Context analysis
context_analyzer = ContextAnalyzer(results)
context_report = context_analyzer.analyze_context_impact()

# Generate report
generator = ReportGenerator(results)
generator.generate_html_report('report.html')
```

### Advanced Filtering

```python
# Filter by scenario and difficulty
filtered_results = [
    r for r in results 
    if r.get('scenario') == 'algorithm_implementation' 
    and r.get('difficulty') == 'intermediate'
]

analyzer = ScenarioAnalyzer(filtered_results)
report = analyzer.analyze_scenarios_and_difficulty()
```

## Contributing

To add new analysis features:

1. Follow the existing class structure with `analyze_*` methods
2. Include statistical validation and error handling
3. Provide visualization methods with save options
4. Add export functionality for results
5. Update this README with usage examples

## License

This analysis toolkit is part of the lm-eval framework and follows the same licensing terms.