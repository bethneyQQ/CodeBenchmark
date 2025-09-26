# Sample Outputs

This directory contains example evaluation results from the single_turn_scenarios task to demonstrate the expected output format and structure.

## Files

- `code_completion_results.json` - Example results from code completion scenario
- `suite_results.json` - Example results from full suite evaluation
- `python_filtered_results.json` - Example results filtered by Python language
- `intermediate_difficulty_results.json` - Example results filtered by intermediate difficulty
- `minimal_context_results.json` - Example results with minimal context mode

## Output Format

All results follow the structured JSON format specified in the design document:

```json
{
  "results": {
    "task_name": {
      "metric_name": value,
      ...
    }
  },
  "config": {
    "model": "model_name",
    "model_args": {...},
    "task": "task_name",
    ...
  },
  "samples": [
    {
      "id": "problem_id",
      "prediction": "model_output",
      "target": "expected_output",
      "metrics": {...}
    }
  ]
}
```

## Usage

These sample outputs can be used for:
- Understanding the expected output format
- Testing analysis tools
- Validating result processing pipelines
- Comparing against your own evaluation results