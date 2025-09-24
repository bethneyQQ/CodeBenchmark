# Multi-Turn Scenarios - Quick Usage Guide

## Quick Start

### 1. Code Review Scenario
```bash
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks multi_turn_scenarios.code_review_3_turn \
    --apply_chat_template
```

### 2. Problem Solving Scenario  
```bash
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks multi_turn_scenarios.iterative_problem_solving \
    --apply_chat_template
```

### 3. Teaching Scenario
```bash
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks multi_turn_scenarios.teaching_dialogue \
    --apply_chat_template
```

## Available Scenarios

| Scenario | ID | Turns | Focus |
|----------|----|----|--------|
| Code Review | `code_review_3_turn` | 3 | Review → Revision → Evaluation |
| Problem Solving | `iterative_problem_solving` | 3 | Solution → Refinement → Optimization |
| Teaching | `teaching_dialogue` | 3 | Explain → Q&A → Assessment |
| Conversation | `conversational_3_turn` | 3 | Natural dialogue flow |

## Key Features

✅ **Chat Template Support** - Works with instruction-tuned models  
✅ **Flexible Scenarios** - Easy to extend and customize  
✅ **Rich Metrics** - Scenario-specific evaluation  
✅ **Full lm-eval Integration** - Standard evaluation workflow  

## Sample Workflow: Code Review

**Input**: Python factorial function  
**Turn 1**: Model provides detailed code review  
**Turn 2**: Model revises code based on feedback  
**Turn 3**: Model evaluates the improvements  

## Quick Python API

```python
from lm_eval.tasks.multi_turn_scenarios import MultiTurnEvaluationEngine

engine = MultiTurnEvaluationEngine()

# List scenarios
scenarios = engine.get_scenario_registry()
print(list(scenarios.keys()))

# Evaluate one
results = engine.evaluate_scenario(
    "code_review_3_turn",
    {"code": "def factorial(n): ...", "language": "python"},
    model_generate_function
)
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| `coherence_score` | Multi-turn consistency |
| `response_quality` | Individual response quality |
| `turn_completion_rate` | Success rate |
| `scenario_specific_*` | Custom metrics per scenario |

## Environment Setup

```bash
# Debug mode
export MULTI_TURN_SCENARIOS_DEBUG=1

# Custom chat format  
export DEFAULT_CHAT_FORMAT=chatml
```

## Chat Template Formats

Supported: `chatml`, `llama`, `alpaca`, `vicuna`, `openai`, `anthropic`

Always use `--apply_chat_template` for best results!