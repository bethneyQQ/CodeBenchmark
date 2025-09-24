# Multi-Turn Generic - Quick Usage Guide

## Quick Start

### 1. Basic Evaluation
```bash
lm_eval --model hf --model_args pretrained=your-model-name --tasks multi_turn_generic
```

### 2. With Chat Template (Recommended)
```bash
lm_eval --model hf --model_args pretrained=your-model-name --tasks multi_turn_generic --apply_chat_template
```

### 3. Limited Dataset
```bash
lm_eval --model hf --model_args pretrained=your-model-name --tasks multi_turn_generic --limit 5
```

## What It Evaluates

**Three Sequential Phases:**
1. **Problem Analysis** - Understanding requirements
2. **Solution Design** - Planning the approach  
3. **Implementation** - Writing working code

## Key Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `multi_turn_score` | Overall performance | 0.0-1.0 |
| `phase_consistency` | Cross-phase coherence | 0.0-1.0 |
| `solution_quality` | Final solution quality | 0.0-1.0 |

## Sample Problem

```json
{
  "title": "Implement Binary Search",
  "description": "Write an efficient binary search for a sorted array",
  "difficulty": "medium",
  "category": "algorithms"
}
```

## Expected Model Flow

1. **Analysis**: "This needs O(log n) binary search with two pointers..."
2. **Design**: "Initialize left=0, right=len-1, compare middle element..."
3. **Implementation**: Complete working binary search code

## Common Commands

```bash
# Debug mode
export MULTI_TURN_DEBUG=1

# Custom timeout
export MULTI_TURN_TIMEOUT=600

# Evaluate specific problems  
lm_eval --tasks multi_turn_generic --limit 10
```