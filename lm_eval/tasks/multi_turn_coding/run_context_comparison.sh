#!/bin/bash

# Multi-Turn Coding Evaluation - Context Comparison Script
# This script runs the evaluation with different context configurations

set -e

MODEL="claude-3-haiku-20240307"
LIMIT=5
OUTPUT_DIR="results/context_comparison"

echo "🚀 Starting Multi-Turn Coding Context Comparison"
echo "Model: $MODEL"
echo "Problems per scenario: $LIMIT"
echo "Output directory: $OUTPUT_DIR"
echo "=" * 60

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Scenario 1: Full Context (Default)
echo "📋 Running Scenario 1: Full Context"
export ENABLE_PRD_CONTEXT=true
export ENABLE_DESIGN_CONTEXT=true
export ENABLE_CODE_CONTEXT=true
export ENABLE_QUALITY_CONTEXT=true

lm_eval \
  --model claude-code-local \
  --model_args model=$MODEL,multi_turn=true,permission_mode=bypassPermissions \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path "$OUTPUT_DIR/full_context_results.json" \
  --log_samples \
  --limit $LIMIT \
  --batch_size 1

# Scenario 2: No Context (Baseline)
echo "📋 Running Scenario 2: No Context (Baseline)"
export ENABLE_PRD_CONTEXT=false
export ENABLE_DESIGN_CONTEXT=false
export ENABLE_CODE_CONTEXT=false
export ENABLE_QUALITY_CONTEXT=false

lm_eval \
  --model claude-code-local \
  --model_args model=$MODEL,multi_turn=true,permission_mode=bypassPermissions \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path "$OUTPUT_DIR/no_context_results.json" \
  --log_samples \
  --limit $LIMIT \
  --batch_size 1

# Scenario 3: PRD Context Only
echo "📋 Running Scenario 3: PRD Context Only"
export ENABLE_PRD_CONTEXT=true
export ENABLE_DESIGN_CONTEXT=false
export ENABLE_CODE_CONTEXT=false
export ENABLE_QUALITY_CONTEXT=false

lm_eval \
  --model claude-code-local \
  --model_args model=$MODEL,multi_turn=true,permission_mode=bypassPermissions \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path "$OUTPUT_DIR/prd_only_results.json" \
  --log_samples \
  --limit $LIMIT \
  --batch_size 1

# Scenario 4: Technical Context Only (Design + Code + Quality)
echo "📋 Running Scenario 4: Technical Context Only"
export ENABLE_PRD_CONTEXT=false
export ENABLE_DESIGN_CONTEXT=true
export ENABLE_CODE_CONTEXT=true
export ENABLE_QUALITY_CONTEXT=true

lm_eval \
  --model claude-code-local \
  --model_args model=$MODEL,multi_turn=true,permission_mode=bypassPermissions \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path "$OUTPUT_DIR/technical_only_results.json" \
  --log_samples \
  --limit $LIMIT \
  --batch_size 1

# Scenario 5: Design + Code Context Only
echo "📋 Running Scenario 5: Design + Code Context Only"
export ENABLE_PRD_CONTEXT=false
export ENABLE_DESIGN_CONTEXT=true
export ENABLE_CODE_CONTEXT=true
export ENABLE_QUALITY_CONTEXT=false

lm_eval \
  --model claude-code-local \
  --model_args model=$MODEL,multi_turn=true,permission_mode=bypassPermissions \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path "$OUTPUT_DIR/design_code_only_results.json" \
  --log_samples \
  --limit $LIMIT \
  --batch_size 1

echo "✅ Context comparison evaluation complete!"
echo "📊 Results saved in: $OUTPUT_DIR"
echo ""
echo "📈 To analyze results, run:"
echo "python analyze_context_impact.py --results_dir $OUTPUT_DIR"