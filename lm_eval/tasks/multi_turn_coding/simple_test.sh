#!/bin/bash

# Multi-Turn Coding Evaluation - Simple Test Script
# Runs 1 problem with full context and no context for quick validation

# Removed set -e to allow better error debugging
# set -e

# Function to handle errors
handle_error() {
    echo "❌ Error occurred in script at line $1"
    echo "Command that failed: $2"
    return 1
}

# Set up error trap (but don't exit immediately)
# trap 'handle_error $LINENO "$BASH_COMMAND"' ERR

# Default values
MODEL_BACKEND="claude-code"
MODEL_NAME="claude-3-haiku-20240307"
LIMIT=1
OUTPUT_DIR="results/simple_test"
DIFFICULTY=""
DEBUG=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --difficulty)
            DIFFICULTY="$2"
            shift 2
            ;;
        --model-backend)
            MODEL_BACKEND="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model)
            # Legacy support: --model can specify either backend or full model name
            if [[ "$2" == *"claude-code"* ]] || [[ "$2" == *"deepseek"* ]] || [[ "$2" == *"openai"* ]] || [[ "$2" == *"anthropic"* ]]; then
                MODEL_BACKEND="$2"
            else
                MODEL_NAME="$2"
            fi
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --difficulty LEVEL       Problem difficulty: easy, simple, medium, complex (default: random)"
            echo "  --model-backend BACKEND  Model backend: claude-code, deepseek, openai, anthropic, universal (default: claude-code)"
            echo "  --model-name NAME        Specific model name (default: claude-3-haiku-20240307)"
            echo "  --model NAME             Legacy: model name or backend (deprecated, use --model-backend and --model-name)"
            echo "  --limit N                Number of problems to test (default: 1)"
            echo "  --debug                  Show detailed problem information before running"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Model Backend Examples:"
            echo "  claude-code    - Claude Code SDK (best for file operations)"
            echo "  deepseek       - DeepSeek models (cost-effective, good for code)"
            echo "  openai         - OpenAI GPT models (reliable, well-tested)"
            echo "  anthropic      - Anthropic Claude API (reasoning-focused)"
            echo "  universal      - Universal config (works with any model)"
            echo ""
            echo "Model Name Examples:"
            echo "  Claude: claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229"
            echo "  DeepSeek: deepseek-v3.1, deepseek-v3, deepseek-r1"
            echo "  OpenAI: gpt-4-turbo, gpt-4, gpt-3.5-turbo"
            echo ""
            echo "Usage Examples:"
            echo "  $0                                          # Claude Code with Haiku (default)"
            echo "  $0 --difficulty easy                        # Test 1 easy problem with default model"
            echo "  $0 --model-backend deepseek --model-name deepseek-v3.1    # Use DeepSeek"
            echo "  $0 --model-backend openai --model-name gpt-4-turbo        # Use OpenAI GPT-4"
            echo "  $0 --model-backend anthropic --model-name claude-3-sonnet-20240229  # Use Anthropic API"
            echo "  $0 --difficulty medium --limit 2 --model-backend deepseek # Test 2 medium problems with DeepSeek"
            echo "  $0 --debug --model-backend universal       # Debug mode with universal config"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set metadata based on difficulty
METADATA_ARGS=""
if [[ -n "$DIFFICULTY" ]]; then
    case $DIFFICULTY in
        easy|simple|medium|complex)
            METADATA_ARGS="--metadata '{\"difficulty_filter\":\"$DIFFICULTY\"}'"
            OUTPUT_DIR="results/simple_test_${DIFFICULTY}"
            ;;
        *)
            echo "❌ Invalid difficulty: $DIFFICULTY"
            echo "Valid options: easy, simple, medium, complex"
            exit 1
            ;;
    esac
fi

# Determine model configuration based on backend
case $MODEL_BACKEND in
    claude-code)
        LM_EVAL_MODEL="claude-code-local"
        LM_EVAL_TASK="multi_turn_coding_eval_claude_code"
        MODEL_ARGS="model=$MODEL_NAME,multi_turn=true,debug=true,permission_mode=bypassPermissions"
        ;;
    deepseek)
        LM_EVAL_MODEL="deepseek"
        LM_EVAL_TASK="multi_turn_coding_eval_deepseek"
        MODEL_ARGS="model=$MODEL_NAME"
        ;;
    openai)
        LM_EVAL_MODEL="openai-completions"
        LM_EVAL_TASK="multi_turn_coding_eval_openai"
        MODEL_ARGS="model=$MODEL_NAME"
        ;;
    anthropic)
        LM_EVAL_MODEL="anthropic_llms"
        LM_EVAL_TASK="multi_turn_coding_eval_universal"
        MODEL_ARGS="model=$MODEL_NAME"
        ;;
    universal)
        LM_EVAL_MODEL="anthropic_llms"  # Default to anthropic for universal
        LM_EVAL_TASK="multi_turn_coding_eval_universal"
        MODEL_ARGS="model=$MODEL_NAME"
        ;;
    *)
        echo "❌ Unsupported model backend: $MODEL_BACKEND"
        echo "Supported backends: claude-code, deepseek, openai, anthropic, universal"
        exit 1
        ;;
esac

echo "🧪 Multi-Turn Coding Evaluation - Simple Test"
echo "Model Backend: $MODEL_BACKEND"
echo "Model Name: $MODEL_NAME"
echo "LM-Eval Model: $LM_EVAL_MODEL"
echo "Task: $LM_EVAL_TASK"
echo "Problems: $LIMIT"
if [[ -n "$DIFFICULTY" ]]; then
    echo "Difficulty: $DIFFICULTY"
else
    echo "Difficulty: random"
fi
echo "Output directory: $OUTPUT_DIR"
if [[ "$DEBUG" == "true" ]]; then
    echo "Debug mode: enabled"
    if [[ -n "$METADATA_ARGS" ]]; then
        echo "Metadata filter: $METADATA_ARGS"
    else
        echo "Metadata filter: none (random selection)"
    fi
fi
echo "=================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to show problem details when debug is enabled
show_problem_details() {
    echo "🔍 Debug: Loading problem details..."
    python3 -c "
import sys
import os
import json

try:
    # Check if datasets is available
    try:
        import datasets
        use_datasets = True
    except ImportError:
        print('⚠️  datasets package not installed - using direct JSON parsing')
        use_datasets = False
    
    if use_datasets:
        # Use the full utils.py approach
        sys.path.append('.')
        from utils import load_dataset
        
        # Load dataset with same parameters as the evaluation
        metadata = {}
        if '$DIFFICULTY':
            metadata['difficulty_filter'] = '$DIFFICULTY'
        
        data = load_dataset(metadata=metadata if metadata else None)
        problems = data['test']
    else:
        # Fallback: directly parse problems.jsonl
        with open('problems.jsonl', 'r') as f:
            all_problems = json.load(f)
        
        # Filter by difficulty if specified
        if '$DIFFICULTY':
            problems = [p for p in all_problems if p.get('complexity') == '$DIFFICULTY']
        else:
            problems = all_problems
    
    if problems:
        # Show details for the first problem(s) that will be tested
        limit = min($LIMIT, len(problems))
        print(f'📋 Will test {limit} problem(s) from {len(problems)} available:')
        print('=' * 60)
        
        for i in range(limit):
            problem = problems[i]
            print(f'Problem {i+1}: {problem[\"problem_id\"]}')
            print(f'  Complexity: {problem[\"complexity\"]}')
            print(f'  Domain: {problem[\"domain\"]}')
            print(f'  Description: {problem[\"problem_description\"]}')
            print()
            print(f'  PRD Context: {problem[\"prd_context\"]}')
            print()
            print(f'  Design Context: {problem[\"design_context\"]}')
            print()
            print(f'  Code Context: {problem[\"code_context\"]}')
            print()
            print(f'  Quality Context: {problem[\"quality_context\"]}')
            print('=' * 60)
    else:
        print('❌ No problems found matching criteria')
        
except Exception as e:
    print(f'❌ Error loading problem details: {e}')
    import traceback
    traceback.print_exc()
"
}

# Show problem details if debug is enabled
if [[ "$DEBUG" == "true" ]] || [[ "$*" == *"debug=true"* ]]; then
    show_problem_details
fi

# Test 1: Full Context
echo "📋 Test 1: Running with Full Context"
echo "Setting context environment variables..."
export ENABLE_PRD_CONTEXT=true
export ENABLE_DESIGN_CONTEXT=true
export ENABLE_CODE_CONTEXT=true
export ENABLE_QUALITY_CONTEXT=true

if [[ "$DEBUG" == "true" ]]; then
    echo "🔧 Context Settings:"
    echo "  ENABLE_PRD_CONTEXT=$ENABLE_PRD_CONTEXT"
    echo "  ENABLE_DESIGN_CONTEXT=$ENABLE_DESIGN_CONTEXT"
    echo "  ENABLE_CODE_CONTEXT=$ENABLE_CODE_CONTEXT"
    echo "  ENABLE_QUALITY_CONTEXT=$ENABLE_QUALITY_CONTEXT"
fi

echo "Starting lm_eval command..."

# Function to run lm_eval with timeout (macOS compatible)
run_lm_eval_with_timeout() {
    local timeout_duration=1800  # 30 minutes
    local cmd="$1"
    
    echo "Command to run: $cmd"
    echo "⏱️  Starting evaluation (timeout: ${timeout_duration}s)..."
    
    # Run command in background and get PID
    eval "$cmd" &
    local cmd_pid=$!
    
    # Wait for command with timeout
    local count=0
    while kill -0 $cmd_pid 2>/dev/null; do
        if [ $count -ge $timeout_duration ]; then
            echo "❌ Command timed out after ${timeout_duration} seconds"
            kill -TERM $cmd_pid 2>/dev/null
            sleep 5
            kill -KILL $cmd_pid 2>/dev/null
            return 1
        fi
        sleep 10
        count=$((count + 10))
        if [ $((count % 300)) -eq 0 ]; then  # Every 5 minutes
            echo "⏱️  Still running... (${count}s elapsed)"
        fi
    done
    
    # Get exit code
    wait $cmd_pid
    return $?
}

if [[ -n "$METADATA_ARGS" ]]; then
    CMD="lm_eval --model $LM_EVAL_MODEL --model_args $MODEL_ARGS --tasks $LM_EVAL_TASK --output_path $OUTPUT_DIR/full_context_test.json --log_samples --limit $LIMIT --batch_size 1 $METADATA_ARGS"
    run_lm_eval_with_timeout "$CMD" || {
        echo "❌ First lm_eval command failed or timed out"
        exit 1
    }
else
    CMD="lm_eval --model $LM_EVAL_MODEL --model_args $MODEL_ARGS --tasks $LM_EVAL_TASK --output_path $OUTPUT_DIR/full_context_test.json --log_samples --limit $LIMIT --batch_size 1"
    run_lm_eval_with_timeout "$CMD" || {
        echo "❌ First lm_eval command failed or timed out"
        exit 1
    }
fi

echo ""
echo "✅ Full context test completed"

# Force cleanup between tests to prevent hanging
echo "🧹 Cleaning up resources between tests..."
python3 -c "
import gc
import asyncio
import sys

# Force garbage collection
gc.collect()

# Try to close any remaining event loops
try:
    loop = asyncio.get_event_loop()
    if not loop.is_closed():
        # Don't close the loop if it's running, just clean up
        pass
except RuntimeError:
    # No event loop, that's fine
    pass

print('   ✅ Cleanup completed')
" 2>/dev/null || echo "   ⚠️  Cleanup had minor issues (continuing anyway)"

# Small delay to ensure cleanup
sleep 2

# Test 2: No Context (Baseline)
echo "📋 Test 2: Running with No Context (Baseline)"
echo "Setting no-context environment variables..."
export ENABLE_PRD_CONTEXT=false
export ENABLE_DESIGN_CONTEXT=false
export ENABLE_CODE_CONTEXT=false
export ENABLE_QUALITY_CONTEXT=false

if [[ "$DEBUG" == "true" ]]; then
    echo "🔧 Context Settings:"
    echo "  ENABLE_PRD_CONTEXT=$ENABLE_PRD_CONTEXT"
    echo "  ENABLE_DESIGN_CONTEXT=$ENABLE_DESIGN_CONTEXT"
    echo "  ENABLE_CODE_CONTEXT=$ENABLE_CODE_CONTEXT"
    echo "  ENABLE_QUALITY_CONTEXT=$ENABLE_QUALITY_CONTEXT"
fi

echo "Starting lm_eval command..."

if [[ -n "$METADATA_ARGS" ]]; then
    CMD="lm_eval --model $LM_EVAL_MODEL --model_args $MODEL_ARGS --tasks $LM_EVAL_TASK --output_path $OUTPUT_DIR/no_context_test.json --log_samples --limit $LIMIT --batch_size 1 $METADATA_ARGS"
    run_lm_eval_with_timeout "$CMD" || {
        echo "❌ Second lm_eval command failed or timed out"
        exit 1
    }
else
    CMD="lm_eval --model $LM_EVAL_MODEL --model_args $MODEL_ARGS --tasks $LM_EVAL_TASK --output_path $OUTPUT_DIR/no_context_test.json --log_samples --limit $LIMIT --batch_size 1"
    run_lm_eval_with_timeout "$CMD" || {
        echo "❌ Second lm_eval command failed or timed out"
        exit 1
    }
fi

echo ""
echo "✅ No context test completed"

# Quick comparison
echo ""
echo "📊 Quick Results Comparison"
echo "========================================"

if [ -f "$OUTPUT_DIR/full_context_test.json" ] && [ -f "$OUTPUT_DIR/no_context_test.json" ]; then
    echo "Full Context Results:"
    python -c "
import json
try:
    with open('$OUTPUT_DIR/full_context_test.json', 'r') as f:
        data = json.load(f)
    results = data.get('results', {})
    for task, metrics in results.items():
        print(f'  Task: {task}')
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f'    {metric}: {value:.3f}')
except Exception as e:
    print(f'  Error reading results: {e}')
"

    echo ""
    echo "No Context Results:"
    python -c "
import json
try:
    with open('$OUTPUT_DIR/no_context_test.json', 'r') as f:
        data = json.load(f)
    results = data.get('results', {})
    for task, metrics in results.items():
        print(f'  Task: {task}')
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f'    {metric}: {value:.3f}')
except Exception as e:
    print(f'  Error reading results: {e}')
"
else
    echo "⚠️  Could not find result files for comparison"
fi

echo ""
echo "📁 Generated Files:"
echo "  Results: $OUTPUT_DIR/"
echo "  Output artifacts: ./output/"
echo ""
echo "🔍 To inspect generated artifacts:"
echo "  ls -la ./output/*/  # View created files"
echo "  cat ./output/*/prd.md  # View PRD content"
echo "  cat ./output/*/design.md  # View design content"
echo "  ls -la ./output/*/src/  # View code structure"
echo ""
echo "📈 To run full analysis:"
echo "  python analyze_context_impact.py --results_dir $OUTPUT_DIR"
if [[ "$DEBUG" == "true" ]]; then
    echo ""
    echo "🔍 Debug Summary:"
    echo "  Model backend: $MODEL_BACKEND"
    echo "  Model name: $MODEL_NAME"
    echo "  LM-Eval model: $LM_EVAL_MODEL"
    echo "  Task: $LM_EVAL_TASK"
    echo "  Problems tested: $LIMIT"
    echo "  Difficulty filter: ${DIFFICULTY:-random}"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Full context test: $([ -f "$OUTPUT_DIR/full_context_test.json" ] && echo "✅ completed" || echo "❌ failed")"
    echo "  No context test: $([ -f "$OUTPUT_DIR/no_context_test.json" ] && echo "✅ completed" || echo "❌ failed")"
fi

echo ""
echo "✅ Simple test completed successfully!"