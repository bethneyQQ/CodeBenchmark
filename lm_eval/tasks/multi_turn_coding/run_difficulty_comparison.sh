#!/bin/bash

# Multi-Turn Coding Evaluation - Difficulty-Based Context Comparison
# Compares Full Context vs No Context across all difficulty levels
# Uses 4 problems from each difficulty (16 total) for efficient evaluation

set -e

# Configuration
MODEL="claude-3-5-haiku-20241022"
PROBLEMS_PER_DIFFICULTY=2
OUTPUT_BASE="results/difficulty_comparison"
DIFFICULTIES=("easy" "simple" "medium" "complex")

echo "🎯 Multi-Turn Coding Evaluation - Difficulty Context Comparison"
echo "Model: $MODEL"
echo "Problems per difficulty: $PROBLEMS_PER_DIFFICULTY"
echo "Total problems: $((${#DIFFICULTIES[@]} * $PROBLEMS_PER_DIFFICULTY * 2))"
echo "Output base directory: $OUTPUT_BASE"
echo "======================================================================"

# Create base output directory
mkdir -p "$OUTPUT_BASE"

# Function to run lm_eval with timeout (macOS compatible)
run_lm_eval_with_timeout() {
    local timeout_duration=1800
    local cmd="$1"
    
    echo "Command to run: $cmd"
    echo "Starting evaluation timeout: ${timeout_duration}s"
    
    # Run command in background and get PID
    eval "$cmd" &
    local cmd_pid=$!
    
    # Wait for command with timeout
    local count=0
    while kill -0 $cmd_pid 2>/dev/null; do
        if [ $count -ge $timeout_duration ]; then
            echo "ERROR: Command timed out after ${timeout_duration} seconds"
            kill -TERM $cmd_pid 2>/dev/null
            sleep 5
            kill -KILL $cmd_pid 2>/dev/null
            return 1
        fi
        sleep 10
        count=$((count + 10))
        if [ $((count % 300)) -eq 0 ]; then
            echo "Still running... ${count}s elapsed"
        fi
    done
    
    # Get exit code
    wait $cmd_pid
    return $?
}

# Function to run evaluation for a specific difficulty and context
run_evaluation() {
    local difficulty=$1
    local context_type=$2
    local enable_context=$3
    
    echo "📋 Running: $difficulty difficulty - $context_type"
    
    # Set context environment variables
    if [ "$enable_context" = "true" ]; then
        export ENABLE_PRD_CONTEXT=true
        export ENABLE_DESIGN_CONTEXT=true
        export ENABLE_CODE_CONTEXT=true
        export ENABLE_QUALITY_CONTEXT=true
    else
        export ENABLE_PRD_CONTEXT=false
        export ENABLE_DESIGN_CONTEXT=false
        export ENABLE_CODE_CONTEXT=false
        export ENABLE_QUALITY_CONTEXT=false
    fi
    
    echo "🔧 Context Settings:"
    echo "  ENABLE_PRD_CONTEXT=$ENABLE_PRD_CONTEXT"
    echo "  ENABLE_DESIGN_CONTEXT=$ENABLE_DESIGN_CONTEXT"
    echo "  ENABLE_CODE_CONTEXT=$ENABLE_CODE_CONTEXT"
    echo "  ENABLE_QUALITY_CONTEXT=$ENABLE_QUALITY_CONTEXT"
    
    # Create difficulty-specific output directory
    local output_dir="${OUTPUT_BASE}/${difficulty}"
    mkdir -p "$output_dir"
    
    # Build command with proper arguments (including debug=true)
    local cmd="lm_eval --model claude-code-local --model_args model=$MODEL,multi_turn=true,debug=true,permission_mode=bypassPermissions --tasks multi_turn_coding_eval_claude_code --output_path ${output_dir}/${context_type}_results.json --metadata '{\"difficulty_filter\":\"$difficulty\"}' --log_samples --limit $PROBLEMS_PER_DIFFICULTY --batch_size 1"
    
    # Run evaluation with timeout
    run_lm_eval_with_timeout "$cmd" || {
        echo "ERROR: Evaluation failed or timed out: $difficulty - $context_type"
        return 1
    }
    
    echo "  COMPLETED: $difficulty - $context_type"
    
    # Force cleanup between tests to prevent hanging
    echo "Cleaning up resources..."
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

print('   Cleanup completed')
" 2>/dev/null || echo "   ⚠️  Cleanup had minor issues (continuing anyway)"

    # Small delay to ensure cleanup
    sleep 2
}

# Function to show progress
show_progress() {
    local current=$1
    local total=$2
    local percentage=$((current * 100 / total))
    echo "📊 Progress: $current/$total ($percentage%) completed"
}

# Calculate total evaluations
TOTAL_EVALUATIONS=$((${#DIFFICULTIES[@]} * 2))
CURRENT_EVALUATION=0

echo ""
echo "🚀 Starting evaluations..."
echo ""

# Run evaluations for each difficulty level
for difficulty in "${DIFFICULTIES[@]}"; do
    echo "🎯 Processing $difficulty difficulty level..."
    echo "--------------------------------------------------"
    
    # Full Context evaluation
    ((CURRENT_EVALUATION++))
    show_progress $CURRENT_EVALUATION $TOTAL_EVALUATIONS
    run_evaluation "$difficulty" "full_context" "true"
    
    # No Context evaluation  
    ((CURRENT_EVALUATION++))
    show_progress $CURRENT_EVALUATION $TOTAL_EVALUATIONS
    run_evaluation "$difficulty" "no_context" "false"
    
    echo ""
done

echo "✅ All evaluations completed!"
echo ""

# Generate summary
echo "📊 Evaluation Summary"
echo "=================================================="
echo "Difficulty levels tested: ${#DIFFICULTIES[@]}"
echo "Problems per difficulty: $PROBLEMS_PER_DIFFICULTY"
echo "Context scenarios: 2 (Full Context, No Context)"
echo "Total evaluations: $TOTAL_EVALUATIONS"
echo "Total problems evaluated: $((${#DIFFICULTIES[@]} * $PROBLEMS_PER_DIFFICULTY * 2))"
echo ""

# List generated files
echo "📁 Generated result files:"
for difficulty in "${DIFFICULTIES[@]}"; do
    echo "  ${OUTPUT_BASE}/${difficulty}/"
    echo "    ├── full_context_results.json"
    echo "    └── no_context_results.json"
done
echo ""

# Analysis instructions
echo "📈 Analysis Options:"
echo ""
echo "1. Individual difficulty analysis:"
for difficulty in "${DIFFICULTIES[@]}"; do
    echo "   python quick_results.py ${OUTPUT_BASE}/${difficulty}"
done
echo ""

echo "2. Cross-difficulty comparison:"
echo "   python analyze_context_impact.py \\"
echo "     --difficulty_dirs \\"
for difficulty in "${DIFFICULTIES[@]}"; do
    echo "       ${OUTPUT_BASE}/${difficulty} \\"
done
echo ""

echo "3. Automated difficulty analysis:"
echo "   # First, copy results to expected locations"
for difficulty in "${DIFFICULTIES[@]}"; do
    echo "   cp -r ${OUTPUT_BASE}/${difficulty} results/subset_test_${difficulty}"
done
echo "   # Then run analysis"
echo "   ./run_difficulty_analysis.sh"
echo ""

echo "4. Generate comprehensive report:"
echo "   python analyze_context_impact.py --difficulty_dirs ${OUTPUT_BASE}/easy ${OUTPUT_BASE}/simple ${OUTPUT_BASE}/medium ${OUTPUT_BASE}/complex"
echo ""

echo "✅ Difficulty-based context comparison completed!"
echo "📊 Use the analysis commands above to explore the results."
