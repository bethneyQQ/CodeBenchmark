#!/bin/bash

# Multi-Turn Coding Evaluation - Difficulty Analysis Runner
# Analyzes context impact across different difficulty levels

set -e

echo "🎯 Multi-Turn Coding Evaluation - Difficulty Analysis"
echo "=" * 60

# Check if results directories exist
RESULTS_BASE="results"
DIFFICULTY_DIRS=()

# Look for difficulty-specific result directories
for difficulty in easy simple medium complex; do
    for pattern in "simple_test_${difficulty}" "${difficulty}_test" "*${difficulty}*"; do
        if ls ${RESULTS_BASE}/${pattern} 1> /dev/null 2>&1; then
            for dir in ${RESULTS_BASE}/${pattern}; do
                if [ -d "$dir" ]; then
                    DIFFICULTY_DIRS+=("$dir")
                    echo "✅ Found: $dir"
                    break
                fi
            done
            break
        fi
    done
done

if [ ${#DIFFICULTY_DIRS[@]} -eq 0 ]; then
    echo "❌ No difficulty-specific results directories found!"
    echo ""
    echo "Expected directories like:"
    echo "  - results/simple_test_easy"
    echo "  - results/simple_test_simple" 
    echo "  - results/simple_test_medium"
    echo "  - results/simple_test_complex"
    echo ""
    echo "Run evaluations first:"
    echo "  ./simple_test.sh --difficulty easy"
    echo "  ./simple_test.sh --difficulty simple"
    echo "  ./simple_test.sh --difficulty medium"
    echo "  ./simple_test.sh --difficulty complex"
    exit 1
fi

echo ""
echo "📊 Running difficulty analysis on ${#DIFFICULTY_DIRS[@]} directories..."

# Run the difficulty analysis
python analyze_context_impact.py --difficulty_dirs "${DIFFICULTY_DIRS[@]}"

echo ""
echo "✅ Difficulty analysis completed!"
echo ""
echo "📁 Generated files:"
echo "  - results/difficulty_context_analysis.png"
echo "  - results/difficulty_context_heatmap.png" 
echo "  - results/difficulty_context_analysis_report.md"
echo ""
echo "📖 View the report:"
echo "  open results/difficulty_context_analysis_report.md"
echo ""
echo "🔍 Quick summary:"
python -c "
import os
if os.path.exists('results/difficulty_context_analysis_report.md'):
    with open('results/difficulty_context_analysis_report.md', 'r') as f:
        lines = f.readlines()
    
    # Find and print key insights
    in_insights = False
    for line in lines:
        if '## Key Insights' in line:
            in_insights = True
            continue
        elif in_insights and line.startswith('##'):
            break
        elif in_insights and line.strip():
            print(line.rstrip())
else:
    print('Report not found')
"