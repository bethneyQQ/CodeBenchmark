# Multi-Turn Coding Evaluation - Simple Test Script (PowerShell)
# Runs 1 problem with full context and no context for quick validation

param(
    [string]$Difficulty = "",
    [string]$ModelBackend = "claude-code",
    [string]$ModelName = "claude-3-haiku-20240307",
    [int]$Limit = 1,
    [switch]$Debug,
    [switch]$Help
)

# Show help
if ($Help) {
    Write-Host "Usage: .\simple_test.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Difficulty LEVEL       Problem difficulty: easy, simple, medium, complex (default: random)"
    Write-Host "  -ModelBackend BACKEND   Model backend: claude-code, deepseek, openai, anthropic, universal (default: claude-code)"
    Write-Host "  -ModelName NAME         Specific model name (default: claude-3-haiku-20240307)"
    Write-Host "  -Limit N                Number of problems to test (default: 1)"
    Write-Host "  -Debug                  Show detailed problem information before running"
    Write-Host "  -Help                   Show this help message"
    Write-Host ""
    Write-Host "Model Backend Examples:"
    Write-Host "  claude-code    - Claude Code SDK (best for file operations)"
    Write-Host "  deepseek       - DeepSeek models (cost-effective, good for code)"
    Write-Host "  openai         - OpenAI GPT models (reliable, well-tested)"
    Write-Host "  anthropic      - Anthropic Claude API (reasoning-focused)"
    Write-Host "  universal      - Universal config (works with any model)"
    Write-Host ""
    Write-Host "Model Name Examples:"
    Write-Host "  Claude: claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229"
    Write-Host "  DeepSeek: deepseek-v3.1, deepseek-v3, deepseek-r1"
    Write-Host "  OpenAI: gpt-4-turbo, gpt-4, gpt-3.5-turbo"
    Write-Host ""
    Write-Host "Usage Examples:"
    Write-Host "  .\simple_test.ps1                                          # Claude Code with Haiku (default)"
    Write-Host "  .\simple_test.ps1 -Difficulty easy                         # Test 1 easy problem with default model"
    Write-Host "  .\simple_test.ps1 -ModelBackend deepseek -ModelName deepseek-v3.1    # Use DeepSeek"
    Write-Host "  .\simple_test.ps1 -ModelBackend openai -ModelName gpt-4-turbo        # Use OpenAI GPT-4"
    Write-Host "  .\simple_test.ps1 -ModelBackend anthropic -ModelName claude-3-sonnet-20240229  # Use Anthropic API"
    Write-Host "  .\simple_test.ps1 -Difficulty medium -Limit 2 -ModelBackend deepseek # Test 2 medium problems with DeepSeek"
    Write-Host "  .\simple_test.ps1 -Debug -ModelBackend universal           # Debug mode with universal config"
    exit 0
}

# Determine model configuration based on backend
switch ($ModelBackend) {
    "claude-code" {
        $LmEvalModel = "claude-code-local"
        $LmEvalTask = "multi_turn_coding_eval_claude_code"
        $ModelArgs = "model=$ModelName,multi_turn=true,debug=true,permission_mode=bypassPermissions"
    }
    "deepseek" {
        $LmEvalModel = "deepseek"
        $LmEvalTask = "multi_turn_coding_eval_deepseek"
        $ModelArgs = "model=$ModelName"
    }
    "openai" {
        $LmEvalModel = "openai-completions"
        $LmEvalTask = "multi_turn_coding_eval_openai"
        $ModelArgs = "model=$ModelName"
    }
    "anthropic" {
        $LmEvalModel = "anthropic_llms"
        $LmEvalTask = "multi_turn_coding_eval_universal"
        $ModelArgs = "model=$ModelName"
    }
    "universal" {
        $LmEvalModel = "anthropic_llms"
        $LmEvalTask = "multi_turn_coding_eval_universal"
        $ModelArgs = "model=$ModelName"
    }
    default {
        Write-Host "❌ Unsupported model backend: $ModelBackend" -ForegroundColor Red
        Write-Host "Supported backends: claude-code, deepseek, openai, anthropic, universal"
        exit 1
    }
}

# Set output directory
$OutputDir = "results/simple_test"
if ($Difficulty) {
    $OutputDir = "results/simple_test_$Difficulty"
}

# Set metadata args
$MetadataArgs = ""
if ($Difficulty) {
    switch ($Difficulty) {
        { $_ -in @("easy", "simple", "medium", "complex") } {
            $MetadataArgs = "--metadata '{`"difficulty_filter`":`"$Difficulty`"}'"
        }
        default {
            Write-Host "❌ Invalid difficulty: $Difficulty" -ForegroundColor Red
            Write-Host "Valid options: easy, simple, medium, complex"
            exit 1
        }
    }
}

Write-Host "🧪 Multi-Turn Coding Evaluation - Simple Test" -ForegroundColor Cyan
Write-Host "Model Backend: $ModelBackend" -ForegroundColor Yellow
Write-Host "Model Name: $ModelName" -ForegroundColor Yellow
Write-Host "LM-Eval Model: $LmEvalModel" -ForegroundColor Yellow
Write-Host "Task: $LmEvalTask" -ForegroundColor Yellow
Write-Host "Problems: $Limit" -ForegroundColor White
if ($Difficulty) {
    Write-Host "Difficulty: $Difficulty" -ForegroundColor White
} else {
    Write-Host "Difficulty: random" -ForegroundColor White
}
Write-Host "Output directory: $OutputDir" -ForegroundColor White
if ($Debug) {
    Write-Host "Debug mode: enabled" -ForegroundColor Green
    if ($MetadataArgs) {
        Write-Host "Metadata filter: $MetadataArgs" -ForegroundColor Green
    } else {
        Write-Host "Metadata filter: none (random selection)" -ForegroundColor Green
    }
}
Write-Host "==================================================" -ForegroundColor Cyan

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Function to run lm_eval with timeout
function Invoke-LmEvalWithTimeout {
    param(
        [string]$Command,
        [int]$TimeoutSeconds = 1800
    )
    
    Write-Host "Command to run: $Command" -ForegroundColor Gray
    Write-Host "⏱️  Starting evaluation (timeout: ${TimeoutSeconds}s)..." -ForegroundColor Yellow
    
    try {
        $job = Start-Job -ScriptBlock {
            param($cmd)
            Invoke-Expression $cmd
        } -ArgumentList $Command
        
        $completed = Wait-Job $job -Timeout $TimeoutSeconds
        
        if ($completed) {
            $result = Receive-Job $job
            Remove-Job $job
            Write-Host $result
            return $true
        } else {
            Write-Host "❌ Command timed out after $TimeoutSeconds seconds" -ForegroundColor Red
            Stop-Job $job
            Remove-Job $job
            return $false
        }
    } catch {
        Write-Host "❌ Error running command: $_" -ForegroundColor Red
        return $false
    }
}

# Test 1: Full Context
Write-Host "📋 Test 1: Running with Full Context" -ForegroundColor Cyan
Write-Host "Setting context environment variables..." -ForegroundColor Yellow
$env:ENABLE_PRD_CONTEXT = "true"
$env:ENABLE_DESIGN_CONTEXT = "true"
$env:ENABLE_CODE_CONTEXT = "true"
$env:ENABLE_QUALITY_CONTEXT = "true"

if ($Debug) {
    Write-Host "🔧 Context Settings:" -ForegroundColor Green
    Write-Host "  ENABLE_PRD_CONTEXT=$($env:ENABLE_PRD_CONTEXT)" -ForegroundColor Green
    Write-Host "  ENABLE_DESIGN_CONTEXT=$($env:ENABLE_DESIGN_CONTEXT)" -ForegroundColor Green
    Write-Host "  ENABLE_CODE_CONTEXT=$($env:ENABLE_CODE_CONTEXT)" -ForegroundColor Green
    Write-Host "  ENABLE_QUALITY_CONTEXT=$($env:ENABLE_QUALITY_CONTEXT)" -ForegroundColor Green
}

Write-Host "Starting lm_eval command..." -ForegroundColor Yellow

if ($MetadataArgs) {
    $cmd = "python -m lm_eval --model $LmEvalModel --model_args $ModelArgs --tasks $LmEvalTask --output_path $OutputDir/full_context_test.json --log_samples --limit $Limit --batch_size 1 $MetadataArgs"
} else {
    $cmd = "python -m lm_eval --model $LmEvalModel --model_args $ModelArgs --tasks $LmEvalTask --output_path $OutputDir/full_context_test.json --log_samples --limit $Limit --batch_size 1"
}

if (-not (Invoke-LmEvalWithTimeout -Command $cmd)) {
    Write-Host "❌ First lm_eval command failed or timed out" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Full context test completed" -ForegroundColor Green

# Small delay between tests
Start-Sleep -Seconds 2

# Test 2: No Context (Baseline)
Write-Host "📋 Test 2: Running with No Context (Baseline)" -ForegroundColor Cyan
Write-Host "Setting no-context environment variables..." -ForegroundColor Yellow
$env:ENABLE_PRD_CONTEXT = "false"
$env:ENABLE_DESIGN_CONTEXT = "false"
$env:ENABLE_CODE_CONTEXT = "false"
$env:ENABLE_QUALITY_CONTEXT = "false"

if ($Debug) {
    Write-Host "🔧 Context Settings:" -ForegroundColor Green
    Write-Host "  ENABLE_PRD_CONTEXT=$($env:ENABLE_PRD_CONTEXT)" -ForegroundColor Green
    Write-Host "  ENABLE_DESIGN_CONTEXT=$($env:ENABLE_DESIGN_CONTEXT)" -ForegroundColor Green
    Write-Host "  ENABLE_CODE_CONTEXT=$($env:ENABLE_CODE_CONTEXT)" -ForegroundColor Green
    Write-Host "  ENABLE_QUALITY_CONTEXT=$($env:ENABLE_QUALITY_CONTEXT)" -ForegroundColor Green
}

Write-Host "Starting lm_eval command..." -ForegroundColor Yellow

if ($MetadataArgs) {
    $cmd = "python -m lm_eval --model $LmEvalModel --model_args $ModelArgs --tasks $LmEvalTask --output_path $OutputDir/no_context_test.json --log_samples --limit $Limit --batch_size 1 $MetadataArgs"
} else {
    $cmd = "python -m lm_eval --model $LmEvalModel --model_args $ModelArgs --tasks $LmEvalTask --output_path $OutputDir/no_context_test.json --log_samples --limit $Limit --batch_size 1"
}

if (-not (Invoke-LmEvalWithTimeout -Command $cmd)) {
    Write-Host "❌ Second lm_eval command failed or timed out" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ No context test completed" -ForegroundColor Green

# Quick comparison
Write-Host ""
Write-Host "📊 Quick Results Comparison" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ((Test-Path "$OutputDir/full_context_test.json") -and (Test-Path "$OutputDir/no_context_test.json")) {
    Write-Host "Full Context Results:" -ForegroundColor Yellow
    python -c @"
import json
try:
    with open('$OutputDir/full_context_test.json', 'r') as f:
        data = json.load(f)
    results = data.get('results', {})
    for task, metrics in results.items():
        print(f'  Task: {task}')
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f'    {metric}: {value:.3f}')
except Exception as e:
    print(f'  Error reading results: {e}')
"@

    Write-Host ""
    Write-Host "No Context Results:" -ForegroundColor Yellow
    python -c @"
import json
try:
    with open('$OutputDir/no_context_test.json', 'r') as f:
        data = json.load(f)
    results = data.get('results', {})
    for task, metrics in results.items():
        print(f'  Task: {task}')
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f'    {metric}: {value:.3f}')
except Exception as e:
    print(f'  Error reading results: {e}')
"@
} else {
    Write-Host "⚠️  Could not find result files for comparison" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📁 Generated Files:" -ForegroundColor Cyan
Write-Host "  Results: $OutputDir/" -ForegroundColor White
Write-Host "  Output artifacts: ./output/" -ForegroundColor White
Write-Host ""
Write-Host "🔍 To inspect generated artifacts:" -ForegroundColor Cyan
Write-Host "  Get-ChildItem -Recurse ./output/  # View created files" -ForegroundColor Gray
Write-Host "  Get-Content ./output/*/prd.md    # View PRD content" -ForegroundColor Gray
Write-Host "  Get-Content ./output/*/design.md # View design content" -ForegroundColor Gray
Write-Host "  Get-ChildItem ./output/*/src/    # View code structure" -ForegroundColor Gray
Write-Host ""
Write-Host "📈 To run full analysis:" -ForegroundColor Cyan
Write-Host "  python analyze_context_impact.py --results_dir $OutputDir" -ForegroundColor Gray

if ($Debug) {
    Write-Host ""
    Write-Host "🔍 Debug Summary:" -ForegroundColor Green
    Write-Host "  Model backend: $ModelBackend" -ForegroundColor White
    Write-Host "  Model name: $ModelName" -ForegroundColor White
    Write-Host "  LM-Eval model: $LmEvalModel" -ForegroundColor White
    Write-Host "  Task: $LmEvalTask" -ForegroundColor White
    Write-Host "  Problems tested: $Limit" -ForegroundColor White
    Write-Host "  Difficulty filter: $(if ($Difficulty) { $Difficulty } else { 'random' })" -ForegroundColor White
    Write-Host "  Output directory: $OutputDir" -ForegroundColor White
    Write-Host "  Full context test: $(if (Test-Path "$OutputDir/full_context_test.json") { '✅ completed' } else { '❌ failed' })" -ForegroundColor White
    Write-Host "  No context test: $(if (Test-Path "$OutputDir/no_context_test.json") { '✅ completed' } else { '❌ failed' })" -ForegroundColor White
}

Write-Host ""
Write-Host "✅ Simple test completed successfully!" -ForegroundColor Green