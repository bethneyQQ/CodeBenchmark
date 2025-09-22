# Single-turn Metrics

This document summarizes all metric calculation functions found in `utils.py` that correspond to the `metric: !function` keywords in YAML configuration files.

## Text Generation Metrics

### 1. BLEU Score (`bleu_score`)
**Location**: `utils.py:639-661`

**Input**:
- `gold_and_result`: List containing [references, predictions]
- `references`: String or list of reference texts
- `predictions`: String or list of predicted texts

**Output**:
- Dictionary with key `"bleu"` containing float score (0.0-1.0)

**Calculation Method**:
Uses HuggingFace's BLEU metric implementation. Computes BLEU score by comparing predicted text against reference text using n-gram precision with brevity penalty. Higher scores indicate better match.

### 2. ROUGE Score (`rouge_score`)
**Location**: `utils.py:663-690`

**Input**:
- `gold_and_result`: List containing [references, predictions]
- `references`: String or list of reference texts
- `predictions`: String or list of predicted texts

**Output**:
- Dictionary with keys `"rouge1"`, `"rouge2"`, `"rougeL"` containing float scores (0.0-1.0)

**Calculation Method**:
Uses HuggingFace's ROUGE metric with stemmer enabled. Computes recall-oriented metrics:
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence

### 3. METEOR Score (`meteor_score`)
**Location**: `utils.py:692-714`

**Input**:
- `gold_and_result`: List containing [references, predictions]
- `references`: String or list of reference texts
- `predictions`: String or list of predicted texts

**Output**:
- Dictionary with key `"meteor"` containing float score (0.0-1.0)

**Calculation Method**:
Uses HuggingFace's METEOR metric implementation. Computes score based on unigram precision and recall with additional features like stemming and synonym matching.

## Code-Specific Metrics

### 4. CodeBLEU Score (`code_bleu`)
**Location**: `utils.py:716-738`

**Input**:
- `gold_and_result`: List containing [references, predictions]
- `references`: String or list of reference code texts
- `predictions`: String or list of predicted code texts

**Output**:
- Dictionary with key `"code_bleu"` containing float score (0.0-1.0)

**Calculation Method**:
Currently uses standard BLEU metric implementation. Designed for code translation tasks, measuring syntactic similarity between code segments.

### 5. Pass@k Score (`pass_at_k`)
**Location**: `utils.py:740-757`

**Input**:
- `references`: List of test case strings
- `predictions`: List of lists of code predictions
- `k`: List of integers or single integer (default [1])

**Output**:
- Dictionary with keys `"pass@{k}"` containing float scores (0.0-1.0)

**Calculation Method**:
Uses HuggingFace's code_eval metric. Executes generated code against test cases and calculates the percentage of problems solved correctly in the first k attempts.

### 6. Edit Distance (`edit_distance`)
**Location**: `utils.py:759-799`

**Input**:
- `gold_and_result`: List containing [references, predictions]
- `references`: String or list of reference texts
- `predictions`: String or list of predicted texts

**Output**:
- Dictionary with key `"edit_distance"` containing float score (0.0-1.0)

**Calculation Method**:
Implements Levenshtein distance algorithm to calculate minimum number of character edits needed to transform prediction into reference. Score is normalized by maximum string length. Lower scores indicate better similarity.

## Context-Aware Quality Metrics

### 7. Context Adherence Score (`context_adherence_score`)
**Location**: `utils.py:802-853`

**Input**:
- `gold_and_result`: List containing [references, predictions]
- Only uses predictions for scoring

**Output**:
- Dictionary with key `"context_adherence_score"` containing float score (0.0-1.0)

**Calculation Method**:
Heuristic-based scoring examining:
- Descriptive variable names (>2 characters)
- Type hints presence
- Docstring presence
- Line length compliance (≤80 chars)
- Proper function structure (def + return)
Maximum score of 5 points, normalized to 0.0-1.0 range.

### 8. Style Compliance Score (`style_compliance_score`)
**Location**: `utils.py:855-915`

**Input**:
- `gold_and_result`: List containing [references, predictions]
- Only uses predictions for scoring

**Output**:
- Dictionary with key `"style_compliance_score"` containing float score (0.0-1.0)

**Calculation Method**:
Python style guideline compliance checking:
- Proper indentation (4 spaces)
- Descriptive naming (80% of names >2 chars)
- Docstring presence
- Type hints usage
- Line length compliance (≤80 chars)
- Proper operator spacing
Maximum score of 6 points, normalized to 0.0-1.0 range.

### 9. Security Compliance Score (`security_compliance_score`)
**Location**: `utils.py:917-961`

**Input**:
- `gold_and_result`: List containing [references, predictions]
- Only uses predictions for scoring

**Output**:
- Dictionary with key `"security_compliance_score"` containing float score (0.0-1.0)

**Calculation Method**:
Security best practices assessment:
- Input validation presence (validate, check, isinstance, etc.)
- Error handling implementation (try/except blocks)
- No hardcoded credentials detection
- Safe function usage (no eval, exec, compile)
Maximum score of 4 points, normalized to 0.0-1.0 range.

### 10. Performance Awareness Score (`performance_awareness_score`)
**Location**: `utils.py:963-1009`

**Input**:
- `gold_and_result`: List containing [references, predictions]
- Only uses predictions for scoring

**Output**:
- Dictionary with key `"performance_awareness_score"` containing float score (0.0-1.0)

**Calculation Method**:
Performance optimization patterns detection:
- List comprehensions instead of loops
- Generator expressions usage
- Avoiding nested loops
- Efficient built-in functions usage (sum, max, min, etc.)
Maximum score of 4 points, normalized to 0.0-1.0 range.

## Usage in YAML Files

These metrics are referenced in the following task configurations:

- **Code Completion**: `bleu_score`, `context_adherence_score`, `style_compliance_score`, `security_compliance_score`, `performance_awareness_score`
- **Code Repair**: `edit_distance`, `context_adherence_score`, `style_compliance_score`, `security_compliance_score`
- **Code Translation**: `bleu_score`, `code_bleu`, `context_adherence_score`, `style_compliance_score`
- **Docstring Generation**: `bleu_score`, `rouge_score`, `meteor_score`, `context_adherence_score`, `style_compliance_score`
- **Function Generation**: `pass_at_k`, `context_adherence_score`, `style_compliance_score`, `security_compliance_score`, `performance_awareness_score`

## Error Handling

All metric functions include try-catch blocks and return default values (typically 0.0) when calculation fails, ensuring robust evaluation pipeline operation.