# Multi-Turn Coding Metrics Summary

This document provides a comprehensive overview of all metric calculation functions used in the multi-turn coding evaluation system. These metrics are defined in `custom_metrics.py` and referenced in the YAML configuration files using the `metric: !function` pattern.

## Metric Functions Overview

### 1. file_existence_check

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `file_existence_check` and float value representing completion rate

**Calculation Method:**
Checks if the three required files exist for each problem:
- `prd.md` - Product Requirements Document
- `design.md` - Technical Design Document
- `src/` - Source code directory

Score = (number_of_existing_files / 3) averaged across all problems

### 2. prd_quality_from_file

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `prd_quality_from_file` and float value representing PRD quality score

**Calculation Method:**
Evaluates PRD content quality by checking for:
- Basic sections (0.1 each): problem statement, user stories, acceptance criteria, functional requirements, non-functional requirements
- Advanced indicators (0.1 each): specific metrics/KPIs, security considerations, performance requirements, integration requirements, business value
- Word count penalty applied if content < 200 words

Total score capped at 1.0

### 3. design_coherence_from_file

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `design_coherence_from_file` and float value representing design quality score

**Calculation Method:**
Assesses design document quality through:
- Basic elements (0.08 each): architecture/system design, API/interface, database/data model, security
- Advanced elements (0.08 each): scalability, reliability patterns, data consistency, monitoring, deployment, design patterns, technology rationale, specific technologies
- Bonus for diagrams (0.04)
- Word count penalty if content < 300 words

Total score capped at 1.0

### 4. code_execution_test

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `code_execution_test` and float value representing code execution quality

**Calculation Method:**
Comprehensive code quality assessment with four components:
- **Syntax validation (0.2)**: Checks if Python files parse without syntax errors
- **Code quality indicators (0.3)**: Type hints, docstrings, error handling, logging, configuration, import organization
- **Test coverage and execution (0.3)**: Runs tests if they exist, gives partial credit if no tests found
- **Project structure quality (0.2)**: Checks for proper Python package structure, configuration files, tests, documentation

Total score capped at 1.0

### 5. project_structure_validation

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `project_structure_validation` and float value representing structure quality

**Calculation Method:**
Validates project structure with context-aware scoring:
- **Basic structure (0.4)**: requirements.txt, __init__.py, main.py, presence of .py files
- **Advanced structure (0.6)**: test structure, configuration files, documentation, module organization, DevOps files, QA files

Total score capped at 1.0

### 6. integration_test

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `integration_test` and float value representing integration consistency

**Calculation Method:**
Tests consistency between PRD, Design, and Implementation:
- **Technical consistency (0.25)**: Alignment of technical decisions between design and code
- **Architecture alignment (0.25)**: Implementation of architectural patterns mentioned in design
- **Feature completeness (0.25)**: Features from PRD addressed in design and code
- **Quality requirements traceability (0.25)**: Quality requirements addressed throughout all phases

Total score capped at 1.0

### 7. architecture_quality_assessment

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `architecture_quality_assessment` and float value representing architecture quality

**Calculation Method:**
Assesses architecture quality based on design and implementation:
- **Design pattern recognition (0.3)**: Identifies use of common design patterns
- **Scalability considerations (0.2)**: Presence of scalability-related concepts
- **Security considerations (0.2)**: Security measures and practices
- **Performance considerations (0.15)**: Performance optimization techniques
- **Implementation alignment (0.15)**: Alignment between design concepts and code structure

Total score capped at 1.0

### 8. policy_utilization_score

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `policy_utilization_score` and float value representing context utilization

**Calculation Method:**
Assesses how well the solution utilizes provided context:
- Extracts context from original problem data (prd_context, design_context, code_context, quality_context)
- Checks if context terms appear in generated content
- Maps context types to target files: PRD context → prd.md, Design context → design.md, Code context → src/, Quality context → all files
- Score = weighted average of utilization across all context types (0.25 each)

Total score capped at 1.0

### 9. policy_adherence_score

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `policy_adherence_score` and float value representing requirement adherence

**Calculation Method:**
Checks for specific requirements from context in the solution:
- Browser compatibility requirements
- Performance metrics (specific latency, load times, user capacity)
- Accessibility requirements (WCAG, screen readers, ARIA)
- Security standards (OAuth, JWT, encryption, compliance)
- Technology constraints (specific frameworks, databases)
- Architecture patterns (CQRS, microservices, DDD)

Score = weighted average across requirement categories

### 10. technical_constraint_adherence

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `technical_constraint_adherence` and float value representing constraint compliance

**Calculation Method:**
Checks adherence to technical constraints specified in context:
- **Framework constraints (0.2)**: Required frameworks implemented
- **Database constraints (0.15)**: Database technologies used as specified
- **Testing constraints (0.15)**: Testing frameworks and approaches
- **Code style constraints (0.15)**: PEP 8, type hints, docstrings
- **Architecture constraints (0.2)**: Architectural patterns implementation
- **Security constraints (0.15)**: Authentication, authorization implementations

Total score capped at 1.0

### 11. performance_requirement_coverage

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `performance_requirement_coverage` and float value representing performance coverage

**Calculation Method:**
Checks if performance requirements from context are addressed:
- Identifies performance requirements in context (latency, throughput, scalability)
- Checks for performance implementation patterns:
  - Caching (Redis, Memcached) - 0.2
  - Database optimization (indexing, connection pooling) - 0.2
  - Async/concurrent processing - 0.2
  - Load balancing/scaling - 0.15
  - Performance monitoring - 0.15
  - CDN/static optimization - 0.1

Returns 0.5 if no performance requirements in context, otherwise weighted sum

### 12. security_compliance_check

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `security_compliance_check` and float value representing security compliance

**Calculation Method:**
Checks if security and compliance requirements are addressed:
- Identifies security requirements in context
- Checks for security implementation patterns:
  - Authentication/Authorization - 0.25
  - Input validation - 0.2
  - HTTPS/Encryption - 0.2
  - Security headers (CSRF, CORS, CSP) - 0.15
  - Compliance measures (audit, logging) - 0.1
  - Accessibility - 0.1

Returns 0.5 if no security requirements in context, otherwise weighted sum

### 13. execution_time_efficiency

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `execution_time_efficiency` and float value representing average execution time in seconds

**Calculation Method:**
Measures execution time efficiency (lower is better):
- Extracts timing information from evaluation logs if available
- Falls back to estimating based on file modification timestamps
- Returns average execution time across all problems
- Default fallback: 45 seconds if no timing data available

### 14. token_cost_estimation

**Input:**
- `references`: List[str] - Reference data (not used in implementation)
- `predictions`: List[List[str]] - List of predictions for each problem

**Output:**
- Dictionary with key `token_cost_estimation` and float value representing average cost in USD

**Calculation Method:**
Estimates token cost based on generated content (lower is better):
- Extracts actual API cost from logs if available
- Falls back to estimation based on content length:
  - Counts characters in all generated files (PRD, design, code, predictions)
  - Estimates tokens using 4 characters per token ratio
  - Calculates cost using Claude 3.5 Sonnet pricing:
    - Input tokens: $3.00 per 1M tokens
    - Output tokens: $15.00 per 1M tokens
  - Includes estimation of input context size from problem data

## Summary

These metrics provide comprehensive evaluation of multi-turn software engineering tasks across multiple dimensions:

1. **Completeness**: File existence and basic requirements fulfillment
2. **Quality**: Content quality of PRD, design documents, and code
3. **Integration**: Consistency and alignment between different phases
4. **Architecture**: Design patterns and architectural considerations
5. **Context Utilization**: How well provided context is leveraged
6. **Compliance**: Adherence to specific requirements and constraints
7. **Efficiency**: Time and cost considerations

Each metric returns a score between 0.0 and 1.0 (except time and cost metrics which return absolute values), providing quantitative assessment of solution quality across all aspects of the software development lifecycle.