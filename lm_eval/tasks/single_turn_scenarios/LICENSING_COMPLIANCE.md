# Licensing and Compliance Documentation

## Overview

This document outlines the licensing and compliance measures implemented for the single_turn_scenarios evaluation task. All problems and associated code follow strict licensing requirements and metadata standards.

## Licensing Information

### Primary License
- **License Type**: MIT License
- **Coverage**: All problems, test cases, and reference implementations
- **Rationale**: MIT provides maximum compatibility with the lm-eval framework while ensuring open-source accessibility

### License Text
```
MIT License

Copyright (c) 2025 lm-eval Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Metadata Standards

### Required Metadata Fields

Each problem in `problems.jsonl` includes the following metadata:

1. **time_limit_s**: Maximum execution time in seconds
   - Range: 5-60 seconds
   - Purpose: Ensures reasonable performance expectations
   - Validation: Must be positive number

2. **memory_limit_mb**: Maximum memory usage in megabytes
   - Range: 100-500 MB
   - Purpose: Prevents resource exhaustion
   - Validation: Must be positive number

3. **seed**: Random seed for reproducibility
   - Type: Integer
   - Purpose: Ensures consistent evaluation results
   - Validation: Must be integer value

4. **author**: Attribution information
   - Value: "system" (for generated problems)
   - Purpose: Clear attribution and responsibility
   - Validation: Must be non-empty string

5. **license**: License identifier
   - Value: "MIT"
   - Purpose: Legal compliance and usage rights
   - Validation: Must be valid SPDX license identifier

### Metadata Validation

The validation system (`validate_problems.py`) enforces:

- **Schema Compliance**: All required fields present
- **Type Validation**: Correct data types for each field
- **Value Validation**: Acceptable ranges and formats
- **Consistency Checks**: Uniform licensing across problems
- **Integrity Verification**: No duplicate IDs or missing sequences

## Compliance Measures

### 1. Automated Validation

```bash
# Run comprehensive validation
python validate_problems.py

# Check test coverage
python tests/validate_tests.py
```

### 2. Licensing Audit Trail

- All problems tracked with unique identifiers
- Consistent MIT licensing across all content
- Author attribution maintained in metadata
- License validation in CI/CD pipeline

### 3. Third-Party Content

**Policy**: No third-party copyrighted content included
- All problems are original implementations
- Reference solutions created specifically for this task
- No external code copied without proper attribution
- Algorithm descriptions use standard computer science knowledge

### 4. Attribution Requirements

When using this task:
1. Maintain original license headers
2. Include attribution to lm-eval project
3. Preserve metadata in problem files
4. Follow MIT license terms for derivatives

## Data Protection and Privacy

### Personal Information
- **No PII**: No personally identifiable information in problems
- **Generic Examples**: All examples use placeholder data
- **Anonymized Content**: No real names, addresses, or sensitive data

### Security Considerations
- **Safe Execution**: All code designed for sandboxed execution
- **No Malicious Code**: Strict review process for all implementations
- **Resource Limits**: Enforced limits prevent resource abuse

## Compliance Verification

### Automated Checks

The validation system performs:

```python
# Schema validation
validate_problem_schema(problem)

# Metadata validation  
validate_metadata(problem)

# License compliance
validate_licensing_compliance(problems)

# Integrity checks
validate_problem_integrity(problems)
```

### Manual Review Process

1. **Code Review**: All implementations reviewed for quality and safety
2. **License Review**: Licensing compliance verified before inclusion
3. **Content Review**: Problems checked for appropriateness and accuracy
4. **Security Review**: Code examined for potential security issues

## Usage Guidelines

### For Researchers
- Cite the lm-eval project when using this task
- Maintain license attribution in publications
- Report any licensing concerns to maintainers

### For Developers
- Follow MIT license terms for modifications
- Preserve metadata when extending the task
- Validate new problems using provided tools

### For Organizations
- Review license compatibility with internal policies
- Ensure compliance with data protection regulations
- Maintain audit trail for usage and modifications

## Reporting Issues

### License Violations
- Report to: lm-eval maintainers
- Include: Specific problem ID and violation details
- Response: Investigation within 48 hours

### Metadata Errors
- Use: GitHub issues with "metadata" label
- Include: Validation output and expected behavior
- Response: Fix in next release cycle

### Compliance Questions
- Contact: Project maintainers
- Include: Specific use case and compliance requirements
- Response: Guidance within one week

## Version Control

### Change Tracking
- All modifications tracked in version control
- License changes require explicit approval
- Metadata updates validated automatically

### Release Process
1. Validation passes all checks
2. License compliance verified
3. Metadata report generated
4. Changes documented in changelog

## Legal Disclaimer

This compliance documentation is provided for informational purposes. Users are responsible for ensuring their use complies with applicable laws and regulations. The lm-eval project makes no warranties regarding legal compliance beyond the scope of the MIT license.

---

**Last Updated**: 2025-09-25  
**Version**: 1.0  
**Validator**: validate_problems.py v1.0