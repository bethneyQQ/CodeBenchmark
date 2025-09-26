# Single Turn Scenarios Integrity Report

**Generated**: 2025-09-25 15:20:42  
**Overall Status**: OK

## Problems Dataset Integrity

**Status**: OK

- **Problems Count**: 21
- **File Hash**: 12b0d72ed831ebeeeadfd8cebf889851b48de2a0b3789c31d20c1e5e818f2327

## Test Suite Integrity

**Status**: OK

- **Test Files Count**: 10

### Test Files
- **test_st_0001.py**: 3.9 KB (hash: 1286bf47...)
- **test_st_0002.py**: 4.0 KB (hash: de94b0be...)
- **test_st_0003.py**: 4.3 KB (hash: 6074c91f...)
- **test_st_0005.py**: 1.5 KB (hash: 916fe5f5...)
- **test_st_0006.py**: 4.4 KB (hash: e56c18bc...)
- **test_st_0007.py**: 1.8 KB (hash: 5a5e53f8...)
- **test_st_0019.py**: 6.1 KB (hash: 5b5789fc...)
- **test_st_0004.js**: 3.3 KB (hash: 22847c03...)
- **test_st_0018.js**: 3.0 KB (hash: c9970b63...)
- **test_st_0014.java**: 5.6 KB (hash: 00c0f545...)

## Documentation Integrity

**Status**: OK

**Documentation Files**:
- **README.md**: 6.0 KB (hash: 31087bde...)
- **LICENSING_COMPLIANCE.md**: 6.7 KB (hash: 0b1c50cf...)
- **validation_report.md**: 1.3 KB (hash: 22ba3816...)

## Configuration Integrity

**Status**: OK

**Configuration Files**:
- **context_configs.json**: 0.7 KB (hash: 624ed405...)
- **single_turn_scenarios_suite.yaml**: 2.0 KB (hash: 1e4e21eb...)

## Recommendations

✅ All integrity checks passed. The task suite is ready for use.

## Verification Commands

To verify integrity manually:

```bash
# Validate problems schema and metadata
python validate_problems.py

# Check test coverage and syntax
python tests/validate_tests.py

# Run comprehensive test suite
python tests/run_all_tests.py

# Generate fresh integrity report
python check_integrity.py
```

---
*This report was generated automatically by the integrity checker.*
