# Single Turn Scenarios Test Suite

This directory contains a comprehensive test suite for the single_turn_scenarios evaluation task, covering unit tests, integration tests, performance tests, and security tests.

## Test Structure

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_utils.py      # Tests for utils.py module
│   ├── test_metrics.py    # Tests for metrics.py module
│   ├── test_sandbox.py    # Tests for sandbox.py module
│   ├── test_config_manager.py  # Tests for config_manager.py module
│   ├── test_context_application.py  # Tests for context application
│   ├── test_runner.py     # Unit test runner
│   └── mock_objects.py    # Mock objects and fixtures
├── integration/           # Integration tests
│   ├── test_evaluation_pipeline.py  # End-to-end pipeline tests
│   ├── test_multi_model_backends.py  # Multi-model backend tests
│   └── test_cross_language_support.py  # Cross-language tests
├── performance/           # Performance and scalability tests
│   └── test_scalability.py  # Scalability and resource usage tests
├── security/              # Security tests
│   ├── test_sandbox_security.py  # Sandbox security tests
│   └── test_timeout_handling.py  # Timeout and resource exhaustion tests
├── run_all_tests.py      # Comprehensive test runner
├── pytest.ini           # Pytest configuration
└── README.md            # This file
```

## Test Categories

### Unit Tests
- **Purpose**: Test individual components in isolation
- **Coverage**: utils.py, metrics.py, sandbox.py, config_manager.py
- **Features**: Mock objects, isolated testing, fast execution
- **Run with**: `python -m pytest tests/unit/ -m "not integration"`

### Integration Tests
- **Purpose**: Test component interactions and end-to-end workflows
- **Coverage**: Complete evaluation pipeline, multi-model backends, cross-language support
- **Features**: Real component integration, workflow validation
- **Run with**: `python -m pytest tests/integration/ -m integration`

### Performance Tests
- **Purpose**: Test scalability, resource usage, and performance regression
- **Coverage**: Dataset loading, concurrent processing, memory usage, CPU usage
- **Features**: Benchmark tests, resource monitoring, regression detection
- **Run with**: `python -m pytest tests/performance/ -m performance`

### Security Tests
- **Purpose**: Test security measures and malicious code detection
- **Coverage**: Sandbox escape attempts, malicious code detection, timeout handling
- **Features**: Security violation detection, resource exhaustion prevention
- **Run with**: `python -m pytest tests/security/ -m security`

## Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_all_tests.py

# Run specific category
python tests/run_all_tests.py --categories unit integration

# Run with coverage
python tests/run_all_tests.py --coverage

# Run smoke tests (fast validation)
python tests/run_all_tests.py --smoke

# Generate detailed report
python tests/run_all_tests.py --report --output test_report.txt
```

### Using pytest directly
```bash
# Run unit tests
python -m pytest tests/unit/ -v

# Run integration tests
python -m pytest tests/integration/ -v -m integration

# Run performance tests
python -m pytest tests/performance/ -v -m performance

# Run security tests
python -m pytest tests/security/ -v -m security

# Run with coverage
python -m pytest tests/unit/ tests/integration/ --cov=../ --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_utils.py -v

# Run specific test function
python -m pytest tests/unit/test_utils.py::TestLoadDataset::test_load_dataset_success -v
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.security` - Security tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.benchmark` - Benchmark tests

## Mock Objects and Fixtures

The test suite includes comprehensive mock objects in `unit/mock_objects.py`:

- `MockExecutionResult` - Mock sandbox execution results
- `MockSandboxExecutor` - Mock sandbox executor
- `MockDataset` - Mock dataset for testing
- `MockConfigManager` - Mock configuration manager
- `MockMetricsCalculator` - Mock metrics calculator

## Test Configuration

### pytest.ini
- Configures test discovery, markers, and output options
- Sets timeout limits and coverage options
- Defines exclusion patterns for coverage

### Dependencies
Tests require the following packages:
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `psutil` - System resource monitoring (for performance tests)

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-mock psutil
```

## Continuous Integration

The test suite is designed for CI/CD integration:

### GitHub Actions Example
```yaml
- name: Run Tests
  run: |
    python tests/run_all_tests.py --categories unit integration
    
- name: Run Performance Tests
  run: |
    python tests/run_all_tests.py --categories performance
    
- name: Generate Coverage Report
  run: |
    python tests/run_all_tests.py --coverage
```

### Test Reports
- JUnit XML reports are generated for each test category
- Coverage reports in HTML and XML formats
- Detailed test summaries and performance metrics

## Best Practices

### Writing Tests
1. **Isolation**: Unit tests should be isolated and not depend on external resources
2. **Mocking**: Use mock objects for external dependencies
3. **Assertions**: Include clear, specific assertions
4. **Documentation**: Document test purpose and expected behavior
5. **Performance**: Keep unit tests fast, mark slow tests appropriately

### Test Data
1. **Fixtures**: Use pytest fixtures for reusable test data
2. **Mock Data**: Create realistic but minimal mock data
3. **Edge Cases**: Test boundary conditions and error cases
4. **Security**: Include tests for security vulnerabilities

### Maintenance
1. **Regular Updates**: Keep tests updated with code changes
2. **Coverage Monitoring**: Maintain high test coverage (>70%)
3. **Performance Monitoring**: Watch for performance regressions
4. **Security Updates**: Regularly update security tests

## Troubleshooting

### Common Issues

1. **Docker Not Available**
   - Some integration and security tests require Docker
   - Tests will skip gracefully if Docker is not available
   - Install Docker for complete test coverage

2. **Import Errors**
   - Ensure the parent directory is in Python path
   - Tests automatically add the parent directory to sys.path

3. **Timeout Issues**
   - Performance and security tests may timeout on slow systems
   - Adjust timeout values in pytest.ini if needed

4. **Permission Errors**
   - Some tests create temporary files
   - Ensure write permissions in the test directory

### Debug Mode
Run tests with additional debugging:
```bash
# Verbose output with full tracebacks
python -m pytest tests/unit/ -vvv --tb=long

# Stop on first failure
python -m pytest tests/unit/ -x

# Run specific test with debugging
python -m pytest tests/unit/test_utils.py::TestLoadDataset::test_load_dataset_success -vvv -s
```

## Contributing

When adding new functionality:

1. **Add Unit Tests**: Create unit tests for new functions/classes
2. **Add Integration Tests**: Test integration with existing components
3. **Update Mocks**: Update mock objects if interfaces change
4. **Performance Tests**: Add performance tests for critical paths
5. **Security Tests**: Add security tests for new execution paths
6. **Documentation**: Update this README with new test information

## Test Coverage Goals

- **Unit Tests**: >90% line coverage
- **Integration Tests**: >80% workflow coverage
- **Performance Tests**: All critical performance paths
- **Security Tests**: All security-sensitive operations

Current coverage can be viewed by running:
```bash
python tests/run_all_tests.py --coverage
open coverage_html/index.html
```