# CI/CD Workflows for Single Turn Scenarios

This directory contains GitHub Actions workflows for automated testing and validation of the single_turn_scenarios evaluation task.

## Workflows

### 1. Main CI Pipeline (`main.yml`)
- Runs on every push and pull request
- Tests across multiple Python versions and operating systems
- Validates code quality and runs comprehensive test suite
- Checks security and compliance requirements

### 2. Model Backend Testing (`model-backends.yml`)
- Tests integration with all supported model backends
- Validates API connectivity and configuration
- Runs smoke tests for each model type

### 3. Security Validation (`security.yml`)
- Performs security audits and penetration testing
- Validates sandbox isolation and security measures
- Checks for security vulnerabilities in dependencies

### 4. Performance Testing (`performance.yml`)
- Runs scalability and performance tests
- Monitors resource usage and execution times
- Validates performance benchmarks

### 5. Deployment Validation (`deployment.yml`)
- Validates deployment readiness
- Runs final integration tests
- Checks environment setup procedures

## Usage

These workflows are automatically triggered by GitHub Actions when:
- Code is pushed to main branch
- Pull requests are created or updated
- Manual workflow dispatch is triggered
- Scheduled runs (for periodic validation)

## Configuration

Workflows use the following secrets and variables:
- `ANTHROPIC_API_KEY`: For Claude model testing
- `OPENAI_API_KEY`: For OpenAI model testing
- `DEEPSEEK_API_KEY`: For DeepSeek model testing
- `DOCKER_HUB_TOKEN`: For Docker image operations

## Local Testing

To run tests locally before pushing:

```bash
# Run unit tests
cd lm_eval/tasks/single_turn_scenarios
python -m pytest tests/unit/ -v

# Run integration tests
python -m pytest tests/integration/ -v

# Run security tests
python -m pytest tests/security/ -v

# Run performance tests
python -m pytest tests/performance/ -v
```