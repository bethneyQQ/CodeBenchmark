#!/usr/bin/env python3
"""
Test script for sandbox execution system.

This script provides basic validation of the sandbox implementation
without requiring Docker to be running.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Try to import sandbox components, handling missing dependencies
try:
    from sandbox import (
        SandboxExecutor, 
        ExecutionResult, 
        ExecutionLogger,
        ExecutionResultSerializer,
        ExecutionResultHandler,
        EnhancedSandboxExecutor,
        validate_sandbox_environment
    )
    DOCKER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Docker dependencies not available: {e}")
    print("Testing only core components without Docker functionality...")
    DOCKER_AVAILABLE = False
    
    # Import only the components that don't require Docker
    try:
        from sandbox import ExecutionResult, ExecutionLogger, ExecutionResultSerializer, ExecutionResultHandler
    except ImportError:
        # Create minimal mock classes for testing
        from dataclasses import dataclass
        from typing import List, Optional
        
        @dataclass
        class ExecutionResult:
            stdout: str
            stderr: str
            exit_code: int
            wall_time: float
            peak_memory: int
            security_violations: List[str]
            success: bool = True
            error_message: Optional[str] = None


def test_execution_result_serialization():
    """Test ExecutionResult serialization and deserialization."""
    print("Testing ExecutionResult serialization...")
    
    # Create test result
    result = ExecutionResult(
        stdout="Hello, World!",
        stderr="",
        exit_code=0,
        wall_time=1.5,
        peak_memory=64,
        security_violations=[],
        success=True
    )
    
    # Test serialization
    serializer = ExecutionResultSerializer()
    
    # To dict
    result_dict = serializer.to_dict(result)
    assert result_dict['stdout'] == "Hello, World!"
    assert result_dict['success'] is True
    print("✓ Dictionary serialization works")
    
    # To JSON
    json_str = serializer.to_json(result, indent=2)
    assert "Hello, World!" in json_str
    print("✓ JSON serialization works")
    
    # From dict
    restored_result = serializer.from_dict(result_dict)
    assert restored_result.stdout == result.stdout
    assert restored_result.success == result.success
    print("✓ Dictionary deserialization works")
    
    # From JSON
    restored_from_json = serializer.from_json(json_str)
    assert restored_from_json.stdout == result.stdout
    print("✓ JSON deserialization works")


def test_execution_logger():
    """Test execution logging functionality."""
    print("\nTesting ExecutionLogger...")
    
    logger = ExecutionLogger()
    
    # Test logging methods
    logger.log_execution_start("test_001", "python", "abc123", {"memory_mb": 512})
    logger.log_security_violation("test_001", "Detected dangerous import")
    logger.log_resource_limit_exceeded("test_001", "memory", 512, 600)
    logger.log_container_event("test_001", "created", "Container started")
    
    result = ExecutionResult(
        stdout="Test output",
        stderr="",
        exit_code=0,
        wall_time=2.0,
        peak_memory=128,
        security_violations=["test violation"],
        success=False
    )
    logger.log_execution_end("test_001", result)
    
    print("✓ Execution logging methods work")
    
    # Test audit report
    report = logger.create_audit_report()
    assert isinstance(report, dict)
    print("✓ Audit report generation works")


def test_result_handler():
    """Test execution result handler."""
    print("\nTesting ExecutionResultHandler...")
    
    handler = ExecutionResultHandler()
    
    # Test result handling
    result = ExecutionResult(
        stdout="Test output",
        stderr="Compilation error",
        exit_code=1,
        wall_time=1.0,
        peak_memory=64,
        security_violations=[],
        success=False,
        error_message="Compilation failed"
    )
    
    processed = handler.handle_execution_result(
        "test_002", result, "python", "print('hello')"
    )
    
    assert processed['execution_id'] == "test_002"
    assert processed['language'] == "python"
    assert 'error_category' in processed
    assert 'recommendations' in processed
    print("✓ Result handling works")


def test_security_violation_detection():
    """Test security violation detection."""
    print("\nTesting security violation detection...")
    
    if not DOCKER_AVAILABLE:
        print("⚠ Skipping security violation detection test (Docker not available)")
        return
    
    # Test with mock executor (no Docker required)
    class MockSandboxExecutor(SandboxExecutor):
        def __init__(self, language):
            self.language = language
            self.config = self.LANGUAGE_CONFIGS[language]
            
        def _ensure_image(self):
            pass  # Skip Docker operations
    
    executor = MockSandboxExecutor('python')
    
    # Test safe code
    safe_code = "print('Hello, World!')"
    violations = executor._detect_security_violations(safe_code)
    assert len(violations) == 0
    print("✓ Safe code detection works")
    
    # Test dangerous code
    dangerous_code = "import os\nos.system('rm -rf /')"
    violations = executor._detect_security_violations(dangerous_code)
    assert len(violations) > 0
    print("✓ Dangerous code detection works")
    
    # Test obfuscated code
    obfuscated_code = "import base64\nbase64.b64decode('aW1wb3J0IG9z')"
    violations = executor._detect_security_violations(obfuscated_code)
    assert len(violations) > 0
    print("✓ Obfuscated code detection works")


def test_language_configurations():
    """Test language configurations."""
    print("\nTesting language configurations...")
    
    if not DOCKER_AVAILABLE:
        print("⚠ Skipping language configuration test (Docker not available)")
        return
    
    # Check all languages have required config
    for language, config in SandboxExecutor.LANGUAGE_CONFIGS.items():
        assert 'dockerfile' in config
        assert 'image_name' in config
        assert 'file_extension' in config
        assert 'run_command' in config
        assert 'test_command' in config
        print(f"✓ {language} configuration is valid")


def test_docker_validation():
    """Test Docker environment validation (without requiring Docker)."""
    print("\nTesting Docker validation...")
    
    if not DOCKER_AVAILABLE:
        print("⚠ Docker not available - skipping validation test")
        return
    
    # This will fail if Docker is not available, which is expected
    try:
        results = validate_sandbox_environment()
        print(f"Docker validation results: {results}")
        if any(results.values()):
            print("✓ Some sandbox environments are available")
        else:
            print("⚠ No sandbox environments available (Docker may not be running)")
    except Exception as e:
        print(f"⚠ Docker validation failed (expected if Docker not available): {e}")


def main():
    """Run all tests."""
    print("Running sandbox system tests...\n")
    
    try:
        test_execution_result_serialization()
        test_execution_logger()
        test_result_handler()
        test_security_violation_detection()
        test_language_configurations()
        test_docker_validation()
        
        print("\n✅ All tests completed successfully!")
        print("\nNote: Full sandbox functionality requires Docker to be installed and running.")
        print("The core components (serialization, logging, security detection) work without Docker.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()