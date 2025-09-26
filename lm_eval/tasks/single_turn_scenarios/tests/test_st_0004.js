#!/usr/bin/env node
/**
 * Test suite for st_0004: Python to JavaScript translation
 * Tests the calculateFactorial function implementation.
 */

// Simple test framework for Node.js
function assert(condition, message) {
    if (!condition) {
        throw new Error(message || 'Assertion failed');
    }
}

function assertEqual(actual, expected, message) {
    if (actual !== expected) {
        throw new Error(message || `Expected ${expected}, but got ${actual}`);
    }
}

// Reference implementation for testing
function calculateFactorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * calculateFactorial(n - 1);
}

function testFactorialBasic() {
    console.log('Testing basic factorial calculations...');
    assertEqual(calculateFactorial(0), 1, 'Factorial of 0 should be 1');
    assertEqual(calculateFactorial(1), 1, 'Factorial of 1 should be 1');
    assertEqual(calculateFactorial(2), 2, 'Factorial of 2 should be 2');
    assertEqual(calculateFactorial(3), 6, 'Factorial of 3 should be 6');
    assertEqual(calculateFactorial(4), 24, 'Factorial of 4 should be 24');
    assertEqual(calculateFactorial(5), 120, 'Factorial of 5 should be 120');
    console.log('✓ Basic factorial tests passed');
}

function testFactorialLarger() {
    console.log('Testing larger factorial calculations...');
    assertEqual(calculateFactorial(6), 720, 'Factorial of 6 should be 720');
    assertEqual(calculateFactorial(7), 5040, 'Factorial of 7 should be 5040');
    assertEqual(calculateFactorial(10), 3628800, 'Factorial of 10 should be 3628800');
    console.log('✓ Larger factorial tests passed');
}

function testFactorialEdgeCases() {
    console.log('Testing edge cases...');
    
    // Test negative numbers (should return 1 based on the condition n <= 1)
    assertEqual(calculateFactorial(-1), 1, 'Factorial of -1 should be 1');
    assertEqual(calculateFactorial(-5), 1, 'Factorial of -5 should be 1');
    
    console.log('✓ Edge case tests passed');
}

function testFactorialType() {
    console.log('Testing return type...');
    assert(typeof calculateFactorial(5) === 'number', 'Factorial should return a number');
    assert(Number.isInteger(calculateFactorial(5)), 'Factorial should return an integer');
    console.log('✓ Type tests passed');
}

function testFactorialPerformance() {
    console.log('Testing performance with moderate input...');
    const start = Date.now();
    const result = calculateFactorial(12);
    const end = Date.now();
    
    assertEqual(result, 479001600, 'Factorial of 12 should be 479001600');
    assert(end - start < 100, 'Factorial calculation should be fast');
    console.log('✓ Performance tests passed');
}

// Run all tests
function runTests() {
    console.log('Running JavaScript factorial tests...\n');
    
    try {
        testFactorialBasic();
        testFactorialLarger();
        testFactorialEdgeCases();
        testFactorialType();
        testFactorialPerformance();
        
        console.log('\n🎉 All tests passed!');
        process.exit(0);
    } catch (error) {
        console.error('\n❌ Test failed:', error.message);
        process.exit(1);
    }
}

// Run tests if this file is executed directly
if (require.main === module) {
    runTests();
}

module.exports = {
    calculateFactorial,
    runTests
};