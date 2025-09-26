#!/usr/bin/env node
/**
 * Test suite for st_0018: Simple code completion - JavaScript
 * Tests the findMax function implementation.
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
function findMax(arr) {
    if (arr.length === 0) return null;
    let max = arr[0];
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

function testFindMaxBasic() {
    console.log('Testing basic findMax functionality...');
    assertEqual(findMax([1, 2, 3, 4, 5]), 5, 'Should find maximum in ascending array');
    assertEqual(findMax([5, 4, 3, 2, 1]), 5, 'Should find maximum in descending array');
    assertEqual(findMax([3, 1, 4, 1, 5, 9, 2, 6]), 9, 'Should find maximum in mixed array');
    console.log('✓ Basic findMax tests passed');
}

function testFindMaxEdgeCases() {
    console.log('Testing edge cases...');
    assertEqual(findMax([]), null, 'Empty array should return null');
    assertEqual(findMax([42]), 42, 'Single element array should return that element');
    assertEqual(findMax([-1, -2, -3]), -1, 'Should work with negative numbers');
    assertEqual(findMax([0, 0, 0]), 0, 'Should work with duplicate values');
    console.log('✓ Edge case tests passed');
}

function testFindMaxTypes() {
    console.log('Testing with different number types...');
    assertEqual(findMax([1.5, 2.7, 1.2]), 2.7, 'Should work with decimals');
    assertEqual(findMax([-1.5, -0.5, -2.7]), -0.5, 'Should work with negative decimals');
    console.log('✓ Type tests passed');
}

function testFindMaxPerformance() {
    console.log('Testing performance with large array...');
    const largeArray = Array.from({length: 10000}, (_, i) => Math.random() * 1000);
    
    const start = Date.now();
    const result = findMax(largeArray);
    const end = Date.now();
    
    assert(typeof result === 'number', 'Should return a number');
    assert(end - start < 100, 'Should complete quickly for large arrays');
    console.log('✓ Performance tests passed');
}

// Run all tests
function runTests() {
    console.log('Running JavaScript findMax tests...\n');
    
    try {
        testFindMaxBasic();
        testFindMaxEdgeCases();
        testFindMaxTypes();
        testFindMaxPerformance();
        
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
    findMax,
    runTests
};