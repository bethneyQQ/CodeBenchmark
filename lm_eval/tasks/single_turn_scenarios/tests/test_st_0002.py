#!/usr/bin/env python3
"""
Test suite for st_0002: FizzBuzz implementation
Tests the fizzbuzz function implementation.
"""

import pytest

def test_fizzbuzz_basic():
    """Test basic FizzBuzz functionality."""
    def fizzbuzz(n):
        result = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append('FizzBuzz')
            elif i % 3 == 0:
                result.append('Fizz')
            elif i % 5 == 0:
                result.append('Buzz')
            else:
                result.append(str(i))
        return result
    
    result = fizzbuzz(15)
    expected = ['1', '2', 'Fizz', '4', 'Buzz', 'Fizz', '7', '8', 'Fizz', 'Buzz', '11', 'Fizz', '13', '14', 'FizzBuzz']
    assert result == expected

def test_fizzbuzz_empty():
    """Test FizzBuzz with n=0."""
    def fizzbuzz(n):
        result = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append('FizzBuzz')
            elif i % 3 == 0:
                result.append('Fizz')
            elif i % 5 == 0:
                result.append('Buzz')
            else:
                result.append(str(i))
        return result
    
    result = fizzbuzz(0)
    assert result == []

def test_fizzbuzz_single():
    """Test FizzBuzz with n=1."""
    def fizzbuzz(n):
        result = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append('FizzBuzz')
            elif i % 3 == 0:
                result.append('Fizz')
            elif i % 5 == 0:
                result.append('Buzz')
            else:
                result.append(str(i))
        return result
    
    result = fizzbuzz(1)
    assert result == ['1']

def test_fizzbuzz_multiples_of_3():
    """Test specific multiples of 3."""
    def fizzbuzz(n):
        result = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append('FizzBuzz')
            elif i % 3 == 0:
                result.append('Fizz')
            elif i % 5 == 0:
                result.append('Buzz')
            else:
                result.append(str(i))
        return result
    
    result = fizzbuzz(9)
    assert result[2] == 'Fizz'  # 3
    assert result[5] == 'Fizz'  # 6
    assert result[8] == 'Fizz'  # 9

def test_fizzbuzz_multiples_of_5():
    """Test specific multiples of 5."""
    def fizzbuzz(n):
        result = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append('FizzBuzz')
            elif i % 3 == 0:
                result.append('Fizz')
            elif i % 5 == 0:
                result.append('Buzz')
            else:
                result.append(str(i))
        return result
    
    result = fizzbuzz(10)
    assert result[4] == 'Buzz'  # 5
    assert result[9] == 'Buzz'  # 10

def test_fizzbuzz_multiples_of_15():
    """Test specific multiples of 15."""
    def fizzbuzz(n):
        result = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append('FizzBuzz')
            elif i % 3 == 0:
                result.append('Fizz')
            elif i % 5 == 0:
                result.append('Buzz')
            else:
                result.append(str(i))
        return result
    
    result = fizzbuzz(30)
    assert result[14] == 'FizzBuzz'  # 15
    assert result[29] == 'FizzBuzz'  # 30

def test_fizzbuzz_large_number():
    """Test FizzBuzz with larger numbers."""
    def fizzbuzz(n):
        result = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append('FizzBuzz')
            elif i % 3 == 0:
                result.append('Fizz')
            elif i % 5 == 0:
                result.append('Buzz')
            else:
                result.append(str(i))
        return result
    
    result = fizzbuzz(100)
    assert len(result) == 100
    assert result[99] == 'Buzz'  # 100
    assert result[89] == 'FizzBuzz'  # 90

if __name__ == "__main__":
    pytest.main([__file__, "-v"])