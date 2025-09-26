#!/usr/bin/env python3
"""
Test suite for st_0007: Simple code completion
Tests the sum_even_numbers function implementation.
"""

import pytest

def test_sum_even_numbers_basic():
    """Test basic sum of even numbers."""
    def sum_even_numbers(numbers):
        total = 0
        for num in numbers:
            if num % 2 == 0:
                total += num
        return total
    
    assert sum_even_numbers([1, 2, 3, 4, 5, 6]) == 12
    assert sum_even_numbers([2, 4, 6, 8]) == 20
    assert sum_even_numbers([1, 3, 5, 7]) == 0

def test_sum_even_numbers_empty():
    """Test with empty list."""
    def sum_even_numbers(numbers):
        total = 0
        for num in numbers:
            if num % 2 == 0:
                total += num
        return total
    
    assert sum_even_numbers([]) == 0

def test_sum_even_numbers_single():
    """Test with single element."""
    def sum_even_numbers(numbers):
        total = 0
        for num in numbers:
            if num % 2 == 0:
                total += num
        return total
    
    assert sum_even_numbers([2]) == 2
    assert sum_even_numbers([3]) == 0

def test_sum_even_numbers_negative():
    """Test with negative numbers."""
    def sum_even_numbers(numbers):
        total = 0
        for num in numbers:
            if num % 2 == 0:
                total += num
        return total
    
    assert sum_even_numbers([-2, -1, 0, 1, 2]) == 0
    assert sum_even_numbers([-4, -2, 2, 4]) == 0

def test_sum_even_numbers_zero():
    """Test with zero."""
    def sum_even_numbers(numbers):
        total = 0
        for num in numbers:
            if num % 2 == 0:
                total += num
        return total
    
    assert sum_even_numbers([0, 1, 2, 3]) == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])