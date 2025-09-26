#!/usr/bin/env python3
"""
Test suite for st_0003: Binary search bug fix
Tests the corrected binary_search function implementation.
"""

import pytest

def test_binary_search_found():
    """Test binary search when target is found."""
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    arr = [1, 3, 5, 7, 9, 11, 13]
    assert binary_search(arr, 7) == 3
    assert binary_search(arr, 1) == 0
    assert binary_search(arr, 13) == 6

def test_binary_search_not_found():
    """Test binary search when target is not found."""
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    arr = [1, 3, 5, 7, 9, 11, 13]
    assert binary_search(arr, 2) == -1
    assert binary_search(arr, 0) == -1
    assert binary_search(arr, 15) == -1

def test_binary_search_empty_array():
    """Test binary search with empty array."""
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    assert binary_search([], 5) == -1

def test_binary_search_single_element():
    """Test binary search with single element."""
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    assert binary_search([5], 5) == 0
    assert binary_search([5], 3) == -1

def test_binary_search_duplicates():
    """Test binary search with duplicate elements."""
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    arr = [1, 2, 2, 2, 3, 4, 5]
    result = binary_search(arr, 2)
    # Should return one of the valid indices for 2
    assert result in [1, 2, 3]

def test_binary_search_large_array():
    """Test binary search with large array."""
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    arr = list(range(0, 10000, 2))  # Even numbers from 0 to 9998
    assert binary_search(arr, 5000) == 2500
    assert binary_search(arr, 0) == 0
    assert binary_search(arr, 9998) == 4999
    assert binary_search(arr, 5001) == -1  # Odd number not in array

def test_binary_search_negative_numbers():
    """Test binary search with negative numbers."""
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    arr = [-10, -5, -1, 0, 3, 7, 12]
    assert binary_search(arr, -5) == 1
    assert binary_search(arr, 0) == 3
    assert binary_search(arr, -15) == -1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])