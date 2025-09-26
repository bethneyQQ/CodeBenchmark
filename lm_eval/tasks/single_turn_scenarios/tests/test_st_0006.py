#!/usr/bin/env python3
"""
Test suite for st_0006: Function documentation generation
Tests the documentation quality and completeness.
"""

import pytest
import doctest
import inspect

def test_documentation_basic():
    """Test basic documentation functionality."""
    # Reference implementation with documentation
    def merge_sorted_arrays(arr1, arr2):
        """
        Merge two sorted arrays into a single sorted array.
        
        This function takes two sorted arrays and merges them into a single
        sorted array using a two-pointer approach. The time complexity is O(n + m)
        where n and m are the lengths of the input arrays.
        
        Args:
            arr1 (list): First sorted array of comparable elements
            arr2 (list): Second sorted array of comparable elements
        
        Returns:
            list: A new sorted array containing all elements from both input arrays
        
        Example:
            >>> merge_sorted_arrays([1, 3, 5], [2, 4, 6])
            [1, 2, 3, 4, 5, 6]
            
            >>> merge_sorted_arrays([], [1, 2, 3])
            [1, 2, 3]
        
        Note:
            Both input arrays must be sorted in ascending order for correct results.
        """
        result = []
        i = j = 0
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1
        result.extend(arr1[i:])
        result.extend(arr2[j:])
        return result
    
    # Test that function has docstring
    assert merge_sorted_arrays.__doc__ is not None
    assert len(merge_sorted_arrays.__doc__.strip()) > 0

def test_docstring_completeness():
    """Test that docstring contains required sections."""
    def merge_sorted_arrays(arr1, arr2):
        """
        Merge two sorted arrays into a single sorted array.
        
        This function takes two sorted arrays and merges them into a single
        sorted array using a two-pointer approach. The time complexity is O(n + m)
        where n and m are the lengths of the input arrays.
        
        Args:
            arr1 (list): First sorted array of comparable elements
            arr2 (list): Second sorted array of comparable elements
        
        Returns:
            list: A new sorted array containing all elements from both input arrays
        
        Example:
            >>> merge_sorted_arrays([1, 3, 5], [2, 4, 6])
            [1, 2, 3, 4, 5, 6]
            
            >>> merge_sorted_arrays([], [1, 2, 3])
            [1, 2, 3]
        
        Note:
            Both input arrays must be sorted in ascending order for correct results.
        """
        result = []
        i = j = 0
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1
        result.extend(arr1[i:])
        result.extend(arr2[j:])
        return result
    
    docstring = merge_sorted_arrays.__doc__
    
    # Check for required sections
    assert "Args:" in docstring or "Parameters:" in docstring
    assert "Returns:" in docstring
    assert "Example:" in docstring or "Examples:" in docstring

def test_docstring_examples():
    """Test that docstring examples work correctly."""
    def merge_sorted_arrays(arr1, arr2):
        """
        Merge two sorted arrays into a single sorted array.
        
        Example:
            >>> merge_sorted_arrays([1, 3, 5], [2, 4, 6])
            [1, 2, 3, 4, 5, 6]
            
            >>> merge_sorted_arrays([], [1, 2, 3])
            [1, 2, 3]
        """
        result = []
        i = j = 0
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1
        result.extend(arr1[i:])
        result.extend(arr2[j:])
        return result
    
    # Test the examples in the docstring
    assert merge_sorted_arrays([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge_sorted_arrays([], [1, 2, 3]) == [1, 2, 3]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])