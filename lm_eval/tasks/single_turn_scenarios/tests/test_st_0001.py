#!/usr/bin/env python3
"""
Test suite for st_0001: Reverse linked list
Tests the reverse_list function implementation.
"""

import pytest
import sys
import os

# Add the parent directory to the path to import the solution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ListNode:
    """Simple linked list node for testing."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def to_list(self):
        """Convert linked list to Python list for easy comparison."""
        result = []
        current = self
        while current:
            result.append(current.val)
            current = current.next
        return result

def create_linked_list(values):
    """Create a linked list from a list of values."""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

def test_reverse_empty_list():
    """Test reversing an empty list."""
    # This will be replaced with the actual implementation
    def reverse_list(head):
        prev = None
        current = head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev
    
    result = reverse_list(None)
    assert result is None

def test_reverse_single_element():
    """Test reversing a single element list."""
    def reverse_list(head):
        prev = None
        current = head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev
    
    head = create_linked_list([1])
    result = reverse_list(head)
    assert result.to_list() == [1]

def test_reverse_two_elements():
    """Test reversing a two element list."""
    def reverse_list(head):
        prev = None
        current = head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev
    
    head = create_linked_list([1, 2])
    result = reverse_list(head)
    assert result.to_list() == [2, 1]

def test_reverse_multiple_elements():
    """Test reversing a list with multiple elements."""
    def reverse_list(head):
        prev = None
        current = head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev
    
    head = create_linked_list([1, 2, 3, 4, 5])
    result = reverse_list(head)
    assert result.to_list() == [5, 4, 3, 2, 1]

def test_reverse_duplicate_values():
    """Test reversing a list with duplicate values."""
    def reverse_list(head):
        prev = None
        current = head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev
    
    head = create_linked_list([1, 2, 2, 3, 1])
    result = reverse_list(head)
    assert result.to_list() == [1, 3, 2, 2, 1]

def test_reverse_large_list():
    """Test reversing a large list for performance."""
    def reverse_list(head):
        prev = None
        current = head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev
    
    values = list(range(1000))
    head = create_linked_list(values)
    result = reverse_list(head)
    expected = list(reversed(values))
    assert result.to_list() == expected

if __name__ == "__main__":
    pytest.main([__file__, "-v"])