/**
 * Test suite for st_0014: Java algorithm implementation
 * Tests the BinarySearchTree implementation.
 */

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

public class TestST0014 {
    
    private BinarySearchTree bst;
    
    @BeforeEach
    void setUp() {
        bst = new BinarySearchTree();
    }
    
    @Test
    @DisplayName("Test basic BST insertion and search")
    void testBasicInsertionAndSearch() {
        bst.insert(5);
        bst.insert(3);
        bst.insert(7);
        bst.insert(1);
        bst.insert(9);
        
        assertTrue(bst.search(5), "Should find root element");
        assertTrue(bst.search(3), "Should find left child");
        assertTrue(bst.search(7), "Should find right child");
        assertTrue(bst.search(1), "Should find leaf node");
        assertTrue(bst.search(9), "Should find leaf node");
        
        assertFalse(bst.search(2), "Should not find non-existent element");
        assertFalse(bst.search(10), "Should not find non-existent element");
    }
    
    @Test
    @DisplayName("Test BST deletion")
    void testDeletion() {
        // Insert test data
        bst.insert(5);
        bst.insert(3);
        bst.insert(7);
        bst.insert(1);
        bst.insert(4);
        bst.insert(6);
        bst.insert(9);
        
        // Test deletion of leaf node
        bst.delete(1);
        assertFalse(bst.search(1), "Deleted leaf node should not be found");
        
        // Test deletion of node with one child
        bst.delete(3);
        assertFalse(bst.search(3), "Deleted node should not be found");
        assertTrue(bst.search(4), "Child of deleted node should still exist");
        
        // Test deletion of node with two children
        bst.delete(5);
        assertFalse(bst.search(5), "Deleted root should not be found");
        assertTrue(bst.search(7), "Other nodes should still exist");
    }
    
    @Test
    @DisplayName("Test empty BST operations")
    void testEmptyBST() {
        assertFalse(bst.search(1), "Empty BST should not contain any elements");
        
        // Deleting from empty BST should not cause errors
        assertDoesNotThrow(() -> bst.delete(1), "Deleting from empty BST should not throw");
    }
    
    @Test
    @DisplayName("Test BST with duplicate values")
    void testDuplicateValues() {
        bst.insert(5);
        bst.insert(5); // Insert duplicate
        
        assertTrue(bst.search(5), "Should find the value");
        
        bst.delete(5);
        // Behavior with duplicates depends on implementation
        // This test assumes duplicates are not inserted
    }
    
    @Test
    @DisplayName("Test BST performance with many elements")
    void testPerformance() {
        long startTime = System.currentTimeMillis();
        
        // Insert many elements
        for (int i = 0; i < 1000; i++) {
            bst.insert(i);
        }
        
        // Search for elements
        for (int i = 0; i < 1000; i++) {
            assertTrue(bst.search(i), "Should find inserted element " + i);
        }
        
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        
        assertTrue(duration < 1000, "Operations should complete within reasonable time");
    }
}

// Reference BST implementation for testing
class BinarySearchTree {
    private Node root;
    
    private class Node {
        int data;
        Node left, right;
        
        Node(int data) {
            this.data = data;
            left = right = null;
        }
    }
    
    public void insert(int data) {
        root = insertRec(root, data);
    }
    
    private Node insertRec(Node root, int data) {
        if (root == null) {
            root = new Node(data);
            return root;
        }
        
        if (data < root.data) {
            root.left = insertRec(root.left, data);
        } else if (data > root.data) {
            root.right = insertRec(root.right, data);
        }
        
        return root;
    }
    
    public boolean search(int data) {
        return searchRec(root, data);
    }
    
    private boolean searchRec(Node root, int data) {
        if (root == null) {
            return false;
        }
        
        if (data == root.data) {
            return true;
        }
        
        return data < root.data ? searchRec(root.left, data) : searchRec(root.right, data);
    }
    
    public void delete(int data) {
        root = deleteRec(root, data);
    }
    
    private Node deleteRec(Node root, int data) {
        if (root == null) {
            return root;
        }
        
        if (data < root.data) {
            root.left = deleteRec(root.left, data);
        } else if (data > root.data) {
            root.right = deleteRec(root.right, data);
        } else {
            if (root.left == null) {
                return root.right;
            } else if (root.right == null) {
                return root.left;
            }
            
            root.data = minValue(root.right);
            root.right = deleteRec(root.right, root.data);
        }
        
        return root;
    }
    
    private int minValue(Node root) {
        int minValue = root.data;
        while (root.left != null) {
            minValue = root.left.data;
            root = root.left;
        }
        return minValue;
    }
}