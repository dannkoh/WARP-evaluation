/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class BinarySearchTreeHeight {
    static class Node {
        int value;
        Node left, right;
        Node(int v) { value = v; }
    }
    
    static class BST {
        Node root;
        
        public void insert(int v) {
            root = insertRec(root, v);
        }
        
        private Node insertRec(Node root, int v) {
            if (root == null) {
                return new Node(v);
            }
            if (v < root.value) {
                root.left = insertRec(root.left, v);
            } else if (v > root.value) {
                root.right = insertRec(root.right, v);
            }
            return root;
        }
        
        public int height() {
            return heightRec(root);
        }
        
        private int heightRec(Node node) {
            if (node == null) return 0;
            int leftHeight = heightRec(node.left);
            int rightHeight = heightRec(node.right);
            return 1 + Math.max(leftHeight, rightHeight);
        }
    }
    
    public static void main(String[] args) {
        final int N = Integer.parseInt(args[0]);
        BST tree = new BST();
        for (int i = 0; i < N; i++){
            int val = Debug.makeSymbolicInteger("in" + i);
            tree.insert(val);
        }
        
        tree.height();
    }
}
