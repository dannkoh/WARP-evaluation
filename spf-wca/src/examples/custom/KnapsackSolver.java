/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class KnapsackSolver {
    public static int knapsack(int[] weights, int[] values, int n, int capacity) {
        if (n == 0 || capacity == 0) return 0;
        if (weights[n-1] > capacity) {
            return knapsack(weights, values, n - 1, capacity);
        } else {
            int include = values[n-1] + knapsack(weights, values, n - 1, capacity - weights[n-1]);
            int exclude = knapsack(weights, values, n - 1, capacity);
            return (include > exclude) ? include : exclude;
        }
    }
    
    public static void main(String[] args) {
        int n = 3; // Fixed number of items.
        int[] weights = new int[n];
        int[] values = new int[n];
        
        for (int i = 0; i < n; i++){
            weights[i] = Debug.makeSymbolicInteger("w" + i);
            values[i] = Debug.makeSymbolicInteger("v" + i);
        }
        
        int capacity = Debug.makeSymbolicInteger("c");

        
        knapsack(weights, values, n, capacity);

    }
}
