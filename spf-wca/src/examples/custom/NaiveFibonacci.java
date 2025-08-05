/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class NaiveFibonacci {
    /**
     * Computes the Fibonacci number for n recursively.
     * This naive recursion leads to exponential growth in function calls.
     */
    public static long fib(int n) {
        if(n <= 1) {
            return n;
        }
        return fib(n - 1) + fib(n - 2);
    }

    public static void main(String[] args) {
        // Create a symbolic integer for n.
        int n = Debug.makeSymbolicInteger("in");
        // Constrain n to small values to avoid deep recursion.
        if(n < 0 || n > 10) {
            return;
        }
        fib(n);
    }
}
