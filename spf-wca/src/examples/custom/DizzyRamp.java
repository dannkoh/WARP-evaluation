/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class DizzyRamp {

    public static void spin(int[] arr) {
        int N = arr.length;
        boolean climbing = true;
        boolean spinning = false;

        for (int i = 1; i < N; i++) {
            if (arr[i] > arr[i - 1]) {
                if (!climbing) {
                    spinning = true;
                }
            } else if (arr[i] < arr[i - 1]) {
                climbing = false;
            } else {
                spinning = true;
            }
        }

        // Cost simulation: spin triggers a costly loop like SameHundred
        if (spinning) {
            int aa = 9999;
            for (int i = 0; i < N; i++) {
                aa = aa * aa; // Dummy heavy work
            }
        }
    }

    public static void main(String[] args) {
        final int N = Integer.parseInt(args[0]);
        int[] arr = new int[N];
        for (int i = 0; i < N; i++) {
            arr[i] = Debug.makeSymbolicInteger("in" + i);
        }

        // This method's cost will be analyzed symbolically
        spin(arr);
    }
}
