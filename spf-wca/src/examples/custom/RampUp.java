/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class RampUp {

    public static void algo(int[] arr) {
        int N = arr.length;
        boolean allPositive = true;

        for (int i = 0; i < N; i++) {
            if (arr[i] < 0) {
                allPositive = false;
            }
        }

        if (allPositive) {
            int dummy = 1;
            for (int i = 0; i < N * 500; i++) {
                dummy = dummy * dummy;
            }
        }
    }

    public static void main(String[] args) {
        int N = Integer.parseInt(args[0]);
        int[] arr = new int[N];
        for (int i = 0; i < N; i++) {
            arr[i] = Debug.makeSymbolicInteger("in" + i);
        }

        algo(arr);
    }
}
