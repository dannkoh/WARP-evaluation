/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class GreedyStepper {

    public static void walk(int[] arr) {
        int i = 0;
        int steps = 0;

        while (i < arr.length) {
            if (arr[i] % 2 == 0 && i + 1 < arr.length) {
                i += 2; // bigger step if even
            } else {
                i += 1; // normal step
            }
            steps++;
        }

        // Cost simulated: more steps = more work
        for (int j = 0; j < steps * 100; j++) {
            int x = j * j;
        }
    }

    public static void main(String[] args) {
        int N = Integer.parseInt(args[0]);
        int[] arr = new int[N];
        for (int i = 0; i < N; i++) {
            arr[i] = Debug.makeSymbolicInteger("in" + i);
        }

        walk(arr);
    }
}
