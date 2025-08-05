/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class Collatz {
    public static int run(int[] nums) {
        int maxSteps = 0;
        for (int num : nums) {
            int steps = 0;
            int n = num;
            while (n != 1 && n > 0) {
                if (n % 2 == 0) {
                    n = n / 2;
                } else {
                    n = 3 * n + 1;
                }
                steps++;
            }
            if (steps > maxSteps) {
                maxSteps = steps;
            }
        }
        return maxSteps;
    }

    public static void main(String[] args) {
        final int N = Integer.parseInt(args[0]);
        int[] nums = new int[N];
        for (int i = 0; i < N; i++) {
            nums[i] = Debug.makeSymbolicInteger("in" + i);
        }

        run(nums);
    }
}
