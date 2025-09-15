/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class ArrayTwister {
    public static int[] twist(int[] nums) {
        int[] result = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            int val = nums[i];
            if (val % 2 == 0) {
                result[i] = val / 2;
            } else if (val % 3 == 0) {
                result[i] = val * 2;
            } else {
                result[i] = val + 1;
            }
        }
        return result;
    }

    public static void main(String[] args) {
        int N = Integer.parseInt(args[0]);
        int[] input = new int[N];
        for (int i = 0; i < N; i++) {
            input[i] = Debug.makeSymbolicInteger("in" + i);
        }
        twist(input);
    }
}
