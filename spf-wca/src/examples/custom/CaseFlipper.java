/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;


public class CaseFlipper {
    public static int flipCount(char[] chars) {
        int flips = 0;
        for (char c : chars) {
            if (c >= 'a' && c <= 'z') {
                flips++;
            } else if (c >= 'A' && c <= 'Z') {
                flips++;
            }
        }
        return flips;
    }

    public static void main(String[] args) {
        int N = Integer.parseInt(args[0]);
        char[] input = new char[N];
        for (int i = 0; i < N; i++) {
            input[i] = Debug.makeSymbolicChar("ch" + i);
        }
        flipCount(input);
    }
}
