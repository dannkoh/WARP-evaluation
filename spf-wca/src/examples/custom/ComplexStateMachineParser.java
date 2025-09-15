/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class ComplexStateMachineParser {

    /**
     * Processes an input character array through a state machine with several transitions.
     */
    public static void parse(char[] input) {
        int state = 0;
        for (int i = 0; i < input.length; i++) {
            char c = input[i];
            switch (state) {
                case 0:
                    if (c == 'A') state = 1;
                    else if (c == 'B') state = 2;
                    else state = 0;
                    break;
                case 1:
                    if (c == 'C') state = 3;
                    else if (c == 'D') state = 4;
                    else state = 0;
                    break;
                case 2:
                    if (c == 'E') state = 3;
                    else if (c == 'F') state = 4;
                    else state = 0;
                    break;
                case 3:
                    if (c == 'G') state = 5;
                    else if (c == 'H') state = 6;
                    else state = 0;
                    break;
                case 4:
                    if (c == 'I') state = 5;
                    else if (c == 'J') state = 6;
                    else state = 0;
                    break;
                case 5:
                    if (c == 'K') state = 7;
                    else state = 5;
                    break;
                case 6:
                    if (c == 'L') state = 7;
                    else state = 6;
                    break;
                default:
                    state = 0;
                    break;
            }
        }
    }

    public static void main(String[] args) {
        int N = Integer.parseInt(args[0]);
        char[] input = new char[N];
        for (int i = 0; i < N; i++) {
            input[i] = Debug.makeSymbolicChar("in" + i);
        }
        parse(input);
    }
}
