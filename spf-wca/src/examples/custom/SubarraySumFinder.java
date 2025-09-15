/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class SubarraySumFinder {
    public static boolean hasSubarrayWithSum(int[] arr, int target) {
        int n = arr.length;
        for (int i = 0; i < n; i++){
            int sum = 0;
            for (int j = i; j < n; j++){
                sum += arr[j];
                if (sum == target) return true;
            }
        }
        return false;
    }
    
    public static void main(String[] args) {
        int N = Integer.parseInt(args[0]);
        int[] arr = new int[N];
        for (int i = 0; i < N; i++){
            arr[i] = Debug.makeSymbolicInteger("in" + i);
        }
        
        int target = Debug.makeSymbolicInteger("t");
        // Constrain values for manageability.
        for (int i = 0; i < N; i++){
            if (arr[i] < -10 || arr[i] > 10) return;
        }
        if (target < -100 || target > 100) return;
        
        hasSubarrayWithSum(arr, target);
    }
}
