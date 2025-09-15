/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class BinarySearch {
    public static int binarySearch(int[] arr, int target) {
        int low = 0;
        int high = arr.length - 1;
        while(low <= high) {
            int mid = (low + high) / 2;
            if(arr[mid] == target) {
                return mid;
            } else if(arr[mid] < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return -1;
    }
    
    public static void main(String[] args) {
        final int N = Integer.parseInt(args[0]);
        int[] arr = new int[N];
        for(int i = 0; i < N; i++){
            arr[i] = Debug.makeSymbolicInteger("in" + i);
        }
        
        int target = Debug.makeSymbolicInteger("t");
        binarySearch(arr, target);
    }
}
