/**
 * @author Daniel Koh <daniel.koh@student.manchester.ac.uk>
 */
package custom;

import gov.nasa.jpf.symbc.Debug;

public class MazeSolver {
    static final int N = 3;
    static boolean[][] visited = new boolean[N][N];
    
    public static boolean solveMaze(int[][] maze, int x, int y) {
        if(x == N-1 && y == N-1) return maze[x][y] == 0;
        if(x < 0 || x >= N || y < 0 || y >= N) return false;
        if(maze[x][y] == 1 || visited[x][y]) return false;
        
        visited[x][y] = true;
        
        // Explore all four directions.
        if(solveMaze(maze, x+1, y)) return true;
        if(solveMaze(maze, x-1, y)) return true;
        if(solveMaze(maze, x, y+1)) return true;
        if(solveMaze(maze, x, y-1)) return true;
        
        return false;
    }
    
    public static void main(String[] args) {
        int[][] maze = new int[N][N];
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                maze[i][j] = Debug.makeSymbolicInteger("cell_" + i + "_" + j);
                // Constrain each cell to be either 0 (free) or 1 (wall).
                if(maze[i][j] != 0 && maze[i][j] != 1) return;
            }
        }
        
        // Ensure the start and goal cells are free.
        maze[0][0] = 0;
        maze[N-1][N-1] = 0;
        
        solveMaze(maze, 0, 0);
    }
}
