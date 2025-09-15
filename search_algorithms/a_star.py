"""
A* Search Algorithm Implementation
Author: AI Lab
Description: A* is an informed search algorithm that uses heuristics to find optimal paths
"""

import heapq
import math


class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Distance from start node
        self.h = 0  # Heuristic (distance to goal)
        self.f = 0  # Total cost (g + h)
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return self.position == other.position


class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
    
    def heuristic(self, pos1, pos2):
        """Calculate Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_neighbors(self, position):
        """Get valid neighboring positions"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        for dx, dy in directions:
            new_x, new_y = position[0] + dx, position[1] + dy
            
            # Check if position is valid and not an obstacle
            if (0 <= new_x < self.rows and 
                0 <= new_y < self.cols and 
                self.grid[new_x][new_y] != 1):  # 1 represents obstacle
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def search(self, start, goal):
        """
        Perform A* search from start to goal
        Returns path as list of positions, or None if no path exists
        """
        open_list = []
        closed_list = set()
        
        start_node = Node(start)
        start_node.g = 0
        start_node.h = self.heuristic(start, goal)
        start_node.f = start_node.g + start_node.h
        
        heapq.heappush(open_list, start_node)
        
        while open_list:
            current_node = heapq.heappop(open_list)
            closed_list.add(current_node.position)
            
            # Check if we reached the goal
            if current_node.position == goal:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]  # Reverse to get path from start to goal
            
            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current_node.position):
                if neighbor_pos in closed_list:
                    continue
                
                neighbor_node = Node(neighbor_pos, current_node)
                neighbor_node.g = current_node.g + 1
                neighbor_node.h = self.heuristic(neighbor_pos, goal)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                
                # Check if this neighbor is already in open list with lower f
                in_open_list = False
                for open_node in open_list:
                    if open_node.position == neighbor_pos and open_node.f <= neighbor_node.f:
                        in_open_list = True
                        break
                
                if not in_open_list:
                    heapq.heappush(open_list, neighbor_node)
        
        return None  # No path found


def demonstrate_astar():
    """Demonstrate A* algorithm with a grid-based pathfinding problem"""
    print("=== A* Search Algorithm Demonstration ===")
    
    # Create a grid where 0 = free space, 1 = obstacle
    grid = [
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    print("\nGrid (0 = free, 1 = obstacle):")
    for row in grid:
        print(' '.join(map(str, row)))
    
    astar = AStar(grid)
    start = (0, 0)
    goal = (6, 7)
    
    print(f"\nFinding path from {start} to {goal}...")
    path = astar.search(start, goal)
    
    if path:
        print(f"Path found: {path}")
        print(f"Path length: {len(path)}")
        
        # Visualize the path
        print("\nPath visualization (* = path, # = obstacle, . = free):")
        visual_grid = [['.' if cell == 0 else '#' for cell in row] for row in grid]
        
        for pos in path:
            visual_grid[pos[0]][pos[1]] = '*'
        
        visual_grid[start[0]][start[1]] = 'S'  # Start
        visual_grid[goal[0]][goal[1]] = 'G'    # Goal
        
        for row in visual_grid:
            print(' '.join(row))
    else:
        print("No path found!")


if __name__ == "__main__":
    demonstrate_astar()