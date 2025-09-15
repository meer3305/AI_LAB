"""
Breadth-First Search (BFS) Algorithm Implementation
Author: AI Lab
Description: BFS is a graph traversal algorithm that explores neighbors before moving to the next level
"""

from collections import deque


class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, node, neighbor):
        """Add an edge to the graph"""
        if node not in self.graph:
            self.graph[node] = []
        self.graph[node].append(neighbor)
    
    def bfs(self, start, goal=None):
        """
        Perform BFS traversal from start node
        Returns path to goal if goal is specified, otherwise returns traversal order
        """
        visited = set()
        queue = deque([(start, [start])])
        traversal_order = []
        
        while queue:
            node, path = queue.popleft()
            
            if node not in visited:
                visited.add(node)
                traversal_order.append(node)
                
                # If goal is found, return the path
                if goal and node == goal:
                    return path, traversal_order
                
                # Add neighbors to queue
                for neighbor in self.graph.get(node, []):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return None if goal else traversal_order


def demonstrate_bfs():
    """Demonstrate BFS with a sample graph"""
    print("=== Breadth-First Search (BFS) Demonstration ===")
    
    # Create a sample graph
    g = Graph()
    edges = [
        ('A', 'B'), ('A', 'C'),
        ('B', 'D'), ('B', 'E'),
        ('C', 'F'), ('C', 'G'),
        ('D', 'H'), ('E', 'I')
    ]
    
    for node, neighbor in edges:
        g.add_edge(node, neighbor)
    
    print("\nGraph structure:")
    for node, neighbors in g.graph.items():
        print(f"{node} -> {neighbors}")
    
    # Perform BFS traversal
    start_node = 'A'
    traversal = g.bfs(start_node)
    print(f"\nBFS Traversal from {start_node}: {traversal}")
    
    # Find path using BFS
    goal_node = 'H'
    path, traversal_order = g.bfs(start_node, goal_node)
    print(f"\nPath from {start_node} to {goal_node}: {path}")
    print(f"Nodes visited during search: {traversal_order}")


if __name__ == "__main__":
    demonstrate_bfs()