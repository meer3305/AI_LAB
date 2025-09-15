"""
Depth-First Search (DFS) Algorithm Implementation
Author: AI Lab
Description: DFS explores as far as possible along each branch before backtracking
"""


class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, node, neighbor):
        """Add an edge to the graph"""
        if node not in self.graph:
            self.graph[node] = []
        self.graph[node].append(neighbor)
    
    def dfs_recursive(self, node, visited=None, goal=None, path=None):
        """
        Perform DFS traversal recursively
        """
        if visited is None:
            visited = set()
        if path is None:
            path = []
        
        visited.add(node)
        path.append(node)
        
        # If goal is found, return the path
        if goal and node == goal:
            return path.copy()
        
        # Visit neighbors
        for neighbor in self.graph.get(node, []):
            if neighbor not in visited:
                result = self.dfs_recursive(neighbor, visited, goal, path)
                if result:  # Goal found
                    return result
        
        path.pop()  # Backtrack
        return None if goal else list(visited)
    
    def dfs_iterative(self, start, goal=None):
        """
        Perform DFS traversal iteratively using a stack
        """
        visited = set()
        stack = [(start, [start])]
        traversal_order = []
        
        while stack:
            node, path = stack.pop()
            
            if node not in visited:
                visited.add(node)
                traversal_order.append(node)
                
                # If goal is found, return the path
                if goal and node == goal:
                    return path, traversal_order
                
                # Add neighbors to stack (in reverse order for consistent traversal)
                for neighbor in reversed(self.graph.get(node, [])):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
        
        return None if goal else traversal_order


def demonstrate_dfs():
    """Demonstrate DFS with a sample graph"""
    print("=== Depth-First Search (DFS) Demonstration ===")
    
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
    
    # Perform DFS traversal (recursive)
    start_node = 'A'
    traversal = g.dfs_recursive(start_node)
    print(f"\nDFS Traversal (Recursive) from {start_node}: {traversal}")
    
    # Perform DFS traversal (iterative)
    traversal_iter = g.dfs_iterative(start_node)
    print(f"DFS Traversal (Iterative) from {start_node}: {traversal_iter}")
    
    # Find path using DFS
    goal_node = 'H'
    path = g.dfs_recursive(start_node, goal=goal_node)
    print(f"\nPath from {start_node} to {goal_node} (Recursive): {path}")
    
    path_iter, traversal_order = g.dfs_iterative(start_node, goal_node)
    print(f"Path from {start_node} to {goal_node} (Iterative): {path_iter}")
    print(f"Nodes visited during search: {traversal_order}")


if __name__ == "__main__":
    demonstrate_dfs()