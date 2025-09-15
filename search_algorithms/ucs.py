"""
Uniform Cost Search (UCS) Implementation
Author: AI Lab
Description: UCS finds the least-cost path using a priority queue
"""

import heapq


class WeightedGraph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, node, neighbor, weight):
        """Add a weighted edge to the graph"""
        if node not in self.graph:
            self.graph[node] = []
        self.graph[node].append((neighbor, weight))
    
    def ucs(self, start, goal):
        """
        Perform Uniform Cost Search from start to goal
        Returns (path, total_cost) if goal is reached, otherwise (None, None)
        """
        # Priority queue: (cost, node, path)
        priority_queue = [(0, start, [start])]
        visited = set()
        
        while priority_queue:
            current_cost, current_node, path = heapq.heappop(priority_queue)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Check if we reached the goal
            if current_node == goal:
                return path, current_cost
            
            # Explore neighbors
            for neighbor, edge_cost in self.graph.get(current_node, []):
                if neighbor not in visited:
                    new_cost = current_cost + edge_cost
                    new_path = path + [neighbor]
                    heapq.heappush(priority_queue, (new_cost, neighbor, new_path))
        
        return None, None  # Goal not reachable


def demonstrate_ucs():
    """Demonstrate UCS with a weighted graph"""
    print("=== Uniform Cost Search (UCS) Demonstration ===")
    
    # Create a weighted graph
    g = WeightedGraph()
    
    # Add edges with weights (representing distances or costs)
    edges = [
        ('A', 'B', 4), ('A', 'C', 2),
        ('B', 'C', 1), ('B', 'D', 5),
        ('C', 'D', 8), ('C', 'E', 10),
        ('D', 'E', 2), ('D', 'F', 6),
        ('E', 'F', 3)
    ]
    
    for node, neighbor, weight in edges:
        g.add_edge(node, neighbor, weight)
    
    print("\nWeighted graph edges:")
    for node, neighbors in g.graph.items():
        for neighbor, weight in neighbors:
            print(f"{node} -> {neighbor} (cost: {weight})")
    
    # Find path using UCS
    start_node = 'A'
    goal_node = 'F'
    
    path, cost = g.ucs(start_node, goal_node)
    
    if path:
        print(f"\nOptimal path from {start_node} to {goal_node}: {' -> '.join(path)}")
        print(f"Total cost: {cost}")
    else:
        print(f"\nNo path found from {start_node} to {goal_node}")


if __name__ == "__main__":
    demonstrate_ucs()