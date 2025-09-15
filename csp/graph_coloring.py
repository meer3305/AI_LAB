"""
Graph Coloring Problem using Constraint Satisfaction
Author: AI Lab  
Description: Solve graph coloring using backtracking with constraint propagation
"""

from collections import defaultdict
import random


class GraphColoring:
    def __init__(self, vertices, edges, num_colors):
        self.vertices = vertices
        self.edges = edges
        self.num_colors = num_colors
        self.adjacency_list = self.build_adjacency_list()
        self.coloring = {}
        self.backtrack_calls = 0
    
    def build_adjacency_list(self):
        """Build adjacency list representation of the graph"""
        adj_list = defaultdict(list)
        for u, v in self.edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        return adj_list
    
    def is_safe(self, vertex, color):
        """Check if assigning color to vertex is safe"""
        for neighbor in self.adjacency_list[vertex]:
            if neighbor in self.coloring and self.coloring[neighbor] == color:
                return False
        return True
    
    def get_uncolored_vertex(self):
        """Get next uncolored vertex (using MRV heuristic)"""
        uncolored = [v for v in self.vertices if v not in self.coloring]
        
        if not uncolored:
            return None
        
        # Most Constraining Variable (MRV) heuristic:
        # Choose vertex with fewest legal values remaining
        min_legal_colors = float('inf')
        best_vertex = None
        
        for vertex in uncolored:
            legal_colors = 0
            for color in range(self.num_colors):
                if self.is_safe(vertex, color):
                    legal_colors += 1
            
            if legal_colors < min_legal_colors:
                min_legal_colors = legal_colors
                best_vertex = vertex
        
        return best_vertex
    
    def get_ordered_colors(self, vertex):
        """Get colors ordered by least constraining value heuristic"""
        colors = list(range(self.num_colors))
        
        # Least Constraining Value (LCV) heuristic:
        # Order colors by how many choices they eliminate for neighbors
        color_constraints = []
        
        for color in colors:
            if not self.is_safe(vertex, color):
                continue
            
            # Count how many neighbor choices this color eliminates
            eliminated_choices = 0
            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in self.coloring:  # Only uncolored neighbors
                    neighbor_legal_colors = 0
                    for neighbor_color in range(self.num_colors):
                        if (neighbor_color != color and 
                            self.is_safe_assuming_assignment(neighbor, neighbor_color, vertex, color)):
                            neighbor_legal_colors += 1
                    
                    # More legal colors remaining = fewer constraints
                    eliminated_choices += max(0, self.num_colors - 1 - neighbor_legal_colors)
            
            color_constraints.append((eliminated_choices, color))
        
        # Sort by fewest constraints (least constraining first)
        color_constraints.sort()
        return [color for _, color in color_constraints]
    
    def is_safe_assuming_assignment(self, vertex, color, assumed_vertex, assumed_color):
        """Check if assignment is safe assuming another assignment"""
        for neighbor in self.adjacency_list[vertex]:
            if neighbor in self.coloring and self.coloring[neighbor] == color:
                return False
            if neighbor == assumed_vertex and assumed_color == color:
                return False
        return True
    
    def solve_backtracking(self):
        """Solve graph coloring using backtracking with heuristics"""
        self.backtrack_calls += 1
        
        # Choose next vertex to color
        vertex = self.get_uncolored_vertex()
        
        if vertex is None:
            # All vertices colored successfully
            return True
        
        # Try colors in order determined by LCV heuristic
        for color in self.get_ordered_colors(vertex):
            if self.is_safe(vertex, color):
                # Assign color
                self.coloring[vertex] = color
                
                # Recursive call
                if self.solve_backtracking():
                    return True
                
                # Backtrack
                del self.coloring[vertex]
        
        return False
    
    def solve(self):
        """Solve the graph coloring problem"""
        self.coloring = {}
        self.backtrack_calls = 0
        
        success = self.solve_backtracking()
        return success, self.coloring, self.backtrack_calls
    
    def is_valid_coloring(self, coloring):
        """Verify if a coloring is valid"""
        for u, v in self.edges:
            if u in coloring and v in coloring and coloring[u] == coloring[v]:
                return False
        return True
    
    def print_coloring(self, coloring):
        """Print the coloring solution"""
        color_names = ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Orange', 'Pink', 'Brown']
        
        print("Graph coloring:")
        for vertex in sorted(self.vertices):
            if vertex in coloring:
                color_name = color_names[coloring[vertex]] if coloring[vertex] < len(color_names) else f"Color_{coloring[vertex]}"
                print(f"  Vertex {vertex}: {color_name}")


def create_sample_graphs():
    """Create various sample graphs for testing"""
    graphs = {
        'triangle': {
            'vertices': [1, 2, 3],
            'edges': [(1, 2), (2, 3), (3, 1)],
            'description': 'Triangle (3-clique)',
            'chromatic_number': 3
        },
        'square': {
            'vertices': [1, 2, 3, 4],
            'edges': [(1, 2), (2, 3), (3, 4), (4, 1)],
            'description': 'Square (4-cycle)',
            'chromatic_number': 2
        },
        'petersen': {
            'vertices': list(range(1, 11)),
            'edges': [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1),  # Outer cycle
                     (6, 7), (7, 8), (8, 9), (9, 10), (10, 6),  # Inner cycle
                     (1, 6), (2, 7), (3, 8), (4, 9), (5, 10)],  # Connections
            'description': 'Petersen Graph',
            'chromatic_number': 3
        },
        'wheel': {
            'vertices': list(range(1, 8)),
            'edges': [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)] + # Outer cycle
                    [(7, i) for i in range(1, 7)],  # Center to all vertices
            'description': 'Wheel Graph (6 spokes)',
            'chromatic_number': 3
        }
    }
    return graphs


def demonstrate_graph_coloring():
    """Demonstrate graph coloring on various graphs"""
    print("=== Graph Coloring Demonstration ===")
    
    graphs = create_sample_graphs()
    
    for name, graph_data in graphs.items():
        print(f"\n--- {graph_data['description']} ---")
        print(f"Vertices: {graph_data['vertices']}")
        print(f"Edges: {graph_data['edges']}")
        print(f"Theoretical chromatic number: {graph_data['chromatic_number']}")
        
        # Try with minimum colors first
        for num_colors in range(graph_data['chromatic_number'], graph_data['chromatic_number'] + 2):
            print(f"\nTrying with {num_colors} colors:")
            
            gc = GraphColoring(graph_data['vertices'], graph_data['edges'], num_colors)
            success, coloring, backtrack_calls = gc.solve()
            
            if success:
                print(f"  ✓ Solution found with {backtrack_calls} backtrack calls")
                gc.print_coloring(coloring)
                
                # Verify solution
                is_valid = gc.is_valid_coloring(coloring)
                print(f"  Solution is valid: {is_valid}")
                break
            else:
                print(f"  ✗ No solution found with {num_colors} colors")


def analyze_heuristics_effectiveness():
    """Analyze the effectiveness of different heuristics"""
    print("\n=== Heuristics Effectiveness Analysis ===")
    
    class BasicGraphColoring(GraphColoring):
        """Graph coloring without heuristics (basic backtracking)"""
        
        def get_uncolored_vertex(self):
            """Get first uncolored vertex (no heuristic)"""
            for vertex in self.vertices:
                if vertex not in self.coloring:
                    return vertex
            return None
        
        def get_ordered_colors(self, vertex):
            """Get colors in natural order (no heuristic)"""
            return [c for c in range(self.num_colors) if self.is_safe(vertex, c)]
    
    # Test on Petersen graph
    graphs = create_sample_graphs()
    graph_data = graphs['petersen']
    
    print(f"Comparing heuristics on {graph_data['description']}:")
    
    # Basic backtracking
    print(f"\n1. Basic backtracking:")
    gc_basic = BasicGraphColoring(graph_data['vertices'], graph_data['edges'], 3)
    success_basic, coloring_basic, calls_basic = gc_basic.solve()
    print(f"   Success: {success_basic}, Backtrack calls: {calls_basic}")
    
    # With heuristics
    print(f"\n2. With heuristics (MRV + LCV):")
    gc_heuristic = GraphColoring(graph_data['vertices'], graph_data['edges'], 3)
    success_heuristic, coloring_heuristic, calls_heuristic = gc_heuristic.solve()
    print(f"   Success: {success_heuristic}, Backtrack calls: {calls_heuristic}")
    
    if success_basic and success_heuristic:
        improvement = (calls_basic - calls_heuristic) / calls_basic * 100
        print(f"\nImprovement with heuristics: {improvement:.1f}% fewer backtrack calls")


def demonstrate_map_coloring():
    """Demonstrate map coloring (classic 4-color problem)"""
    print("\n=== Map Coloring Demonstration ===")
    
    # Simplified map of some US states (not geographically accurate)
    states = {
        'vertices': ['CA', 'NV', 'OR', 'WA', 'ID', 'UT', 'AZ', 'MT', 'WY', 'CO'],
        'edges': [
            ('CA', 'NV'), ('CA', 'OR'), ('CA', 'AZ'),
            ('NV', 'UT'), ('NV', 'ID'), ('NV', 'OR'),
            ('OR', 'WA'), ('OR', 'ID'),
            ('WA', 'ID'),
            ('ID', 'MT'), ('ID', 'UT'), ('ID', 'WY'),
            ('UT', 'WY'), ('UT', 'CO'), ('UT', 'AZ'),
            ('AZ', 'CO'),
            ('MT', 'WY'),
            ('WY', 'CO')
        ],
        'description': 'Simplified US States Map'
    }
    
    print(f"Coloring {states['description']}")
    print(f"States: {states['vertices']}")
    print(f"Borders: {len(states['edges'])} connections")
    
    # Try with 4 colors (famous 4-color theorem)
    print(f"\nTrying with 4 colors:")
    gc = GraphColoring(states['vertices'], states['edges'], 4)
    success, coloring, backtrack_calls = gc.solve()
    
    if success:
        print(f"✓ Successfully colored with 4 colors in {backtrack_calls} backtrack calls")
        gc.print_coloring(coloring)
        
        # Verify no adjacent states have same color
        print(f"\nVerifying adjacent states have different colors:")
        conflicts = 0
        for state1, state2 in states['edges']:
            if coloring[state1] == coloring[state2]:
                print(f"  CONFLICT: {state1} and {state2} both have color {coloring[state1]}")
                conflicts += 1
        
        if conflicts == 0:
            print(f"  ✓ No conflicts found - valid coloring!")
    else:
        print(f"✗ Could not color with 4 colors")


def generate_random_graph(n_vertices, edge_probability=0.3, seed=42):
    """Generate a random graph for testing"""
    random.seed(seed)
    
    vertices = list(range(1, n_vertices + 1))
    edges = []
    
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if random.random() < edge_probability:
                edges.append((i + 1, j + 1))
    
    return vertices, edges


def test_scalability():
    """Test scalability on random graphs of increasing size"""
    print("\n=== Scalability Testing ===")
    
    sizes = [5, 8, 10, 12]
    
    for size in sizes:
        print(f"\nTesting random graph with {size} vertices:")
        
        vertices, edges = generate_random_graph(size, edge_probability=0.4)
        print(f"  Generated graph with {len(edges)} edges")
        
        # Try with increasing number of colors until solution found
        for num_colors in range(2, size + 1):
            gc = GraphColoring(vertices, edges, num_colors)
            success, coloring, backtrack_calls = gc.solve()
            
            if success:
                print(f"  ✓ Solved with {num_colors} colors in {backtrack_calls} calls")
                break
        else:
            print(f"  ✗ Could not solve with up to {size} colors")


if __name__ == "__main__":
    demonstrate_graph_coloring()
    analyze_heuristics_effectiveness()
    demonstrate_map_coloring()
    test_scalability()