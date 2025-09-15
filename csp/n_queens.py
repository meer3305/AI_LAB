"""
N-Queens Problem Solver using Constraint Satisfaction
Author: AI Lab
Description: Solve the N-Queens problem using backtracking with constraint propagation
"""

import random


class NQueens:
    def __init__(self, n=8):
        self.n = n
        self.board = [-1] * n  # board[i] represents the column of queen in row i
        self.solutions = []
        self.nodes_explored = 0
    
    def is_safe(self, row, col):
        """Check if placing a queen at (row, col) is safe"""
        for i in range(row):
            # Check if queens are in the same column or diagonal
            if (self.board[i] == col or 
                abs(self.board[i] - col) == abs(i - row)):
                return False
        return True
    
    def solve_backtracking(self, row=0):
        """Solve N-Queens using backtracking"""
        self.nodes_explored += 1
        
        # Base case: all queens are placed
        if row == self.n:
            self.solutions.append(self.board[:])
            return True
        
        # Try placing queen in each column of current row
        for col in range(self.n):
            if self.is_safe(row, col):
                # Place queen
                self.board[row] = col
                
                # Recursive call for next row
                if self.solve_backtracking(row + 1):
                    return True
                
                # Backtrack
                self.board[row] = -1
        
        return False
    
    def solve_all_solutions(self, row=0):
        """Find all solutions to N-Queens problem"""
        self.nodes_explored += 1
        
        # Base case: all queens are placed
        if row == self.n:
            self.solutions.append(self.board[:])
            return
        
        # Try placing queen in each column of current row
        for col in range(self.n):
            if self.is_safe(row, col):
                # Place queen
                self.board[row] = col
                
                # Recursive call for next row
                self.solve_all_solutions(row + 1)
                
                # Backtrack
                self.board[row] = -1
    
    def print_board(self, solution=None):
        """Print the board with queens placed"""
        if solution is None:
            solution = self.board
        
        print("\n" + "+" + "-" * (4 * self.n - 1) + "+")
        for i in range(self.n):
            print("|", end="")
            for j in range(self.n):
                if solution[i] == j:
                    print(" Q ", end="|")
                else:
                    print("   ", end="|")
            print()
            print("+" + "-" * (4 * self.n - 1) + "+")
    
    def print_solution_compact(self, solution):
        """Print solution in compact format"""
        print(" ".join([str(col) for col in solution]))
    
    def is_valid_solution(self, solution):
        """Verify if a solution is valid"""
        n = len(solution)
        for i in range(n):
            for j in range(i + 1, n):
                # Check if queens attack each other
                if (solution[i] == solution[j] or 
                    abs(solution[i] - solution[j]) == abs(i - j)):
                    return False
        return True


class NQueensOptimized:
    """Optimized N-Queens solver using constraint propagation"""
    
    def __init__(self, n=8):
        self.n = n
        self.board = [-1] * n
        self.col_used = [False] * n
        self.diag1_used = [False] * (2 * n - 1)  # main diagonals
        self.diag2_used = [False] * (2 * n - 1)  # anti-diagonals
        self.solutions = []
        self.nodes_explored = 0
    
    def solve_optimized(self, row=0):
        """Solve N-Queens with constraint propagation optimization"""
        self.nodes_explored += 1
        
        if row == self.n:
            self.solutions.append(self.board[:])
            return True
        
        for col in range(self.n):
            diag1 = row - col + self.n - 1
            diag2 = row + col
            
            # Check constraints using precomputed arrays
            if (not self.col_used[col] and 
                not self.diag1_used[diag1] and 
                not self.diag2_used[diag2]):
                
                # Place queen and update constraints
                self.board[row] = col
                self.col_used[col] = True
                self.diag1_used[diag1] = True
                self.diag2_used[diag2] = True
                
                # Recursive call
                if self.solve_optimized(row + 1):
                    return True
                
                # Backtrack and remove constraints
                self.board[row] = -1
                self.col_used[col] = False
                self.diag1_used[diag1] = False
                self.diag2_used[diag2] = False
        
        return False


def demonstrate_nqueens():
    """Demonstrate N-Queens problem solving"""
    print("=== N-Queens Problem Demonstration ===")
    
    # Solve 4-Queens problem
    print("\n--- 4-Queens Problem ---")
    nq4 = NQueens(4)
    nq4.solve_all_solutions()
    
    print(f"Number of solutions for 4-Queens: {len(nq4.solutions)}")
    print(f"Nodes explored: {nq4.nodes_explored}")
    
    if nq4.solutions:
        print("\nAll solutions for 4-Queens:")
        for i, solution in enumerate(nq4.solutions):
            print(f"Solution {i+1}: {solution}")
            nq4.print_board(solution)
    
    # Solve 8-Queens problem (find first solution)
    print("\n--- 8-Queens Problem (First Solution) ---")
    nq8 = NQueens(8)
    success = nq8.solve_backtracking()
    
    if success:
        print("Solution found for 8-Queens:")
        print(f"Queen positions (row -> column): {nq8.board}")
        print(f"Nodes explored: {nq8.nodes_explored}")
        nq8.print_board()
        
        # Verify solution
        is_valid = nq8.is_valid_solution(nq8.board)
        print(f"Solution is valid: {is_valid}")
    else:
        print("No solution found!")
    
    # Compare performance with optimized version
    print("\n--- Performance Comparison ---")
    
    sizes = [8, 10, 12]
    for n in sizes:
        print(f"\nSolving {n}-Queens problem:")
        
        # Standard backtracking
        nq_standard = NQueens(n)
        nq_standard.solve_backtracking()
        
        # Optimized with constraint propagation
        nq_optimized = NQueensOptimized(n)
        nq_optimized.solve_optimized()
        
        print(f"Standard algorithm - Nodes explored: {nq_standard.nodes_explored}")
        print(f"Optimized algorithm - Nodes explored: {nq_optimized.nodes_explored}")
        
        if nq_optimized.nodes_explored > 0:
            speedup = nq_standard.nodes_explored / nq_optimized.nodes_explored
            print(f"Speedup: {speedup:.2f}x")


def analyze_nqueens_complexity():
    """Analyze the complexity of N-Queens for different board sizes"""
    print("\n=== N-Queens Complexity Analysis ===")
    
    print("Analyzing solution counts and search complexity:")
    print("N\tSolutions\tNodes (Standard)\tNodes (Optimized)")
    print("-" * 60)
    
    for n in range(4, 9):
        # Count all solutions
        nq_all = NQueens(n)
        nq_all.solve_all_solutions()
        
        # Find first solution with both algorithms
        nq_standard = NQueens(n)
        nq_standard.solve_backtracking()
        
        nq_optimized = NQueensOptimized(n)
        nq_optimized.solve_optimized()
        
        print(f"{n}\t{len(nq_all.solutions)}\t\t{nq_standard.nodes_explored}\t\t\t{nq_optimized.nodes_explored}")


def queens_puzzle_variations():
    """Demonstrate variations of the Queens puzzle"""
    print("\n=== Queens Puzzle Variations ===")
    
    # Generate a random valid configuration
    print("\n--- Random Valid 8-Queens Configuration ---")
    nq = NQueens(8)
    nq.solve_all_solutions()
    
    if nq.solutions:
        # Pick a random solution
        random_solution = random.choice(nq.solutions)
        print(f"Random solution: {random_solution}")
        nq.print_board(random_solution)
    
    # Demonstrate constraint checking
    print("\n--- Constraint Violation Examples ---")
    invalid_configs = [
        [0, 1, 2, 3, 4, 5, 6, 7],  # All queens in same diagonal
        [0, 0, 0, 0, 0, 0, 0, 0],  # All queens in same column
        [1, 3, 0, 2, 7, 5, 6, 4],  # Valid configuration (for comparison)
    ]
    
    for i, config in enumerate(invalid_configs):
        is_valid = nq.is_valid_solution(config)
        print(f"Configuration {i+1}: {config}")
        print(f"Valid: {is_valid}")
        if not is_valid and i < len(invalid_configs) - 1:
            print("Reason: Queens attack each other")
        print()


if __name__ == "__main__":
    demonstrate_nqueens()
    analyze_nqueens_complexity()
    queens_puzzle_variations()