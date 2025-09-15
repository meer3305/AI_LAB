"""
Alpha-Beta Pruning Algorithm Implementation
Author: AI Lab
Description: Alpha-beta pruning optimization for minimax algorithm
"""

import math
from game_theory.minimax import TicTacToe


class AlphaBetaAI:
    def __init__(self, player='O', opponent='X'):
        self.player = player
        self.opponent = opponent
        self.nodes_evaluated = 0
        self.pruned_branches = 0
    
    def evaluate(self, game):
        """Evaluate the current board state"""
        winner = game.check_winner()
        
        if winner == self.player:
            return 10
        elif winner == self.opponent:
            return -10
        else:
            return 0
    
    def alpha_beta(self, game, depth, alpha, beta, is_maximizing):
        """
        Alpha-beta pruning implementation
        alpha: best value that maximizer can guarantee
        beta: best value that minimizer can guarantee
        """
        self.nodes_evaluated += 1
        
        # Base case: terminal state
        if game.is_game_over():
            score = self.evaluate(game)
            return score
        
        if is_maximizing:
            # AI's turn - maximize score
            max_eval = -math.inf
            
            for row, col in game.get_empty_cells():
                # Make move
                game.make_move(row, col, self.player)
                
                # Recursive call
                eval_score = self.alpha_beta(game, depth + 1, alpha, beta, False)
                
                # Undo move
                game.undo_move(row, col)
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    self.pruned_branches += 1
                    break  # Beta cutoff
            
            return max_eval
        
        else:
            # Opponent's turn - minimize score
            min_eval = math.inf
            
            for row, col in game.get_empty_cells():
                # Make move
                game.make_move(row, col, self.opponent)
                
                # Recursive call
                eval_score = self.alpha_beta(game, depth + 1, alpha, beta, True)
                
                # Undo move
                game.undo_move(row, col)
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    self.pruned_branches += 1
                    break  # Alpha cutoff
            
            return min_eval
    
    def get_best_move(self, game):
        """Get the best move using alpha-beta pruning"""
        best_move = None
        best_value = -math.inf
        
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        
        print("AI is thinking with alpha-beta pruning...")
        
        for row, col in game.get_empty_cells():
            # Make move
            game.make_move(row, col, self.player)
            
            # Get alpha-beta value
            move_value = self.alpha_beta(game, 0, -math.inf, math.inf, False)
            
            # Undo move
            game.undo_move(row, col)
            
            print(f"Move ({row}, {col}): value = {move_value}")
            
            if move_value > best_value:
                best_value = move_value
                best_move = (row, col)
        
        print(f"Best move: {best_move} with value {best_value}")
        print(f"Nodes evaluated: {self.nodes_evaluated}")
        print(f"Branches pruned: {self.pruned_branches}")
        
        return best_move


def compare_minimax_vs_alphabeta():
    """Compare minimax and alpha-beta pruning performance"""
    print("=== Minimax vs Alpha-Beta Pruning Comparison ===")
    
    # Import minimax AI for comparison
    from game_theory.minimax import MinimaxAI
    
    # Create a specific game state for analysis
    game = TicTacToe()
    game.board = [
        ['X', 'O', ' '],
        [' ', 'X', ' '],
        [' ', ' ', 'O']
    ]
    
    print("Analyzing this position:")
    game.print_board()
    
    print(f"\n--- Standard Minimax ---")
    minimax_ai = MinimaxAI('O', 'X')
    minimax_move = minimax_ai.get_best_move(game)
    
    print(f"\n--- Alpha-Beta Pruning ---")
    alphabeta_ai = AlphaBetaAI('O', 'X')
    alphabeta_move = alphabeta_ai.get_best_move(game)
    
    print(f"\n--- Comparison Results ---")
    print(f"Minimax nodes evaluated: {len(game.get_empty_cells()) * 10}")  # Approximate
    print(f"Alpha-beta nodes evaluated: {alphabeta_ai.nodes_evaluated}")
    print(f"Branches pruned by alpha-beta: {alphabeta_ai.pruned_branches}")
    
    if alphabeta_ai.nodes_evaluated > 0:
        efficiency = (1 - alphabeta_ai.nodes_evaluated / (len(game.get_empty_cells()) * 10)) * 100
        print(f"Efficiency improvement: ~{efficiency:.1f}%")
    
    print(f"Same optimal move found: {minimax_move == alphabeta_move}")


def demonstrate_alphabeta():
    """Demonstrate alpha-beta pruning with Tic-Tac-Toe"""
    print("=== Alpha-Beta Pruning Demonstration ===")
    print("Tic-Tac-Toe Game: Human (X) vs AI with Alpha-Beta (O)")
    
    game = TicTacToe()
    ai = AlphaBetaAI('O', 'X')
    
    print("\nGame Rules:")
    print("- You are X, AI is O")
    print("- AI uses alpha-beta pruning for optimal play")
    
    game.print_board()
    
    # Predefined moves for demonstration
    demo_moves = [
        (1, 1),  # Human plays center
        None,    # AI's turn
        (0, 0),  # Human plays top-left
        None,    # AI's turn
    ]
    
    move_count = 0
    
    while not game.is_game_over() and move_count < len(demo_moves):
        if game.current_player == 'X':
            # Human player (demonstration)
            if demo_moves[move_count] is not None:
                row, col = demo_moves[move_count]
                print(f"\nHuman plays: ({row}, {col})")
                
                if game.make_move(row, col, 'X'):
                    game.current_player = 'O'
                else:
                    print("Invalid move!")
                    continue
            else:
                move_count += 1
                continue
        else:
            # AI player with alpha-beta pruning
            print(f"\n=== AI's Turn (Alpha-Beta Pruning) ===")
            best_move = ai.get_best_move(game)
            
            if best_move:
                row, col = best_move
                game.make_move(row, col, 'O')
                print(f"AI plays: ({row}, {col})")
                game.current_player = 'X'
        
        game.print_board()
        move_count += 1
    
    # Check final result
    winner = game.check_winner()
    if winner:
        print(f"\n🎉 Winner: {winner}")
    elif game.is_board_full():
        print(f"\n🤝 It's a tie!")


def analyze_pruning_effectiveness():
    """Analyze the effectiveness of alpha-beta pruning in different scenarios"""
    print("\n=== Alpha-Beta Pruning Effectiveness Analysis ===")
    
    scenarios = [
        {
            'name': 'Early Game (many moves)',
            'board': [
                [' ', ' ', ' '],
                [' ', 'X', ' '],
                [' ', ' ', ' ']
            ]
        },
        {
            'name': 'Mid Game (few moves)',
            'board': [
                ['X', 'O', ' '],
                ['O', 'X', ' '],
                [' ', ' ', ' ']
            ]
        },
        {
            'name': 'End Game (very few moves)',
            'board': [
                ['X', 'O', 'X'],
                ['O', 'X', ' '],
                [' ', 'O', ' ']
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        game = TicTacToe()
        game.board = scenario['board']
        
        ai = AlphaBetaAI('O', 'X')
        empty_cells = game.get_empty_cells()
        
        print(f"Empty cells: {len(empty_cells)}")
        
        if empty_cells:
            best_move = ai.get_best_move(game)
            print(f"Nodes evaluated: {ai.nodes_evaluated}")
            print(f"Branches pruned: {ai.pruned_branches}")
            
            if ai.nodes_evaluated > 0:
                pruning_ratio = ai.pruned_branches / ai.nodes_evaluated * 100
                print(f"Pruning ratio: {pruning_ratio:.1f}%")


def educational_pruning_trace():
    """Educational trace of alpha-beta pruning process"""
    print("\n=== Educational Alpha-Beta Pruning Trace ===")
    
    class TracingAlphaBeta(AlphaBetaAI):
        def __init__(self, player='O', opponent='X'):
            super().__init__(player, opponent)
            self.trace_depth = 0
        
        def alpha_beta(self, game, depth, alpha, beta, is_maximizing):
            indent = "  " * depth
            self.trace_depth = depth
            
            if depth <= 2:  # Only trace first few levels
                player_name = "AI" if is_maximizing else "Human"
                print(f"{indent}Depth {depth} ({player_name}): α={alpha}, β={beta}")
            
            if game.is_game_over():
                score = self.evaluate(game)
                if depth <= 2:
                    print(f"{indent}Terminal: score={score}")
                return score
            
            if is_maximizing:
                max_eval = -math.inf
                
                for row, col in game.get_empty_cells():
                    if depth <= 1:
                        print(f"{indent}Trying move ({row}, {col})")
                    
                    game.make_move(row, col, self.player)
                    eval_score = self.alpha_beta(game, depth + 1, alpha, beta, False)
                    game.undo_move(row, col)
                    
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    
                    if depth <= 1:
                        print(f"{indent}Move ({row}, {col}): score={eval_score}, α={alpha}")
                    
                    if beta <= alpha:
                        if depth <= 1:
                            print(f"{indent}PRUNING: β({beta}) <= α({alpha})")
                        break
                
                return max_eval
            else:
                min_eval = math.inf
                
                for row, col in game.get_empty_cells():
                    game.make_move(row, col, self.opponent)
                    eval_score = self.alpha_beta(game, depth + 1, alpha, beta, True)
                    game.undo_move(row, col)
                    
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    
                    if beta <= alpha:
                        break
                
                return min_eval
    
    # Simple game state for tracing
    game = TicTacToe()
    game.board = [
        [' ', ' ', ' '],
        [' ', 'X', ' '],
        [' ', ' ', ' ']
    ]
    
    print("Tracing alpha-beta pruning for this position:")
    game.print_board()
    
    ai = TracingAlphaBeta('O', 'X')
    print(f"\nAlpha-Beta trace (first 2 levels):")
    best_move = ai.get_best_move(game)


if __name__ == "__main__":
    demonstrate_alphabeta()
    compare_minimax_vs_alphabeta()
    analyze_pruning_effectiveness()
    educational_pruning_trace()