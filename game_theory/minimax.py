"""
Minimax Algorithm Implementation
Author: AI Lab  
Description: Minimax algorithm for game tree search with Tic-Tac-Toe example
"""

import math


class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
    
    def print_board(self):
        """Print the current board state"""
        print("\n   0   1   2")
        for i in range(3):
            print(f"{i}  {self.board[i][0]} | {self.board[i][1]} | {self.board[i][2]}")
            if i < 2:
                print("  ---|---|---")
    
    def make_move(self, row, col, player):
        """Make a move on the board"""
        if self.board[row][col] == ' ':
            self.board[row][col] = player
            return True
        return False
    
    def undo_move(self, row, col):
        """Undo a move on the board"""
        self.board[row][col] = ' '
    
    def check_winner(self):
        """Check if there's a winner"""
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        
        return None
    
    def is_board_full(self):
        """Check if the board is full"""
        for row in self.board:
            for cell in row:
                if cell == ' ':
                    return False
        return True
    
    def get_empty_cells(self):
        """Get list of empty cells"""
        empty_cells = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    empty_cells.append((i, j))
        return empty_cells
    
    def is_game_over(self):
        """Check if the game is over"""
        return self.check_winner() is not None or self.is_board_full()


class MinimaxAI:
    def __init__(self, player='O', opponent='X'):
        self.player = player
        self.opponent = opponent
    
    def evaluate(self, game):
        """Evaluate the current board state"""
        winner = game.check_winner()
        
        if winner == self.player:
            return 10
        elif winner == self.opponent:
            return -10
        else:
            return 0
    
    def minimax(self, game, depth, is_maximizing):
        """
        Minimax algorithm implementation
        is_maximizing: True if it's AI's turn (maximizing player)
        """
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
                eval_score = self.minimax(game, depth + 1, False)
                
                # Undo move
                game.undo_move(row, col)
                
                max_eval = max(max_eval, eval_score)
            
            return max_eval
        
        else:
            # Opponent's turn - minimize score
            min_eval = math.inf
            
            for row, col in game.get_empty_cells():
                # Make move
                game.make_move(row, col, self.opponent)
                
                # Recursive call
                eval_score = self.minimax(game, depth + 1, True)
                
                # Undo move
                game.undo_move(row, col)
                
                min_eval = min(min_eval, eval_score)
            
            return min_eval
    
    def get_best_move(self, game):
        """Get the best move using minimax algorithm"""
        best_move = None
        best_value = -math.inf
        
        print("AI is thinking...")
        
        for row, col in game.get_empty_cells():
            # Make move
            game.make_move(row, col, self.player)
            
            # Get minimax value
            move_value = self.minimax(game, 0, False)
            
            # Undo move
            game.undo_move(row, col)
            
            print(f"Move ({row}, {col}): value = {move_value}")
            
            if move_value > best_value:
                best_value = move_value
                best_move = (row, col)
        
        print(f"Best move: {best_move} with value {best_value}")
        return best_move


def demonstrate_minimax():
    """Demonstrate minimax algorithm with Tic-Tac-Toe"""
    print("=== Minimax Algorithm Demonstration ===")
    print("Tic-Tac-Toe Game: Human (X) vs AI (O)")
    
    game = TicTacToe()
    ai = MinimaxAI('O', 'X')
    
    print("\nGame Rules:")
    print("- You are X, AI is O")
    print("- Enter row and column (0-2) separated by space")
    print("- Example: '1 2' for row 1, column 2")
    
    game.print_board()
    
    # Predefined moves for demonstration
    demo_moves = [
        (1, 1),  # Human plays center
        None,    # AI's turn
        (0, 0),  # Human plays top-left
        None,    # AI's turn
        (2, 2),  # Human plays bottom-right
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
            # AI player
            print(f"\n=== AI's Turn ===")
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
    
    print("\nGame Analysis:")
    print("The AI uses minimax to evaluate all possible future moves")
    print("and chooses the move that leads to the best outcome.")


def analyze_minimax_tree():
    """Analyze the minimax decision tree for a specific position"""
    print("\n=== Minimax Tree Analysis ===")
    
    # Create a specific game state for analysis
    game = TicTacToe()
    
    # Set up a mid-game position
    game.board = [
        ['X', 'O', ' '],
        [' ', 'X', ' '],
        [' ', ' ', 'O']
    ]
    
    print("Analyzing this position:")
    game.print_board()
    
    ai = MinimaxAI('O', 'X')
    
    print(f"\nEvaluating all possible moves for AI (O):")
    print("Move\t\tMinimax Value")
    print("-" * 30)
    
    for row, col in game.get_empty_cells():
        # Make move
        game.make_move(row, col, 'O')
        
        # Get minimax value
        value = ai.minimax(game, 0, False)
        
        # Undo move
        game.undo_move(row, col)
        
        print(f"({row}, {col})\t\t{value}")
    
    best_move = ai.get_best_move(game)
    print(f"\nBest move for AI: {best_move}")


if __name__ == "__main__":
    demonstrate_minimax()
    analyze_minimax_tree()