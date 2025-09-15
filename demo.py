#!/usr/bin/env python3
"""
AI Lab Demo Script
Author: AI Lab
Description: Comprehensive demonstration of all AI lab programs
"""

import os
import sys
import importlib.util


def run_demo_module(module_path, description):
    """Run a demo module and handle any errors"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Module: {module_path}")
    print(f"{'='*80}")
    
    try:
        # Import and run the module
        spec = importlib.util.spec_from_file_location("demo_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"\n✓ {description} completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error in {description}: {str(e)}")
        return False
    
    return True


def main():
    """Run comprehensive AI lab demonstration"""
    print("🧠 AI LAB COMPREHENSIVE DEMONSTRATION 🧠")
    print("This script demonstrates all implemented AI algorithms and concepts.")
    print("Each module will run its demonstration functions.")
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of all demo modules to run
    demos = [
        # Search Algorithms
        {
            'path': os.path.join(base_dir, 'search_algorithms', 'bfs.py'),
            'description': 'Breadth-First Search (BFS) Algorithm'
        },
        {
            'path': os.path.join(base_dir, 'search_algorithms', 'dfs.py'),
            'description': 'Depth-First Search (DFS) Algorithm'
        },
        {
            'path': os.path.join(base_dir, 'search_algorithms', 'a_star.py'),
            'description': 'A* Search Algorithm'
        },
        {
            'path': os.path.join(base_dir, 'search_algorithms', 'ucs.py'),
            'description': 'Uniform Cost Search (UCS) Algorithm'
        },
        
        # Machine Learning
        {
            'path': os.path.join(base_dir, 'machine_learning', 'linear_regression.py'),
            'description': 'Linear Regression Implementation'
        },
        {
            'path': os.path.join(base_dir, 'machine_learning', 'logistic_regression.py'),
            'description': 'Logistic Regression Implementation'
        },
        {
            'path': os.path.join(base_dir, 'machine_learning', 'kmeans.py'),
            'description': 'K-Means Clustering Algorithm'
        },
        {
            'path': os.path.join(base_dir, 'machine_learning', 'decision_tree.py'),
            'description': 'Decision Tree Classifier'
        },
        
        # Neural Networks
        {
            'path': os.path.join(base_dir, 'neural_networks', 'perceptron.py'),
            'description': 'Perceptron Implementation'
        },
        {
            'path': os.path.join(base_dir, 'neural_networks', 'mlp.py'),
            'description': 'Multi-layer Perceptron (MLP)'
        },
        
        # Expert Systems
        {
            'path': os.path.join(base_dir, 'expert_systems', 'forward_chaining.py'),
            'description': 'Expert System with Forward Chaining'
        },
        
        # Game Theory
        {
            'path': os.path.join(base_dir, 'game_theory', 'minimax.py'),
            'description': 'Minimax Algorithm for Game Trees'
        },
        {
            'path': os.path.join(base_dir, 'game_theory', 'alpha_beta.py'),
            'description': 'Alpha-Beta Pruning Optimization'
        },
        
        # Constraint Satisfaction Problems
        {
            'path': os.path.join(base_dir, 'csp', 'n_queens.py'),
            'description': 'N-Queens Problem Solver'
        },
        {
            'path': os.path.join(base_dir, 'csp', 'graph_coloring.py'),
            'description': 'Graph Coloring Problem'
        },
        
        # Natural Language Processing
        {
            'path': os.path.join(base_dir, 'nlp', 'text_preprocessing.py'),
            'description': 'NLP Text Processing and Sentiment Analysis'
        }
    ]
    
    successful_demos = 0
    failed_demos = 0
    
    # Run each demo
    for demo in demos:
        if os.path.exists(demo['path']):
            success = run_demo_module(demo['path'], demo['description'])
            if success:
                successful_demos += 1
            else:
                failed_demos += 1
        else:
            print(f"\n⚠️  Module not found: {demo['path']}")
            failed_demos += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"DEMONSTRATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total modules: {len(demos)}")
    print(f"Successful: {successful_demos}")
    print(f"Failed: {failed_demos}")
    print(f"Success rate: {successful_demos/len(demos)*100:.1f}%")
    
    if failed_demos == 0:
        print("\n🎉 All AI lab demonstrations completed successfully!")
    else:
        print(f"\n⚠️  {failed_demos} demonstrations had issues. Check error messages above.")
    
    print(f"\n📚 For detailed information about each algorithm, check the README.md file.")
    print(f"📁 All source code is available in the respective directories.")


def run_quick_demo():
    """Run a quick demonstration of key algorithms"""
    print("🚀 QUICK AI LAB DEMONSTRATION 🚀")
    print("Running a subset of key algorithms for quick overview...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    quick_demos = [
        {
            'path': os.path.join(base_dir, 'search_algorithms', 'a_star.py'),
            'description': 'A* Pathfinding'
        },
        {
            'path': os.path.join(base_dir, 'machine_learning', 'linear_regression.py'),
            'description': 'Linear Regression'
        },
        {
            'path': os.path.join(base_dir, 'neural_networks', 'perceptron.py'),
            'description': 'Perceptron Neural Network'
        },
        {
            'path': os.path.join(base_dir, 'game_theory', 'minimax.py'),
            'description': 'Minimax Game AI'
        },
        {
            'path': os.path.join(base_dir, 'csp', 'n_queens.py'),
            'description': 'N-Queens Puzzle Solver'
        }
    ]
    
    for demo in quick_demos:
        if os.path.exists(demo['path']):
            run_demo_module(demo['path'], demo['description'])
    
    print(f"\n🎯 Quick demonstration complete! Run 'python demo.py full' for complete demos.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        run_quick_demo()
    else:
        main()