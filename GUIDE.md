# AI Lab Programs - Complete Implementation Guide

## Overview

This repository contains a comprehensive collection of Artificial Intelligence laboratory programs covering fundamental AI concepts, algorithms, and implementations. Each program is self-contained, well-documented, and includes demonstration functions.

## Quick Start

### Installation
```bash
git clone https://github.com/meer3305/AI_LAB.git
cd AI_LAB
pip install -r requirements.txt
```

### Run Demonstrations
```bash
# Quick demo of key algorithms
python demo.py quick

# Complete demonstration of all algorithms
python demo.py

# Run individual algorithms
python search_algorithms/a_star.py
python machine_learning/linear_regression.py
python neural_networks/perceptron.py
```

## Algorithm Categories

### 🔍 Search Algorithms
| Algorithm | File | Description |
|-----------|------|-------------|
| **Breadth-First Search** | `search_algorithms/bfs.py` | Level-by-level graph traversal |
| **Depth-First Search** | `search_algorithms/dfs.py` | Deep exploration with backtracking |
| **A* Search** | `search_algorithms/a_star.py` | Optimal pathfinding with heuristics |
| **Uniform Cost Search** | `search_algorithms/ucs.py` | Least-cost path finding |

### 🤖 Machine Learning
| Algorithm | File | Description |
|-----------|------|-------------|
| **Linear Regression** | `machine_learning/linear_regression.py` | Gradient descent implementation |
| **Logistic Regression** | `machine_learning/logistic_regression.py` | Binary classification with sigmoid |
| **K-Means Clustering** | `machine_learning/kmeans.py` | Unsupervised clustering algorithm |
| **Decision Trees** | `machine_learning/decision_tree.py` | Tree-based classification |

### 🧠 Neural Networks
| Algorithm | File | Description |
|-----------|------|-------------|
| **Perceptron** | `neural_networks/perceptron.py` | Single-layer neural network |
| **Multi-layer Perceptron** | `neural_networks/mlp.py` | Deep network with backpropagation |

### 🎯 Expert Systems
| Algorithm | File | Description |
|-----------|------|-------------|
| **Forward Chaining** | `expert_systems/forward_chaining.py` | Rule-based inference engine |

### 🎮 Game Theory
| Algorithm | File | Description |
|-----------|------|-------------|
| **Minimax** | `game_theory/minimax.py` | Game tree search algorithm |
| **Alpha-Beta Pruning** | `game_theory/alpha_beta.py` | Optimized minimax with pruning |

### 🧩 Constraint Satisfaction
| Algorithm | File | Description |
|-----------|------|-------------|
| **N-Queens** | `csp/n_queens.py` | Classic constraint satisfaction problem |
| **Graph Coloring** | `csp/graph_coloring.py` | Map coloring with backtracking |

### 📝 Natural Language Processing
| Algorithm | File | Description |
|-----------|------|-------------|
| **Text Processing** | `nlp/text_preprocessing.py` | Tokenization, sentiment analysis, n-grams |

## Features

### ✅ Complete Implementations
- All algorithms implemented from scratch
- No external AI libraries required (only numpy, matplotlib for visualization)
- Educational focus with clear, readable code

### ✅ Comprehensive Demonstrations
- Each algorithm includes multiple demonstration functions
- Real-world examples and use cases
- Performance analysis and comparisons

### ✅ Educational Content
- Detailed comments and docstrings
- Step-by-step algorithm explanations
- Complexity analysis and optimization techniques

### ✅ Visualization Support
- Matplotlib-based visualizations for applicable algorithms
- Decision boundaries, training curves, and solution paths
- Optional visualization (graceful fallback if matplotlib unavailable)

## Algorithm Highlights

### Search Algorithms
- **BFS**: Level-order traversal, shortest path in unweighted graphs
- **DFS**: Memory-efficient deep exploration, topological sorting
- **A***: Optimal pathfinding using Manhattan distance heuristic
- **UCS**: Guaranteed optimal paths in weighted graphs

### Machine Learning
- **Linear Regression**: Gradient descent with R² scoring
- **Logistic Regression**: Sigmoid activation, cross-entropy loss
- **K-Means**: Lloyd's algorithm with elbow method analysis
- **Decision Trees**: Information gain, entropy calculations

### Neural Networks
- **Perceptron**: Linear classification, convergence theorem
- **MLP**: XOR problem solver, backpropagation algorithm

### Expert Systems
- **Forward Chaining**: Animal classification, medical diagnosis
- Rule-based inference with conflict resolution

### Game Theory
- **Minimax**: Tic-tac-toe AI, perfect play guarantee
- **Alpha-Beta**: Pruning optimization, efficiency improvements

### Constraint Satisfaction
- **N-Queens**: Backtracking with constraint propagation
- **Graph Coloring**: MRV and LCV heuristics, map coloring

## Educational Value

This repository serves as:
- **Learning Resource**: Understand AI algorithms from first principles
- **Reference Implementation**: Clean, well-commented code examples
- **Practical Exercises**: Ready-to-run demonstrations and experiments
- **Foundation Knowledge**: Core concepts for advanced AI studies

## Testing and Validation

All algorithms include:
- Input validation and error handling
- Performance metrics and analysis
- Correctness verification
- Edge case handling

## Contributing

Feel free to:
- Add new algorithms
- Improve existing implementations
- Enhance documentation
- Add more visualization features
- Create additional test cases

## Dependencies

- **Python 3.7+**: Core language
- **NumPy**: Numerical computations
- **Matplotlib**: Visualizations (optional)
- **Pandas**: Data manipulation (optional)
- **Scikit-learn**: Comparison benchmarks (optional)

## License

This project is open source and available under the MIT License.

---

**Happy Learning! 🚀🧠**

*For questions or suggestions, please open an issue or submit a pull request.*