"""
Perceptron Implementation
Author: AI Lab
Description: Simple perceptron for binary classification
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.errors = []
    
    def activation_function(self, x):
        """Step function activation"""
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        """Train the perceptron"""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for epoch in range(self.max_iterations):
            errors = 0
            
            for i in range(n_samples):
                # Forward pass
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.activation_function(linear_output)
                
                # Update weights if prediction is wrong
                error = y[i] - prediction
                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
                    errors += 1
            
            self.errors.append(errors)
            
            # If no errors, we've converged
            if errors == 0:
                print(f"Converged after {epoch + 1} epochs")
                break
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


def generate_linearly_separable_data(n_samples=100, random_state=42):
    """Generate linearly separable data for binary classification"""
    np.random.seed(random_state)
    
    # Generate two classes
    class_0 = np.random.normal([1, 1], 0.5, (n_samples//2, 2))
    class_1 = np.random.normal([3, 3], 0.5, (n_samples//2, 2))
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    return X, y


def demonstrate_perceptron():
    """Demonstrate perceptron with sample data"""
    print("=== Perceptron Demonstration ===")
    
    # Generate sample data
    X, y = generate_linearly_separable_data(100)
    print(f"Generated {len(X)} data points for binary classification")
    
    # Create and train perceptron
    perceptron = Perceptron(learning_rate=0.1, max_iterations=1000)
    perceptron.fit(X, y)
    
    # Make predictions
    predictions = perceptron.predict(X)
    accuracy = perceptron.score(X, y)
    
    print(f"\nTraining completed:")
    print(f"Final weights: {perceptron.weights}")
    print(f"Final bias: {perceptron.bias:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Plot results
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot data and decision boundary
        plt.subplot(1, 2, 1)
        
        # Plot data points
        class_0_idx = y == 0
        class_1_idx = y == 1
        plt.scatter(X[class_0_idx, 0], X[class_0_idx, 1], c='red', marker='o', label='Class 0', alpha=0.7)
        plt.scatter(X[class_1_idx, 0], X[class_1_idx, 1], c='blue', marker='s', label='Class 1', alpha=0.7)
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        if perceptron.weights[1] != 0:
            x_boundary = np.array([x_min, x_max])
            y_boundary = -(perceptron.weights[0] * x_boundary + perceptron.bias) / perceptron.weights[1]
            plt.plot(x_boundary, y_boundary, 'k-', linewidth=2, label='Decision Boundary')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Perceptron Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot training errors
        plt.subplot(1, 2, 2)
        plt.plot(perceptron.errors)
        plt.xlabel('Epoch')
        plt.ylabel('Number of Errors')
        plt.title('Training Progress')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/perceptron_demo.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("\nPlot saved as '/tmp/perceptron_demo.png'")
        
    except ImportError:
        print("\nMatplotlib not available. Skipping visualization.")
    
    # Test with new data points
    test_points = np.array([[0.5, 0.5], [2, 2], [3.5, 3.5], [1.5, 2.5]])
    test_predictions = perceptron.predict(test_points)
    
    print(f"\nPredictions for test points:")
    for i, (point, pred) in enumerate(zip(test_points, test_predictions)):
        print(f"Point {point} -> Class {pred}")


def demonstrate_xor_limitation():
    """Demonstrate that perceptron cannot solve XOR problem"""
    print("\n=== XOR Problem (Perceptron Limitation) ===")
    
    # XOR data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])  # XOR truth table
    
    print("XOR problem:")
    print("Input -> Output")
    for i in range(len(X_xor)):
        print(f"{X_xor[i]} -> {y_xor[i]}")
    
    # Try to train perceptron on XOR
    perceptron_xor = Perceptron(learning_rate=0.1, max_iterations=100)
    perceptron_xor.fit(X_xor, y_xor)
    
    predictions_xor = perceptron_xor.predict(X_xor)
    accuracy_xor = perceptron_xor.score(X_xor, y_xor)
    
    print(f"\nPerceptron predictions: {predictions_xor}")
    print(f"Expected outputs:       {y_xor}")
    print(f"Accuracy: {accuracy_xor:.4f}")
    print("Note: Perceptron cannot solve XOR (not linearly separable)")


if __name__ == "__main__":
    demonstrate_perceptron()
    demonstrate_xor_limitation()