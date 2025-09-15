"""
Multi-layer Perceptron (MLP) Implementation
Author: AI Lab
Description: Simple MLP with backpropagation for binary classification
"""

import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights randomly
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # For tracking training progress
        self.losses = []
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward propagation"""
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]  # Number of samples
        
        # Calculate output layer error
        output_error = output - y
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Update weights and biases
        self.W2 -= self.learning_rate * self.a1.T.dot(output_delta) / m
        self.b2 -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True) / m
        self.W1 -= self.learning_rate * X.T.dot(hidden_delta) / m
        self.b1 -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs=1000):
        """Train the MLP"""
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Calculate loss (Mean Squared Error)
            loss = np.mean((output - y) ** 2)
            self.losses.append(loss)
            
            # Backward propagation
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return (output > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.forward(X)


def generate_xor_data():
    """Generate XOR dataset"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR truth table
    return X, y


def generate_nonlinear_data(n_samples=200):
    """Generate non-linearly separable data"""
    np.random.seed(42)
    
    # Generate circular data
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radii = np.random.uniform(0, 1, n_samples)
    
    X = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    # Labels: 1 if inside circle of radius 0.6, 0 otherwise
    y = (radii < 0.6).astype(int).reshape(-1, 1)
    
    return X, y


def demonstrate_mlp_xor():
    """Demonstrate MLP solving XOR problem"""
    print("=== MLP XOR Problem Demonstration ===")
    
    # Generate XOR data
    X, y = generate_xor_data()
    
    print("XOR Problem:")
    print("Input -> Output")
    for i in range(len(X)):
        print(f"{X[i]} -> {y[i][0]}")
    
    # Create and train MLP
    mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=1.0)
    
    print(f"\nTraining MLP with architecture: 2-{mlp.hidden_size}-1")
    mlp.train(X, y, epochs=1000)
    
    # Make predictions
    predictions = mlp.predict(X)
    probabilities = mlp.predict_proba(X)
    
    print(f"\nResults:")
    print("Input\t\tExpected\tPredicted\tProbability")
    print("-" * 50)
    for i in range(len(X)):
        print(f"{X[i]}\t{y[i][0]}\t\t{predictions[i][0]}\t\t{probabilities[i][0]:.3f}")
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y)
    print(f"\nAccuracy: {accuracy:.3f}")
    print("Note: MLP can solve XOR (unlike single perceptron)")


def demonstrate_mlp_nonlinear():
    """Demonstrate MLP on non-linearly separable data"""
    print("\n=== MLP Non-linear Classification Demonstration ===")
    
    # Generate non-linear data
    X, y = generate_nonlinear_data(200)
    
    print(f"Generated {len(X)} samples for circular classification")
    print("Task: Classify points inside vs outside a circle")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train MLP
    mlp = MLP(input_size=2, hidden_size=8, output_size=1, learning_rate=0.5)
    
    print(f"\nTraining MLP with architecture: 2-{mlp.hidden_size}-1")
    mlp.train(X_train, y_train, epochs=500)
    
    # Make predictions
    train_predictions = mlp.predict(X_train)
    test_predictions = mlp.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"\nPerformance:")
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Visualize results
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(mlp.losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.grid(True, alpha=0.3)
        
        # Plot data and predictions
        plt.subplot(1, 2, 2)
        
        # Plot training data
        class_0_idx = y_train.flatten() == 0
        class_1_idx = y_train.flatten() == 1
        plt.scatter(X_train[class_0_idx, 0], X_train[class_0_idx, 1], 
                   c='red', alpha=0.6, label='Class 0 (Outside)')
        plt.scatter(X_train[class_1_idx, 0], X_train[class_1_idx, 1], 
                   c='blue', alpha=0.6, label='Class 1 (Inside)')
        
        # Create decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        mesh_points = np.column_stack([xx.ravel(), yy.ravel()])
        Z = mlp.predict_proba(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        
        plt.title('MLP Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/mlp_demo.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("\nPlot saved as '/tmp/mlp_demo.png'")
        
    except ImportError:
        print("\nMatplotlib not available. Skipping visualization.")


def compare_with_perceptron():
    """Compare MLP with single perceptron on XOR"""
    print("\n=== MLP vs Perceptron Comparison ===")
    
    X, y = generate_xor_data()
    
    # Try single perceptron
    from neural_networks.perceptron import Perceptron
    
    print("Single Perceptron on XOR:")
    perceptron = Perceptron(learning_rate=0.1, max_iterations=100)
    perceptron.fit(X, y.flatten())
    
    perceptron_predictions = perceptron.predict(X).reshape(-1, 1)
    perceptron_accuracy = np.mean(perceptron_predictions == y)
    print(f"Perceptron accuracy: {perceptron_accuracy:.3f}")
    
    # MLP on XOR
    print(f"\nMLP on XOR:")
    mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=1.0)
    mlp.train(X, y, epochs=500)
    
    mlp_predictions = mlp.predict(X)
    mlp_accuracy = np.mean(mlp_predictions == y)
    print(f"MLP accuracy: {mlp_accuracy:.3f}")
    
    print(f"\nConclusion:")
    print(f"- Single perceptron cannot solve XOR (linearly separable only)")
    print(f"- MLP with hidden layer can solve XOR (non-linearly separable)")


if __name__ == "__main__":
    demonstrate_mlp_xor()
    demonstrate_mlp_nonlinear()
    # compare_with_perceptron()  # Uncomment if perceptron.py is available