"""
Linear Regression Implementation
Author: AI Lab
Description: Simple linear regression using gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """Train the linear regression model"""
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Calculate cost (Mean Squared Error)
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """Make predictions using the trained model"""
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """Calculate R-squared score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def generate_sample_data(n_samples=100, noise=0.1):
    """Generate sample data for demonstration"""
    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 10
    y = 2 * X.flatten() + 3 + np.random.normal(0, noise, n_samples)
    return X, y


def demonstrate_linear_regression():
    """Demonstrate linear regression with sample data"""
    print("=== Linear Regression Demonstration ===")
    
    # Generate sample data
    X, y = generate_sample_data(100, noise=1.0)
    
    print(f"Generated {len(X)} data points")
    print(f"True relationship: y = 2x + 3 + noise")
    
    # Create and train model
    model = LinearRegression(learning_rate=0.01, max_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2_score = model.score(X, y)
    
    print(f"\nTrained model parameters:")
    print(f"Weight: {model.weights[0]:.4f}")
    print(f"Bias: {model.bias:.4f}")
    print(f"R-squared score: {r2_score:.4f}")
    
    # Plot results
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot data and predictions
        plt.subplot(1, 2, 1)
        plt.scatter(X, y, alpha=0.6, label='Data points')
        plt.plot(X, y_pred, 'r-', label=f'Prediction: y = {model.weights[0]:.2f}x + {model.bias:.2f}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Linear Regression Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot cost history
        plt.subplot(1, 2, 2)
        plt.plot(model.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.title('Training Cost History')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/linear_regression_demo.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("\nPlot saved as '/tmp/linear_regression_demo.png'")
        
    except ImportError:
        print("\nMatplotlib not available. Skipping visualization.")
    
    # Test with new data
    X_test = np.array([[5.0], [7.5], [10.0]])
    y_test_pred = model.predict(X_test)
    
    print(f"\nPredictions for test data:")
    for i, (x_val, y_val) in enumerate(zip(X_test.flatten(), y_test_pred)):
        print(f"X = {x_val:.1f} -> y = {y_val:.2f}")


if __name__ == "__main__":
    demonstrate_linear_regression()