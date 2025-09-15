"""
Logistic Regression Implementation
Author: AI Lab
Description: Logistic regression for binary classification using gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the logistic regression model"""
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Calculate cost (log-likelihood)
            cost = self.compute_cost(y, predictions)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def compute_cost(self, y_true, y_pred):
        """Compute logistic regression cost (cross-entropy)"""
        # Avoid log(0) by clipping
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


def generate_binary_classification_data(n_samples=200, random_state=42):
    """Generate sample data for binary classification"""
    np.random.seed(random_state)
    
    # Generate two classes with some overlap
    # Class 0: centered around (2, 2)
    X_class0 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
    y_class0 = np.zeros(n_samples//2)
    
    # Class 1: centered around (5, 5)
    X_class1 = np.random.multivariate_normal([5, 5], [[1, 0.5], [0.5, 1]], n_samples//2)
    y_class1 = np.ones(n_samples//2)
    
    # Combine classes
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1])
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def demonstrate_logistic_regression():
    """Demonstrate logistic regression with sample data"""
    print("=== Logistic Regression Demonstration ===")
    
    # Generate sample data
    X, y = generate_binary_classification_data(200)
    
    print(f"Generated {len(X)} data points for binary classification")
    print(f"Features: 2D coordinates")
    print(f"Classes: 0 and 1")
    
    # Split data into train and test sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train logistic regression
    model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_probabilities = model.predict_proba(X_train)
    test_probabilities = model.predict_proba(X_test)
    
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\nTrained model parameters:")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias:.4f}")
    print(f"Final cost: {model.cost_history[-1]:.4f}")
    
    print(f"\nPerformance:")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Show some predictions with probabilities
    print(f"\nSample predictions on test set:")
    print("Features\t\tActual\tPredicted\tProbability")
    print("-" * 55)
    for i in range(min(5, len(X_test))):
        actual = int(y_test[i])
        predicted = test_predictions[i]
        prob = test_probabilities[i]
        print(f"[{X_test[i][0]:.2f}, {X_test[i][1]:.2f}]\t{actual}\t{predicted}\t\t{prob:.3f}")
    
    # Visualize results
    try:
        plt.figure(figsize=(15, 5))
        
        # Plot training data and decision boundary
        plt.subplot(1, 3, 1)
        
        # Plot training data
        class_0_idx = y_train == 0
        class_1_idx = y_train == 1
        plt.scatter(X_train[class_0_idx, 0], X_train[class_0_idx, 1], 
                   c='red', alpha=0.6, label='Class 0')
        plt.scatter(X_train[class_1_idx, 0], X_train[class_1_idx, 1], 
                   c='blue', alpha=0.6, label='Class 1')
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        mesh_points = np.column_stack([xx.ravel(), yy.ravel()])
        Z = model.predict_proba(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        plt.colorbar(plt.contourf(xx, yy, Z, levels=50, alpha=0.3))
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Logistic Regression Decision Boundary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot cost history
        plt.subplot(1, 3, 2)
        plt.plot(model.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (Cross-Entropy)')
        plt.title('Training Cost History')
        plt.grid(True, alpha=0.3)
        
        # Plot probability distribution
        plt.subplot(1, 3, 3)
        
        probabilities_class0 = train_probabilities[y_train == 0]
        probabilities_class1 = train_probabilities[y_train == 1]
        
        plt.hist(probabilities_class0, bins=20, alpha=0.6, label='Class 0', color='red')
        plt.hist(probabilities_class1, bins=20, alpha=0.6, label='Class 1', color='blue')
        plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/logistic_regression_demo.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("\nPlot saved as '/tmp/logistic_regression_demo.png'")
        
    except ImportError:
        print("\nMatplotlib not available. Skipping visualization.")


def compare_with_linear_regression():
    """Compare logistic regression with linear regression for classification"""
    print("\n=== Logistic vs Linear Regression Comparison ===")
    
    # Generate sample data
    X, y = generate_binary_classification_data(100)
    
    # Train logistic regression
    logistic_model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    logistic_model.fit(X, y)
    
    # Train linear regression for comparison
    from machine_learning.linear_regression import LinearRegression
    linear_model = LinearRegression(learning_rate=0.1, max_iterations=1000)
    linear_model.fit(X, y)
    
    # Make predictions
    logistic_pred = logistic_model.predict(X)
    logistic_proba = logistic_model.predict_proba(X)
    linear_pred = linear_model.predict(X)
    
    # Convert linear predictions to binary (threshold at 0.5)
    linear_binary = (linear_pred >= 0.5).astype(int)
    
    # Calculate accuracies
    logistic_accuracy = np.mean(logistic_pred == y)
    linear_accuracy = np.mean(linear_binary == y)
    
    print(f"Logistic Regression accuracy: {logistic_accuracy:.4f}")
    print(f"Linear Regression accuracy: {linear_accuracy:.4f}")
    
    print(f"\nKey differences:")
    print(f"- Logistic regression outputs probabilities in [0, 1]")
    print(f"- Linear regression can output values outside [0, 1]")
    print(f"- Logistic regression uses sigmoid function")
    print(f"- Linear regression uses linear function")
    
    # Show range of outputs
    print(f"\nOutput ranges:")
    print(f"Logistic probabilities: [{logistic_proba.min():.3f}, {logistic_proba.max():.3f}]")
    print(f"Linear predictions: [{linear_pred.min():.3f}, {linear_pred.max():.3f}]")


def analyze_sigmoid_function():
    """Analyze the sigmoid function properties"""
    print("\n=== Sigmoid Function Analysis ===")
    
    model = LogisticRegression()
    
    # Test sigmoid function with various inputs
    test_inputs = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    
    print("Input\tSigmoid Output\tInterpretation")
    print("-" * 45)
    for x in test_inputs:
        sigmoid_out = model.sigmoid(x)
        
        if sigmoid_out < 0.1:
            interpretation = "Strong Class 0"
        elif sigmoid_out < 0.4:
            interpretation = "Lean Class 0"
        elif sigmoid_out < 0.6:
            interpretation = "Uncertain"
        elif sigmoid_out < 0.9:
            interpretation = "Lean Class 1"
        else:
            interpretation = "Strong Class 1"
        
        print(f"{x}\t{sigmoid_out:.4f}\t\t{interpretation}")
    
    print(f"\nSigmoid properties:")
    print(f"- Maps any real number to (0, 1)")
    print(f"- S-shaped curve")
    print(f"- Derivative: σ(x) * (1 - σ(x))")
    print(f"- Used for probability interpretation")


def demonstrate_multiclass_extension():
    """Demonstrate how logistic regression can be extended to multiclass"""
    print("\n=== Multiclass Extension (One-vs-Rest) ===")
    
    # Generate 3-class data
    np.random.seed(42)
    n_samples_per_class = 50
    
    # Class 0: center (1, 1)
    X_class0 = np.random.multivariate_normal([1, 1], [[0.5, 0], [0, 0.5]], n_samples_per_class)
    
    # Class 1: center (4, 1)  
    X_class1 = np.random.multivariate_normal([4, 1], [[0.5, 0], [0, 0.5]], n_samples_per_class)
    
    # Class 2: center (2.5, 4)
    X_class2 = np.random.multivariate_normal([2.5, 4], [[0.5, 0], [0, 0.5]], n_samples_per_class)
    
    X_multi = np.vstack([X_class0, X_class1, X_class2])
    y_multi = np.hstack([np.zeros(n_samples_per_class), 
                        np.ones(n_samples_per_class), 
                        np.full(n_samples_per_class, 2)])
    
    print(f"Generated {len(X_multi)} samples for 3-class classification")
    
    # Train one-vs-rest classifiers
    classifiers = {}
    accuracies = {}
    
    for class_label in [0, 1, 2]:
        # Create binary labels (current class vs all others)
        y_binary = (y_multi == class_label).astype(int)
        
        # Train binary classifier
        classifier = LogisticRegression(learning_rate=0.1, max_iterations=500)
        classifier.fit(X_multi, y_binary)
        
        classifiers[class_label] = classifier
        
        # Calculate accuracy for this binary classifier
        predictions = classifier.predict(X_multi)
        accuracy = np.mean(predictions == y_binary)
        accuracies[class_label] = accuracy
        
        print(f"Class {class_label} vs Rest - Accuracy: {accuracy:.4f}")
    
    # Make multiclass predictions using highest probability
    print(f"\nMulticlass prediction strategy:")
    print(f"- Train 3 binary classifiers (one for each class)")
    print(f"- For prediction, choose class with highest probability")
    
    # Example predictions
    test_points = np.array([[1, 1], [4, 1], [2.5, 4]])
    print(f"\nSample predictions:")
    
    for i, point in enumerate(test_points):
        point = point.reshape(1, -1)
        probabilities = []
        
        for class_label in [0, 1, 2]:
            prob = classifiers[class_label].predict_proba(point)[0]
            probabilities.append(prob)
        
        predicted_class = np.argmax(probabilities)
        print(f"Point {point.flatten()}: Class {predicted_class} "
              f"(probabilities: {probabilities})")


if __name__ == "__main__":
    demonstrate_logistic_regression()
    compare_with_linear_regression()
    analyze_sigmoid_function()
    demonstrate_multiclass_extension()