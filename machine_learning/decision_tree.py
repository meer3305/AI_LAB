"""
Decision Tree Implementation
Author: AI Lab
Description: Decision tree classifier using entropy and information gain
"""

import math
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Prediction value (for leaf nodes)


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def calculate_entropy(self, y):
        """Calculate entropy of a set of labels"""
        if len(y) == 0:
            return 0
        
        label_counts = Counter(y)
        total_samples = len(y)
        
        entropy = 0
        for count in label_counts.values():
            probability = count / total_samples
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def split_data(self, X, y, feature, threshold):
        """Split data based on feature and threshold"""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        return X_left, y_left, X_right, y_right
    
    def calculate_information_gain(self, y, y_left, y_right):
        """Calculate information gain from a split"""
        parent_entropy = self.calculate_entropy(y)
        
        n_total = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        weighted_entropy = (n_left / n_total) * self.calculate_entropy(y_left) + \
                          (n_right / n_total) * self.calculate_entropy(y_right)
        
        return parent_entropy - weighted_entropy
    
    def find_best_split(self, X, y):
        """Find the best feature and threshold for splitting"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            feature_values = sorted(set(X[:, feature]))
            
            # Try each unique value as a potential threshold
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                _, y_left, _, y_right = self.split_data(X, y, feature, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                gain = self.calculate_information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(set(y)) == 1 or 
            len(y) < self.min_samples_split):
            # Create leaf node with most common class
            most_common_class = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_class)
        
        # Find best split
        best_feature, best_threshold, best_gain = self.find_best_split(X, y)
        
        if best_gain == 0:
            # No meaningful split found, create leaf node
            most_common_class = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_class)
        
        # Split data
        X_left, y_left, X_right, y_right = self.split_data(X, y, best_feature, best_threshold)
        
        # Recursively build left and right subtrees
        left_child = self.build_tree(X_left, y_left, depth + 1)
        right_child = self.build_tree(X_right, y_right, depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_child, right=right_child)
    
    def fit(self, X, y):
        """Train the decision tree"""
        self.root = self.build_tree(X, y)
        return self
    
    def predict_sample(self, x, node=None):
        """Predict class for a single sample"""
        if node is None:
            node = self.root
        
        if node.value is not None:
            # Leaf node
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)
    
    def predict(self, X):
        """Predict classes for multiple samples"""
        return [self.predict_sample(x) for x in X]
    
    def print_tree(self, node=None, depth=0, prefix="Root"):
        """Print the decision tree structure"""
        if node is None:
            node = self.root
        
        indent = "  " * depth
        
        if node.value is not None:
            print(f"{indent}{prefix}: Predict {node.value}")
        else:
            print(f"{indent}{prefix}: Feature {node.feature} <= {node.threshold:.2f}")
            self.print_tree(node.left, depth + 1, "Left")
            self.print_tree(node.right, depth + 1, "Right")


def generate_sample_data():
    """Generate sample data for demonstration"""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    # Generate synthetic classification dataset
    n_samples = 100
    
    # Features: age, income (in thousands)
    ages = np.random.randint(18, 70, n_samples)
    incomes = np.random.randint(20, 100, n_samples)
    
    # Target: loan approval (0 = rejected, 1 = approved)
    # Simple rule: approve if age > 25 AND income > 40
    labels = []
    for age, income in zip(ages, incomes):
        if age > 25 and income > 40:
            # High probability of approval
            label = 1 if random.random() > 0.1 else 0
        else:
            # Low probability of approval
            label = 1 if random.random() > 0.8 else 0
        labels.append(label)
    
    X = np.column_stack([ages, incomes])
    y = np.array(labels)
    
    return X, y


def demonstrate_decision_tree():
    """Demonstrate decision tree classifier"""
    print("=== Decision Tree Demonstration ===")
    
    # Generate sample data
    X, y = generate_sample_data()
    
    print(f"Generated {len(X)} samples for loan approval prediction")
    print("Features: [Age, Income (thousands)]")
    print("Target: Loan Approval (0 = Rejected, 1 = Approved)")
    
    # Show sample data
    print(f"\nSample data points:")
    for i in range(5):
        print(f"  Age: {X[i][0]}, Income: {X[i][1]}k -> {'Approved' if y[i] else 'Rejected'}")
    
    # Split data into train and test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train decision tree
    dt = DecisionTree(max_depth=5, min_samples_split=5)
    dt.fit(X_train, y_train)
    
    print(f"\nDecision Tree Structure:")
    dt.print_tree()
    
    # Make predictions
    train_predictions = dt.predict(X_train)
    test_predictions = dt.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = sum(p == t for p, t in zip(train_predictions, y_train)) / len(y_train)
    test_accuracy = sum(p == t for p, t in zip(test_predictions, y_test)) / len(y_test)
    
    print(f"\nPerformance:")
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Show some test predictions
    print(f"\nSample test predictions:")
    for i in range(min(5, len(X_test))):
        actual = "Approved" if y_test[i] else "Rejected"
        predicted = "Approved" if test_predictions[i] else "Rejected"
        match = "✓" if y_test[i] == test_predictions[i] else "✗"
        print(f"  Age: {X_test[i][0]}, Income: {X_test[i][1]}k -> "
              f"Actual: {actual}, Predicted: {predicted} {match}")


def analyze_entropy_example():
    """Demonstrate entropy calculation"""
    print("\n=== Entropy Analysis Example ===")
    
    dt = DecisionTree()
    
    # Example datasets with different entropy levels
    examples = [
        ([1, 1, 1, 1], "All same class"),
        ([1, 0, 1, 0], "Balanced classes"),
        ([1, 1, 1, 0], "Mostly one class"),
        ([1, 0, 0, 0, 0], "Imbalanced classes")
    ]
    
    print("Entropy examples:")
    for labels, description in examples:
        entropy = dt.calculate_entropy(labels)
        print(f"  {description}: {labels} -> Entropy = {entropy:.3f}")


def test_feature_importance():
    """Analyze which features are most important"""
    print("\n=== Feature Importance Analysis ===")
    
    X, y = generate_sample_data()
    
    dt = DecisionTree(max_depth=3)
    dt.fit(X, y)
    
    print("Decision tree uses these features for splitting:")
    
    def count_feature_usage(node, feature_counts):
        if node.value is None:  # Not a leaf node
            feature_counts[node.feature] += 1
            count_feature_usage(node.left, feature_counts)
            count_feature_usage(node.right, feature_counts)
    
    feature_counts = [0] * X.shape[1]
    count_feature_usage(dt.root, feature_counts)
    
    feature_names = ["Age", "Income"]
    for i, count in enumerate(feature_counts):
        print(f"  {feature_names[i]}: used {count} times in splits")


if __name__ == "__main__":
    demonstrate_decision_tree()
    analyze_entropy_example()
    test_feature_importance()