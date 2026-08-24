import numpy as np

from src.data import load_data
from src.model import predict, predict_class, gradient_descent
from src.metrics import accuracy, precision, recall, f1_score, confusion_matrix
from src.visualize import plot_data, plot_cost, plot_decision_boundary

def main():
    # Load the dataset
    X, Y, X_test, Y_test, mean, std = load_data()

    # Train the model using gradient descent
    w, b, cost_history = gradient_descent(X, Y, w_in=np.array([2.0] * X.shape[1]), b_in=3, alpha=0.0001, num_iters=50000)

    # Make predictions on the test set
    probabilities = predict(w, b, X_test)
    predicted_classes = predict_class(probabilities)

    # Evaluate the model
    acc = accuracy(Y_test, predicted_classes)
    prec = precision(Y_test, predicted_classes)
    rec = recall(Y_test, predicted_classes)
    f1 = f1_score(Y_test, predicted_classes)
    cm = confusion_matrix(Y_test, predicted_classes)

    print("Predicted classes:", predicted_classes)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Visualize the results
    plot_data(X, Y, mean, std)
    plot_cost(cost_history)
    plot_decision_boundary(X, Y, w, b, mean, std)

if __name__ == "__main__":
    main()

        