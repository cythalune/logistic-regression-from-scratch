import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(w, b, X):
    """
    Predict probabilities using the logistic regression model.

    Parameters:
        w (np.ndarray): Model weights.
        b (float): Model bias.
        X (np.ndarray): Input features.

    Returns:
        np.ndarray: Predicted probabilities.
    """

    z = np.dot(w.T, X) + b
    a = sigmoid(z)

    return a


def compute_cost(a, Y):
    """
    Compute the binary cross-entropy cost.

    Parameters:
        a (np.ndarray): Predicted probabilities.
        Y (np.ndarray): Actual labels.

    Returns:
        float: Binary cross-entropy cost.
    """

    m = Y.shape[0]

    total_cost = 0

    for i in range(m):
        cost_i = Y[i] * np.log(a[i]) + (1 - Y[i]) * np.log(1 - a[i])
        total_cost += cost_i

    total_cost = -total_cost / m

    return total_cost


def compute_gradient(X, Y, w, b):
    """
    Compute the gradients of the cost function.

    Parameters:
        X (np.ndarray): Input features.
        Y (np.ndarray): Actual labels.
        w (np.ndarray): Model weights.
        b (float): Model bias.

    Returns:
        np.ndarray: Gradient with respect to w.
        float: Gradient with respect to b.
    """

    dj_dw = np.zeros(w.shape)
    dj_db = 0

    m = X.shape[1]

    for i in range(m):
        f_wb = predict(w, b, X[:, i])

        error = f_wb - Y[i]

        dj_dw += error * X[:, i]
        dj_db += error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(X, Y, w_in, b_in, alpha, num_iters):
    """
    Train the logistic regression model using gradient descent.

    Parameters:
        X (np.ndarray): Input features.
        Y (np.ndarray): Actual labels.
        w_in (np.ndarray): Initial weights.
        b_in (float): Initial bias.
        alpha (float): Learning rate.
        num_iters (int): Number of iterations.

    Returns:
        np.ndarray: Trained weights.
        float: Trained bias.
        list: Cost history.
    """

    w = w_in.copy()
    b = b_in

    cost_history = []

    for i in range(num_iters):

        dj_dw, dj_db = compute_gradient(X, Y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        a = predict(w, b, X)
        cost = compute_cost(a, Y)

        cost_history.append(cost)

        if num_iters > 10 and i % round(num_iters / 10) == 0:
            print(f"Iterations: {i}  Cost: {cost}")

    return w, b, cost_history

compute_gradient(np.array([[1, 2], [3, 4]]), np.array([0, 1]), np.array([0.5, -0.5]), 0.1)