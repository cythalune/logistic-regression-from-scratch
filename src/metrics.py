import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix for binary classification.

    Parameters:
        y_true (np.ndarray): True labels (0 or 1).
        y_pred (np.ndarray): Predicted labels (0 or 1).

    Returns:
    np.ndarray: A 2x2 confusion matrix in the form
                [[TN, FP],
                 [FN, TP]].
    """
    cm = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm


def accuracy(y_true, y_pred):
    """
    Compute the accuracy of predictions.

    Parameters:
        y_true (np.ndarray): True labels (0 or 1).
        y_pred (np.ndarray): Predicted labels (0 or 1).

    Returns:
        float: Accuracy as a percentage.
    """
    cm = confusion_matrix(y_true, y_pred)
    correct_predictions = cm[0, 0] + cm[1, 1]
    total_predictions = np.sum(cm)
    return (correct_predictions / total_predictions) * 100

def precision(y_true, y_pred):
    """
    Compute the precision of predictions.

    Parameters:
        y_true (np.ndarray): True labels (0 or 1).
        y_pred (np.ndarray): Predicted labels (0 or 1).

    Returns:
        float: Precision as a percentage.
    """
    cm = confusion_matrix(y_true, y_pred)
    true_positive = cm[1, 1]
    false_positive = cm[0, 1]
    if true_positive + false_positive == 0:
        return 0.0
    return (true_positive / (true_positive + false_positive)) * 100

def recall(y_true, y_pred):
    """
    Compute the recall of predictions.

    Parameters:
        y_true (np.ndarray): True labels (0 or 1).
        y_pred (np.ndarray): Predicted labels (0 or 1).

    Returns:
        float: Recall as a percentage.
    """
    cm = confusion_matrix(y_true, y_pred)
    true_positive = cm[1, 1]
    false_negative = cm[1, 0]
    if true_positive + false_negative == 0:
        return 0.0
    return (true_positive / (true_positive + false_negative)) * 100 

def f1_score(y_true, y_pred):
    """
    Compute the F1 score of predictions.

    Parameters:
        y_true (np.ndarray): True labels (0 or 1).
        y_pred (np.ndarray): Predicted labels (0 or 1).

    Returns:
        float: F1 score as a percentage.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return (2 * p * r) / (p + r)    