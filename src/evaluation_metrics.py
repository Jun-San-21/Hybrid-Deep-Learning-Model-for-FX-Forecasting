from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, precision_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np

def calculate_metrics(y_test, y_pred_class):

    mse = mean_squared_error(y_test, y_pred_class)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_class),
        'f1_score': f1_score(y_test, y_pred_class),
        'precision': precision_score(y_test, y_pred_class),
        'recall': recall_score(y_test, y_pred_class),
    }
    return metrics