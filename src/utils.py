import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def reshape_for_lstm(X):
    return np.reshape(X.values, (X.shape[0], 1, X.shape[1]))

def save_model(model, filename):
    joblib.dump(model, filename)

def plot_actual_vs_predicted(y_test, y_pred, title="Actual vs Predicted"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

def plot_residuals(y_test, y_pred, title="Residuals"):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title(title)
    plt.show()

def plot_feature_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()
