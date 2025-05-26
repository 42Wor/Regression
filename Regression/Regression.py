import numpy as np

class MyLinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)
        # Add intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Reshape y if needed
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        # Calculate coefficients
        self.coefficients = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        # Convert X to numpy array
        X = np.array(X)
        # Add intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.coefficients

    def score(self, X, y):
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        y_pred = self.predict(X)
        # Calculate R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)