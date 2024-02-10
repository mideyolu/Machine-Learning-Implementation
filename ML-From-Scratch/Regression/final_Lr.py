import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            F_wb = np.dot(X, self.weights) + self.bias

            # gradient descent
            dj_dw = (1/n_samples) * np.dot(X.T, (F_wb - y))
            dj_db = (1/n_samples) * np.sum(F_wb - y)

            self.weights = self.weights - self.lr * dj_dw
            self.bias = self.bias - self.lr * dj_db
    
    def predict(self, X):
        F_wb = np.dot(X, self.weights) + self.bias
        return F_wb
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def mae(self, y_true, y_pred):
        return np.mean(abs(y_true - y_pred))
    
    def r2(self, y_true, y_pred):
        mean_y_true = np.mean(y_true)
        ss_total = np.sum((y_true - mean_y_true)**2)
        ss_residual = np.sum((y_true - y_pred)**2)
        r2 = 1 - (ss_residual / ss_total)
        return r2
