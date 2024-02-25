import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            #graident descent
            dj_dw = (1/n_samples) * np.dot(X.T, (y_pred -y))
            dj_db = (1/n_samples) * np.sum(y_pred -y)

            #dynamically updating

            self.weights = self.weights - self.lr * dj_dw
            self.bias = self.bias - self.lr * dj_db

    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    

    def mean_squared_error(self, y_pred , y_true):
        mse = np.mean((y_true - y_pred)**2)
        return mse
    
    def mean_absolute_error(self, y_pred , y_true):
        mae = np.mean(abs(y_true - y_pred))
        return mae
    
    def r2_score(self, y_true, y_pred):
        mean_y_true = np.mean(y_true)
        ss_total = np.sum((y_true - mean_y_true)**2)
        ss_residual = np.sum((y_true - y_pred)**2)
        r2 = 1 - (ss_residual / ss_total)
        return r2

