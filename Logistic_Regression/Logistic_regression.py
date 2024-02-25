import numpy as np


# def sigmoid(x):
#     g = 1/(1+ np.exp(-x))
#     return g

def sigmoid(x):
    # Normalize input to prevent overflow
    x = np.clip(x, -500, 500)
    g = 1 / (1 + np.exp(-x))
    return g


class LogisticRegression:
    
    def __init__(self, lr=0.01, n_iters =1000):
        self.lr= lr
        self.n_iters= n_iters
        self.weights = None
        self.bias = None
    
    #traning the model
    def fit(self, X,y):

        n_samples, n_features = X.shape
        self.weights= np.zeros(n_features)
        self.bias =0


        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_pred)


            #graident descent

            dj_dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            dj_db = (1/n_samples) * np.sum( y_pred - y)

            #dynamically updating
            self.weights = self.weights - self.lr * dj_dw
            self.bias = self.bias - self.lr * dj_db

    #making predictions
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)

        threshold_pred = [0 if y <= 0.5 else 1 for y in y_pred ]
        class_pred = threshold_pred

        return class_pred
    
    #accuracy score
    def accuracy_score(self,y_pred, y_true):
        acc = np.sum(y_pred == y_true)/ len(y_true)
        return acc
    
    





    






