import numpy as np

class Perceptron:
    """
       Perceptron Classifier

       Parameters
       ===============================================================
            lr : float
                Learning rate (between 0 and 1.0)
            n_iter : int
                Passes over the training dataset
            random_state : int
                Random number initialization to reduce reproducibility

        Attributes
        ===============================================================
        w_ : 1d-array
            Weights after fitting
        b_ : Scalar
            Bias after fitting

        errors_ : list
             List to store number of misclassifications (updates) based on every epochs

    """


    def __init__(self, lr=0.01, n_iters=50, random_state=42):
        self.lr = lr
        self.n_iters = n_iters
        self.random_state = random_state



    def fit(self, X, y):
        """
           Model Training

           Parameters
           ===========================================================================
           X : {arry-like}, shape = [n_examples, n_features]

           y : array-like shape = [n_examples]
               Target values


           Returns
           ======================
           self : object
        """

        # initialize the random state
        regen = np.random.RandomState(self.random_state)
        #setting the initail values of the weights to the random state
        self.w_ = regen.normal(loc=0.0, scale=0.01, size= X.shape[1])

        #setting the initailize value of the bias
        self.b_ = np.float64(0.)

        # List to store errors
        self.errors = []

        for _ in range(self.n_iters):

            errors=0

            for xi, target in zip(X,y):
                update = self.lr * (target - self.predict(xi))

                #dynamically updating the weights
                self.w_ += update * xi

                #dynamically updating the bias
                self.b_ += update

                errors += int(update != 0.0)

            self.errors.append(errors)


        return self


    def net_input(self, X) -> float :
        """
            Calculate net input
        """
        result = np.dot(X, self.w_) + self.b_
        return result

    def predict(self, X) -> int :
        """
            Return the predicted class label
        """

        label = np.where(self.net_input(X) >=0.0, 1,0)

        return label


    def accuracy(self, y_true, y_pred) -> int :
        """
        """

        acc = np.sum(
            y_pred == y_true
        )/ len(y_true)

        return f"Accuracy Score : {acc}"
