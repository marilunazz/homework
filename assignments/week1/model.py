import numpy as np

class LinearRegression:
    """
    A linear regression model that uses closed form solution to fit the model.
    """
    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = 0
        # w = [b, w]
        # x = [1, x]
        # y = xw
        # OLS: loss = ||y-xw||^2, want w*
        # dl/dw = -2x.T(Y-Xw)
        # w = (X.TX)^-1(X^TY) <-- closed form solution

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The y values.
        Returns:
            None 
        """
        # augument x by a column of ones (column wise concat)
        # X = np.c_[np.ones(X.shape[0]), X]
        # print('hello world XXXXXXXXX')
        # print(X.T @ X)
        # print('pinv')
        # print(np.linalg.inv(X.T @ X))
        # print('x.t @ y')
        # print(X.T @ y)
        # print('answer')
        # print(np.linalg.inv(X.T @ X) @ (X.T @ y))
                
        self.w = np.linalg.inv(X.T @ X) @ (X.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ 
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        y = X @ self.w
        print('hello world XXXXXXXXX')
        print(y.shape)
        return (X @ self.w)

class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def compute_gradient(self, w, X, y):
        N, D = X.shape
        pred = np.dot(X, w)
        loss = np.dot(pred - y, X)
        loss = (2/N)*loss
        dldw = -2* X.T
    
    def has_converged(self, old, new):
        norm = np.sqrt(np.sum((new - old)**2))

        if norm < self.tol:
          return True
      
        return False


    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000) -> None:
        """ fit description """
        self.X = X
        self.y = y
        self.lr = lr
        self.epochs = epochs
        self.w_new = None
        # OLS: loss = ||y-xw||^2
        # 1. compute dl/dw = -2x.T(Y-Xw)
        # clip if gradient is too large
        # 2. w' = w - lr*(dl/dw)
        
        N, D = X.shape
        X = np.hstack((np.ones((N,1)), X))
        curr_w = None

        # Initializing the weights
        if self.w is None:
            curr_w = np.zeros((D + 1,))
        else:
            curr_w = self.w

        for e in range(self.epochs):
            # compute the gradient
            print((-2 * X.T).shape)
            print(curr_w, 'w shape')
            print(self.y.shape, 'y shape')
            print(((self.y - X) @ curr_w).shape)
            grad = -2 * X.T @ ((self.y - X) @ curr_w)
            print(grad, ' grad')
            # clip if gradient is too large
            grad = np.clip(grad, -1, 1)
            print(grad, ' grad clipped')

            # compute w' = w - lr*(dl/dw)
            self.w_new = curr_w - self.lr * grad
            print(self.w_new, 'self.w_new')

            # check if it has converged
            if self.has_converged(curr_w, self.w_new):
                print('CONVERGED')
                break

            curr_w = self.w_new

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        N = X.shape[0]
        X = np.hstack((np.ones((N, 1)), X))
        return np.dot(X, self.w_new)
        



