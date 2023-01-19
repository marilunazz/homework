import numpy as np


class LinearRegression:
    w: np.ndarray
    b: float

    def __init__(self):
        pass
        # w = [b, w]
        # x = [1, x]
        # y = xw
        # OLS: loss = ||y-xw||^2, want w*
        # dl/dw = -2x.T(Y-Xw)
        # w = (X.TX)^-1(X^TY) <-- closed form solution

    def fit(self, X, y):
      # augument x by a column of ones (column wise concat)
      X = np.c_[np.ones(X.shape[0]), X]
      # closed form solution
      self.W = np.linalg.inv(X.T @ X) @ (X.T @ y)

    def predict(self, X):
      X = np.c_[np.ones(X.shape[0]), X]
      return (X @ self.W)


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000) -> None:
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        raise NotImplementedError()
