import numpy as np
import torch


class LinearRegression:
    """
    A linear regression model that uses closed form solution to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.b = 0
        self.w = None
        # w = [b, w]
        # x = [1, x]
        # y = xw
        # OLS: loss = ||y-xw||^2, want w*
        # dl/dw = -2x.T(Y-Xw)
        # w = (X.TX)^-1(X^TY) <-- closed form solution

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The y values.
        Returns:
            None
        """
        # augument x by a column of ones (column wise concat)
        # X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.linalg.inv(X.T @ X) @ (X.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        # y = xw
        # X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.w


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def se(self, preds: torch.tensor, targets: torch.tensor) -> torch.tensor:
        """
        Computes the squared error
        """
        dif = preds - targets
        return torch.sum(dif * dif)

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fits the model using stochastic gradient descent.

        Arguments:
            X (np.ndarray): The input data.
            Y (np.ndarray): The target data.
            lr (float): The learning rate.
            epochs (int): Epochs

        Returns:
            None.

        """
        self.X = torch.from_numpy(X.astype(float))
        self.X.requires_grad_()
        self.y = torch.from_numpy(y.astype(float))
        self.y.requires_grad_()
        self.lr = lr
        self.epochs = epochs
        if self.w is None:
            self.w = torch.zeros(X.shape)
        else:
            self.w = torch.from_numpy(self.w)
        self.w.requires_grad_()

        # OLS: loss = ||y-xw||^2
        # 1. compute dl/dw = -2x.T(Y-Xw)
        # clip if gradient is too large
        # 2. w' = w - lr*(dl/dw)

        N, D = X.shape
        # self.X = np.hstack((np.ones((N, 1)), X))

        for e in range(self.epochs):
            # print(self.X.shape, 'x shape')
            # print(self.w.shape, 'w shape')
            preds = self.X.float() @ self.w.float()
            loss = self.se(preds, self.y)
            loss.backward()
            self.w.data = self.w.data - self.lr * self.w.grad.data
            self.w.grad.data.zero_()
            # print(e, loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        #N = X.shape[0]
        # n
        N = 0
        # X = np.hstack((np.ones((N, 1)), X))
        w = self.w.detach().numpy()
        return X @ w
