import numpy as np


class GradDescLinearRegressor: 
    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters


    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape

        self.weights = np.random.rand(self.n_features)
        self.bias = 0.

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = np.dot(X.T, y_pred - y) / self.n_samples # partial derivitive of MSE w.r.t weights
            db = np.sum(y_pred - y) / self.n_samples # partial derivitive of MSE w.r.t. bias

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class OLSLinearRegressor:
    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        X_with_bias = np.c_[np.ones((self.n_samples, 1)), X] # add bias term (X0 = 1 so bias*X0 == bias)
        self.weights_with_bias = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y) # analytical solution
    
    def predict(self, X):
        return np.dot(X, self.weights_with_bias)


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", default=5_000)
    parser.add_argument("--n_features", default=5)

    args = parser.parse_args()

    X, y, coef = make_regression(
        n_samples=args.n_samples,
        n_features=args.n_features, 
        coef=True
    )

    lin_regressor_grad = GradDescLinearRegressor()
    lin_regressor_grad.fit(X, y)

    lin_regressor_ols = OLSLinearRegressor()
    lin_regressor_ols.fit(X, y)

    print(coef)
    print(lin_regressor_grad.weights)
    print(lin_regressor_ols.weights_with_bias[1:])



