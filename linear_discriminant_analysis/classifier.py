import numpy as np
from collections import namedtuple

class LDAClasifier:
    def fit(self, X, y):
        self.cov = np.cov(X.T)
        self.mu = []
        self.pi = []

        self.classes = np.unique(y)
        for k in self.classes:
            self.mu.append(X[y==k].mean(axis=0))
            self.pi.append(len(X[y==k])/len(X))

        self.cov_inv = np.linalg.inv(self.cov)
    
    def get_discriminant(self, x, k):
        return x.T @ self.cov_inv @ self.mu[k] - 0.5 * self.mu[k].T @ self.cov_inv @ self.mu[k] + np.log(self.pi[k])

    def predict(self, X):
        predictions = []

        for x in X:
            max_discriminant = max([(self.get_discriminant(x, k), k) for k in self.classes])
            predictions.append(max_discriminant[1])
        
        return predictions

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X, y = make_classification(
        n_samples=10_000,
        n_classes=2,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifier = LDAClasifier()
    classifier.fit(X_train, y_train)

    y_test_pred = classifier.predict(X_test)

    sklearn_classifier = LinearDiscriminantAnalysis(solver="lsqr")
    sklearn_classifier.fit(X_train, y_train)

    print("my model", accuracy_score(y_true=y_test, y_pred=y_test_pred))
    print("sklearn model", accuracy_score(y_true=y_test, y_pred=sklearn_classifier.predict(X_test)))


