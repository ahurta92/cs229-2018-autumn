import numpy as np
from matplotlib import pyplot as plt

import util

from linear_model import LinearModel


# theta = theta - inv(H)*grad(l(theta))

def sigmoid(z: np.array):
    return 1 / (1 + np.exp(-z))


def ll_(x, y, theta):
    m, n = x.shape
    llp = 0
    for i in range(m):
        xi = x[i, :]
        yi = y[i]
        ll = yi * np.log(sigmoid(np.dot(xi, theta))) + (1 - yi) * np.log(1 - sigmoid(np.dot(xi, theta)))
        llp = llp + ll
    return llp


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x: np.array, y: np.array):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if self.theta is None:
            theta = np.zeros((n), dtype=np.float64)
        else:
            theta = self.theta
        epsilon = 1000
        iter = 0
        while iter < self.max_iter and epsilon > self.eps:
            sig_z = sigmoid(x @ theta)
            grad_J = 1 / m * np.dot(x.T, (sig_z - y))
            gz = sig_z * (1 - sig_z)
            H = 1 / m * x.transpose() * gz @ x

            theta_update = theta - np.linalg.inv(H) @ grad_J

            epsilon = np.abs((theta_update - theta)).sum()
            print('k: ', iter, ' eps = ', epsilon)
            iter = iter + 1
            theta = theta_update

        # *** END CODE HERE ***
        self.theta = theta

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        prob_y1 = sigmoid(x @ self.theta)
        y_pred = np.zeros(prob_y1.shape)
        y_pred[prob_y1 > 0.5] = 1

        return y_pred


# *** END CODE HERE ***


if __name__ == "__main__":
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # code starts to run here

    train_path = "..\data\ds1_train.csv"
    eval_path = "..\data\ds1_valid.csv"

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_true = util.load_dataset(eval_path, add_intercept=True)
    x_eval = x_eval

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_eval)

    util.plot(x_train, y_train, clf.theta, save_path='train.svg')
    util.plot(x_eval, y_pred, clf.theta, save_path='predict.svg')
    util.plot(x_eval, y_true, clf.theta, save_path='true.svg')

    hello = LogisticRegression.__init__()
    hello.fit(x_train, y_train)
