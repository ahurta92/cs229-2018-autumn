import numpy as np
from matplotlib import pyplot as plt

import util

from linear_model import LinearModel


# theta = theta - inv(H)*grad(l(theta))

def sigmoid(z: np.array):
    return 1 / (1 + np.exp(-z))


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    plt.plot(x_train, y_train, 'o')

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    y_eval=clf.predict(x_eval)

    util.plot(x_eval, y_eval, clf.theta)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


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
            theta = np.zeros((n, 1), dtype=np.float64)
        else:
            theta = self.theta
        epsilon = 1000
        iter = 0
        while iter < self.max_iter & epsilon < self.eps:
            sig_z = sigmoid(x @ theta)
            grad_J = 1 / m * x.transpose() @ (sig_z - y)
            H = 1 / m * (x.transpose() @ sig_z @ (np.ones((1, m), dtype=np.float64) - sig_z.reshape(1, m)))

            theta_update = theta - np.linalg.inv(H) @ grad_J
            epsilon = (theta_update - theta).sum()
            print('k: ', iter, ' eps = ', epsilon)
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
        # *** END CODE HERE ***

# code starts to run here

train_path = "..\data\ds1_train.csv"
eval_path = "..\data\ds1_valid.csv"

main(train_path, eval_path, '')






















