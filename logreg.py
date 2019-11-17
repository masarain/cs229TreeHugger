import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y, theta=None):
    """Train a logistic regression model."""
    if theta is None:
        theta = np.zeros(X.shape[1])
    learning_rate = 0.1
    # theta = np.array([-3000, 0,0])
    max_iterations = 2000000

    i = 0
    while i < max_iterations:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            print('Norm is: ' + str(np.linalg.norm(prev_theta - theta)))
            print("Theta is: " + str(theta))
            break
        else:
            if i % 10000 == 0:
                print("Current norm is: " + str(np.linalg.norm(prev_theta - theta)))
                print("Current theta is: " + str(theta))
    return theta


def predict(x, theta):
    predicted = -1 * np.matmul(theta, np.transpose(x))
    predicted = np.exp(predicted)
    predicted = 1 / (1 + predicted)
    predicted = [round(x) for x in predicted]

    return predicted
