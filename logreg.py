import numpy as np
import math


class LogisticRegression:

    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
                Args:
                    step_size: Step size for iterative solvers only.
                    max_iter: Maximum number of iterations for the solver.
                    eps: Threshold for determining convergence.
                    theta_0: Initial guess for theta. If None, use the zero vector.
                    verbose: Print loss values during training.
                """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """

        # *** START CODE HERE ***

        def print_if_needed(message):
            if iteration_num % 100 == 0:
                print(message)

        current_eps = 5 * self.eps
        d = x.shape[1]
        n = x.shape[0]

        grad_J = np.zeros((d, 1))
        H = np.zeros((d, d))
        theta = self.theta

        iteration_num = 0
        while current_eps >= self.eps and iteration_num < self.max_iter:
            print_if_needed("Starting iteration: " + str(iteration_num))
            print_if_needed("Norm is now: " + str(current_eps))
            loss = 0

            for i in range(n):
                x_i = x[i].reshape((d, 1))
                exp_i = math.exp(-1 * np.dot(np.transpose(theta), x_i))
                hypothesis_i = 1.0 / (1.0 + exp_i)
                loss += (y[i] * math.log(hypothesis_i)) + (1 - y[i]) * math.log(exp_i * hypothesis_i)

                grad_J_i_scalar = y[i] - hypothesis_i

                grad_J_i = grad_J_i_scalar * x_i

                z_i = hypothesis_i * exp_i * hypothesis_i
                H_i = z_i * np.outer(x_i, x_i)

                grad_J += grad_J_i
                H += H_i

            grad_J = grad_J / (-1 * n)
            H = H / n
            loss = loss / (-1 * n)

            if self.verbose:
                print_if_needed("Loss is now: " + str(loss))

            theta_update = np.matmul(np.linalg.inv(H), grad_J)

            step_size = self.step_size
            current_eps = np.linalg.norm(step_size * theta_update)
            theta = theta - step_size * theta_update

            iteration_num += 1

        print("Converged on iteration: " + str(iteration_num))
        np.savetxt("theta.txt", theta)
        self.theta = theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        dot_product = np.matmul(x, self.theta)
        exp_vector = np.exp(-1 * dot_product)
        return 1 / (1 + exp_vector)
        # *** END CODE HERE ***


