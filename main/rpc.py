import numpy as np
import random
from .helpers import approximate_solution_s, compute_P_new, delta_J

class RPC:
    def __init__(self, epsilon):
        """
        Initialize the RPC class with the given convergence threshold.

        Args:
            epsilon (float): Convergence threshold.
        """
        self.epsilon = epsilon

    def fit(self, df):
        """
        Fit the RPC model to the given dataset.

        Args:
            df (pandas.DataFrame): Input data.

        Returns:
            tuple: Coefficients of the polynomial (P) and corresponding scores.
        """
        X = df.values
        X = X.T
        dim = X.shape[0]

        # Initialize P_t with random coefficients
        P_t = np.array([np.zeros(dim),
                        X.T[random.randint(0, X.shape[1] - 1)],
                        X.T[random.randint(0, X.shape[1] - 1)],
                        np.ones(dim)]).T
        
        # Calculate scores using approximate_solution_s
        score_t = [approximate_solution_s(P=P_t, x=[[j] for j in X[:, i]]) for i in range(X.shape[1])]

        # Compute new polynomial coefficients and scores iteratively until convergence
        P_t_1 = compute_P_new(P_t, score_t, X)
        score_t_1 = [approximate_solution_s(P_t_1, x=[[j] for j in X[:, i]]) for i in range(X.shape[1])]

        while delta_J(P_t, score_t, P_t_1, score_t_1, X) > self.epsilon:
            P_t = P_t_1
            score_t = score_t_1
            P_t_1 = compute_P_new(P_t, score_t, X)
            score_t_1 = [approximate_solution_s(P_t_1, x=[[j] for j in X[:, i]]) for i in range(X.shape[1])]

            if delta_J(P_t, score_t, P_t_1, score_t_1, X) < 0:
                break

        return P_t_1, score_t_1
