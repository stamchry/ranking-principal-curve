import numpy as np
import scipy.optimize as opt

def f(s, P):
    M_1 = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    z_1 = np.array([[1], [s], [s**2], [s**3]], dtype=object)
    return np.linalg.multi_dot([P, M_1, z_1])

def derf(s, P):
    M_2 = np.array([[-3, 6, 3], [9, -12, 3], [-9, 6, 0], [3, 0, 0]])
    z_2 = np.array([[1], [s], [s**2]], dtype=object)
    return np.linalg.multi_dot([P, M_2, z_2])

def dist(s, P, x):
    return np.dot(derf(s, P).T, (x - f(s, P))).item()

def approximate_solution_s(P, x):
    return opt.fsolve(dist, args=(P, x), x0=np.array([0.5]))

def compute_P_new(P_old, score, X):
    X = np.array(X)
    M = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    Z = np.array([[1 for s in score], [s for s in score], [(s**2) for s in score], [(s**3) for s in score]], dtype=object)
    Matrix = np.array((np.dot(np.dot(M, Z), np.dot(M, Z).T)), dtype=float)
    gamma = 2 / (np.linalg.eig(Matrix)[0].max() + np.linalg.eig(Matrix)[0].min())
    preconditioner_D = np.diag(np.linalg.norm(Matrix, axis=1))
    P_new = P_old - gamma * (np.dot((np.linalg.multi_dot([P_old, np.dot(M, Z), np.dot(M, Z).T]) - np.linalg.multi_dot([X, np.dot(M, Z).T])), np.linalg.inv(preconditioner_D)))
    return P_new

def J(P, score, X):
    P = np.array(P)
    X = np.array(X)
    M = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    Z = np.array([[1, s, s**2, s**3] for s in score], dtype=object).T
    return np.trace(np.dot(X.T, X)) - 2 * (np.trace(np.linalg.multi_dot([P, M, Z, X.T]))) + np.trace(np.linalg.multi_dot([P, M, Z, Z.T, M.T, P.T]))

def delta_J(P_old, score_old, P_new, score_new, X):
    return J(P_old, score_old, X) - J(P_new, score_new, X)
