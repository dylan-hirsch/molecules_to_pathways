import numpy as np
import warnings
from scipy.linalg import solve_continuous_lyapunov as solve_lyap
from scipy.linalg import solve_continuous_are as solve_are
from scipy.linalg import solve_triangular

def get_bt_transform(A, B, C):
    """
    Computes the transformation matrix T from balanced truncation, where the original system is

    x' = A @ x + B @ u,
    y = C @ x,

    and the transformed "balanced" system corresponding to the change of variables x = Tz is

    z' = (T^{-1} @ A @ T) @ x + (T^{-1} @ B) @ u,
    y = (C @ T) @ x.

    This function takes numpy arrays A,B,C as input and returns (T,T^{-1},S), where S is a vector of the Hankel singular values.
    A must be Hurwitz, (A,B) must be controllable, and (A,C) must be observable.
    Note that the controllability and observability gramians of the balanced system are equal and diagonal, and the diagonal of the
    gramians is precisely S.
    """

    # Compute controllability and observability gramians of the original system
    P = solve_lyap(A, -B @ B.T)
    Q = solve_lyap(A.T, -C.T @ C)

    # Cholesky factorize the controllability and observability grammians
    R = np.linalg.cholesky(P)
    L = np.linalg.cholesky(Q)

    U, S, V = np.linalg.svd(L.T @ R)

    # Compute the transformation matrix relating z and x, i.e. x = Tz, along with the inverse matrix Tinv
    T = R @ V @ np.diag(np.sqrt(1 / S))
    Tinv = np.diag(np.sqrt(1 / S)) @ U.T @ L.T

    return T, Tinv, S

def get_lqg_bt_transform(A, B, C):

    """
    See "A New Set of invariants for Linear Systems -- Application to Reduced Order Compensator Design" by Jonckheere and Silverman (IEEE TAC, 1983).
    This is the original paper on LQG balancing.
    It is assumed (A,B) is controllable and (A,C) is observable.
    """

    # Compute solutions to algebraic ricatti equations
    Rc = np.eye(B.shape[1])
    Qc = C.T @ C
    P = solve_are(A, B, Qc, Rc)

    Ro = np.eye(C.shape[0])
    Qo = B @ B.T
    Pi = solve_are(A.T, C.T, Qo, Ro)

    Rx = np.linalg.cholesky(P, upper = True)
    Ry = np.linalg.cholesky(Pi, upper = False)
    
    U,S,VT = np.linalg.svd(Rx @ Ry)

    T = Ry @ VT.T @ np.diag(1 / np.sqrt(S))
    Tinv = np.diag(1 / np.sqrt(S)) @ U.T @ Rx

    error = np.linalg.norm(T.T @ P @ T - Tinv @ Pi @ Tinv.T, 2)

    return T, Tinv, S, error

def get_minimal_realization(A, B, C):
    pass

def main():
    A = np.array([[1,1],[0,1]])
    B = np.array([[0],[1]]).reshape([2,1])
    C = np.array([[1, 0]]).reshape([1,2])
    T, Tinv, S, error = get_lqg_bt_transform(A, B, C)
    print(error)

if __name__ == "__main__":
    main()