import numpy as np
from scipy.linalg import solve_continuous_lyapunov as solve_lyap

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
