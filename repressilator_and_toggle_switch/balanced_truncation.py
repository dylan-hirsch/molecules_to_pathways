import numpy as np
import warnings
from scipy.linalg import solve_continuous_lyapunov as solve_lyap
from scipy.linalg import solve_continuous_are as solve_are
from numpy.linalg import matrix_rank


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
    Pc = solve_lyap(A, -B @ B.T) # controllability gramian
    Po = solve_lyap(A.T, -C.T @ C) # observability gramian

    # Cholesky factorize the controllability and observability grammians
    R = np.linalg.cholesky(Pc)
    L = np.linalg.cholesky(Po)

    U, S, V = np.linalg.svd(L.T @ R)

    # Compute the transformation matrix relating z and x, i.e. x = Tz, along with the inverse matrix Tinv
    T = R @ V @ np.diag(np.sqrt(1 / S))
    Tinv = np.diag(np.sqrt(1 / S)) @ U.T @ L.T

    error = np.linalg.norm(T.T @ Pc @ T - Tinv @ Po @ Tinv.T, 2)

    return T, Tinv, S, error

def get_lqg_bt_transform(A, B, C):

    """
    See "A New Set of invariants for Linear Systems -- Application to Reduced Order Compensator Design" by Jonckheere and Silverman (IEEE TAC, 1983).
    This is the original paper on LQG balancing.
    It is assumed (A,B) is controllable and (A,C) is observable.
    """

    # Compute solutions to algebraic ricatti equations
    Rc = np.eye(B.shape[1])
    Qc = C.T @ C
    Pc = solve_are(A, B, Qc, Rc) # Called P in original paper

    Ro = np.eye(C.shape[0])
    Qo = B @ B.T
    Po = solve_are(A.T, C.T, Qo, Ro) # Called Pi in original paper

    Rx = np.linalg.cholesky(Pc, upper = True)
    Ry = np.linalg.cholesky(Po, upper = False)
    
    U,S,VT = np.linalg.svd(Rx @ Ry)

    T = Ry @ VT.T @ np.diag(1 / np.sqrt(S))
    Tinv = np.diag(1 / np.sqrt(S)) @ U.T @ Rx

    error = np.linalg.norm(T.T @ Pc @ T - Tinv @ Po @ Tinv.T, 2)

    return T, Tinv, S, error

def get_minimal_realization(A, B, C, tol = 1e-9):
    """
    Minimal realization from Kalman decomposition.
    z = Tx
    """

    n = A.shape[0]

    # Get controllability matrix
    Ctrb = B # Instantiate matrix with first column
    AiB = B # = A @ ... @ A @ B
    for _ in range(n - 1):
        AiB = A @ AiB # Get
        Ctrb = np.hstack((Ctrb, AiB)) # Add column

    # Get observability matrix
    Obsv = C # Controllability matrix
    CAi = C #C @ A @ ... @ A
    for _ in range(n - 1):
        CAi = CAi @ A
        Obsv = np.vstack((Obsv, CAi))

    # Find a basis for the subspace that is both unobservable and controllable
    _, S, Vt = np.linalg.svd(Obsv @ Ctrb)
    rank = (S > tol).sum()
    RObar = Ctrb @ (Vt[rank:,:].T)

    # Find a basis for the orthogonal complement within the controllable subspace
    # of the previous subspace
    _, S, Vt = np.linalg.svd(RObar.T @ Ctrb)
    rank = (S > tol).sum()
    RO = Ctrb @ (Vt[rank:,:].T)

    # Get an orthonormal basis for this new subspace
    U, S, _ = np.linalg.svd(RO)
    rank = (S > tol).sum()
    return U[:,:rank]


def main():
    A = np.array([[1,1],[0,1]])
    B = np.array([[1],[0]]).reshape([2,1])
    C = np.array([[0, 1]]).reshape([1,2])
    print(get_minimal_realization(A, B, C))

    

if __name__ == "__main__":
    main()