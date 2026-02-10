import numpy as np
import warnings
from scipy.linalg import solve_continuous_lyapunov as solve_lyap
from scipy.linalg import solve_continuous_are as solve_are
from numpy.linalg import matrix_rank


def get_petrov_galerkin_projection(P, Q, r):

    # Do cholesky factorizations
    R = np.linalg.cholesky(P, upper = True)
    L = np.linalg.cholesky(Q, upper = True)
    
    # Perform SVD
    U,S,VT = np.linalg.svd(L.T @ R)
    V = VT.T

    # Truncate
    Ur = U[:,:r]
    Sr = S[:r]
    Vr = V[:,:r]

    # Form trial matrix Phi and test matrix Psi
    Phi = R @ Vr @ np.diag(1 / np.sqrt(Sr)) # typically called T_r
    Psi = L @ Ur @ np.diag(1 / np.sqrt(Sr)) # typically called T_r^{-1}.T

    return Phi, Psi, S

def get_bt_transform(A, B, C, r):
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
    P = solve_lyap(A, -B @ B.T) # controllability gramian
    Q = solve_lyap(A.T, -C.T @ C) # observability gramian

    return get_petrov_galerkin_projection(P, Q, r)

def get_lqg_bt_transform(A, B, C, r):

    """
    See "A New Set of invariants for Linear Systems -- Application to Reduced Order Compensator Design" by Jonckheere and Silverman (IEEE TAC, 1983).
    This is the original paper on LQG balancing.
    It is assumed (A,B) is controllable and (A,C) is observable.
    """

    # Compute solutions to algebraic ricatti equations
    cntrl_R = np.eye(B.shape[1]) # the R-matrix for the controllability LQR objective
    cntrl_Q = C.T @ C # the Q-matrix for the controllability LQR objective
    P = solve_are(A, B, cntrl_Q, cntrl_R) # Also called P in original paper

    obsv_R = np.eye(C.shape[0]) # the R-matrix for the observability LQR
    obsv_Q = B @ B.T # The Q-matrix for the observability LQR
    Q = solve_are(A.T, C.T, obsv_Q, obsv_R) # Called Pi in original paper; note that this Q has nothing to do with the Q-matrix

    return get_petrov_galerkin_projection(P,Q,r)

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
