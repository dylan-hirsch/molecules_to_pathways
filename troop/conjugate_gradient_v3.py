import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import random

    import dynamics
    import troop
    import copy

    random.seed(25)
    np.random.seed(25)
    return copy, np, troop


@app.cell
def _(np):
    n = 3  # FOM size
    r = 2  # ROM size

    d = 1  # input size
    m = 1  # output size

    L = 11  # number of time steps
    T = 5  # final time

    A = np.array([[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -5.0]])
    B = np.array([[1.0], [1.0], [1.0]])
    C = np.array([[1.0, 1.0, 1.0]])


    def f(x, u):
        return A @ x + B @ u


    def g(x):
        return C @ x


    def dfdx(x, u):
        return A


    def dgdx(x):
        return C
    return L, T, d, dfdx, dgdx, f, g, m, n, r


@app.cell
def _(copy, np):
    def inner_product(X1, Y1, X2, Y2):
        trace_Xs = np.linalg.trace(X1.T @ X2)
        trace_Ys = np.linalg.trace(Y1.T @ Y2)
        return trace_Xs + trace_Ys


    def geodesic(alpha, Q, U, S, V):
        return Q @ (V * np.cos(alpha * S)) @ V.T + (U * np.sin(alpha * S)) @ V.T


    def d_geodesic(alpha, Q, U, S, V):
        return (
            -Q @ (V * (np.sin(alpha * S) * S)) @ V.T
            + (U * (np.cos(alpha * S) * S)) @ V.T
        )


    def dJdAlpha(
        trooper, temp_trooper, alpha, U, T, L, x0, UX, SX, VX, UY, SY, VY
    ):
        dJdPhi, dJdPsi = temp_trooper.compute_gradient()
        dPhidAlpha = d_geodesic(alpha, trooper.Phi, UX, SX, VX)
        dPsidAlpha = d_geodesic(alpha, trooper.Psi, UY, SY, VY)
        dJdAlpha = np.sum(dJdPhi * dPhidAlpha + dJdPsi * dPsidAlpha)

        return dJdAlpha


    def bisection(trooper, UX, SX, VX, UY, SY, VY, U, T, L, x0, c1=0.4, c2=0.6):
        J_left = trooper.get_cost()

        dJ_left = dJdAlpha(
            trooper, trooper, 0, U, T, L, x0, UX, SX, VX, UY, SY, VY
        )

        alpha = 0
        t = 1
        beta = np.inf

        temp_trooper = copy.deepcopy(trooper)
        while beta - alpha >= 1e-3:
            Phi_left = geodesic(alpha + t, trooper.Phi, UX, SX, VX)
            Psi_left = geodesic(alpha + t, trooper.Psi, UY, SY, VY)

            temp_trooper.Phi = Phi_left
            temp_trooper.Psi = Psi_left
            temp_trooper.standardize_representatives()
            J_right = temp_trooper.get_cost()
            dJ_right = dJdAlpha(
                trooper,
                temp_trooper,
                alpha + t,
                U,
                T,
                L,
                x0,
                UX,
                SX,
                VX,
                UY,
                SY,
                VY,
            )

            if J_right > J_left + c1 * t * dJ_left:
                beta = t
                t = 0.5 * (alpha + beta)
            elif dJ_right < c2 * dJ_left:
                alpha = t
                if beta == np.inf:
                    t = 2 * alpha
                else:
                    t = 0.5 * (alpha + beta)
            else:
                break

        return alpha


    def parallel_translation(alpha, Q, P, U, S, V):
        dQ = d_geodesic(alpha, Q, U, S, V)
        return dQ + P - U @ U.T @ P
    return bisection, geodesic, inner_product, parallel_translation


@app.cell
def _(
    L,
    T,
    bisection,
    d,
    dfdx,
    dgdx,
    f,
    g,
    geodesic,
    inner_product,
    m,
    n,
    np,
    parallel_translation,
    r,
    troop,
):
    U = lambda t: np.array([1])
    x0 = np.array([0.0, 0.0, 0.0])
    trooper = troop.troop(n, r, d, m, f, g, dfdx, dgdx, U=U, x0=x0, T=T, L=L)
    gradJ_Phi, gradJ_Psi = trooper.compute_gradient()

    X = gradJ_Phi.copy()
    Y = gradJ_Psi.copy()
    stopping_criterion = inner_product(gradJ_Phi, gradJ_Psi, gradJ_Phi, gradJ_Psi)

    # while stopping_criterion > 1e-8:
    for i in range(20):
        UX, SX, VXT = np.linalg.svd(X, full_matrices=False)
        UY, SY, VYT = np.linalg.svd(Y, full_matrices=False)

        alpha = bisection(
            trooper,
            UX,
            SX,
            VXT.T,
            UY,
            SY,
            VYT.T,
            U,
            T,
            L,
            x0,
            c1=0.1,
            c2=0.9,
        )

        X_tilde = parallel_translation(alpha, trooper.Phi, X, UX, SX, VXT.T)
        Y_tilde = parallel_translation(alpha, trooper.Psi, Y, UY, SY, VYT.T)

        gradJ_Phi, gradJ_Psi = trooper.compute_gradient()
        denominator_2 = inner_product(gradJ_Phi, gradJ_Psi, X, Y)

        trooper.Phi = geodesic(alpha, trooper.Phi, UX, SX, VXT.T)
        trooper.Psi = geodesic(alpha, trooper.Psi, UY, SY, VYT.T)
        parity = trooper.standardize_representatives()
        X_tilde[:, 0] = parity * X_tilde[:, 0]
        gradJ_Phi, gradJ_Psi = trooper.compute_gradient()

        numerator = inner_product(gradJ_Phi, gradJ_Psi, gradJ_Phi, gradJ_Psi)
        denominator_1 = inner_product(gradJ_Phi, gradJ_Psi, X_tilde, Y_tilde)

        beta = numerator / (denominator_1 + denominator_2)
        X = gradJ_Phi + beta * X_tilde
        Y = gradJ_Psi + beta * Y_tilde

        # print(trooper.get_cost())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
