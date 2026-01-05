import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import dynamics
    import troop
    import copy
    return copy, np, plt, troop


@app.cell
def _(np):
    n = 3  # FOM size
    r = 2  # ROM size

    d = 1  # input size
    m = 1  # output size

    L = 100  # number of time steps
    T = 10  # final time


    def f(x, u):
        x1 = x[0]
        x2 = x[1]
        theta = x[2]
        return np.array([np.cos(theta), np.sin(theta), u]).reshape((3,))


    def g(x):
        x1 = x[0]
        x2 = x[1]
        return np.sqrt(np.array([x1**2 + x2**2])).reshape((1,))


    def dfdx(x, u):
        x1 = x[0]
        x2 = x[1]
        theta = x[2]
        return np.array(
            [[0, 0, -np.sin(theta)], [0, 0, np.cos(theta)], [0, 0, 0]]
        ).reshape((3, 3))


    def dgdx(x):
        x1 = x[0]
        x2 = x[1]
        return np.array([2 * x1, 2 * x2, 0]).reshape((1, 3))
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
        trooper,
        temp_trooper,
        alpha,
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
        gamma=0.01,
    ):
        dJdPhi, dJdPsi = temp_trooper.compute_gradient(U, T, L, x0, gamma=gamma)
        dPhidAlpha = d_geodesic(alpha, trooper.Phi, UX, SX, VX)
        dPsidAlpha = d_geodesic(alpha, trooper.Psi, UY, SY, VY)
        dJdAlpha = np.sum(dJdPhi * dPhidAlpha + dJdPsi * dPsidAlpha)

        return dJdAlpha


    def bisection(
        trooper, UX, SX, VX, UY, SY, VY, U, T, L, x0, gamma=0.01, c1=0.4, c2=0.6
    ):
        J_left = trooper.get_cost(U, T, L, x0, gamma)

        dJ_left = dJdAlpha(
            trooper, trooper, 0, U, T, L, x0, UX, SX, VX, UY, SY, VY, gamma
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
            temp_trooper.update_M()

            J_right = temp_trooper.get_cost(U, T, L, x0, gamma=gamma)
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
    trooper = troop.troop(n, r, d, m, f, g, dfdx, dgdx)

    U = lambda t: 1  # np.sin(np.pi * t)
    x0 = np.zeros((3,))
    gradJ_Phi, gradJ_Psi = trooper.compute_gradient(U, T, L, x0)
    X = gradJ_Phi
    Y = gradJ_Psi
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
            gamma=0.01,
            c1=0.1,
            c2=0.9,
        )

        X_tilde = parallel_translation(alpha, trooper.Phi, X, UX, SX, VXT.T)
        Y_tilde = parallel_translation(alpha, trooper.Psi, Y, UY, SY, VYT.T)

        gradJ_Phi, gradJ_Psi = trooper.compute_gradient(U, T, L, x0)
        denominator_2 = inner_product(gradJ_Phi, gradJ_Psi, X, Y)

        trooper.Phi = geodesic(alpha, trooper.Phi, UX, SX, VXT.T)
        trooper.Psi = geodesic(alpha, trooper.Psi, UY, SY, VYT.T)
        parity = trooper.standardize_representatives()
        trooper.update_M()
        X_tilde[:, 0] = parity * X_tilde[:, 0]
        gradJ_Phi, gradJ_Psi = trooper.compute_gradient(U, T, L, x0)

        numerator = inner_product(gradJ_Phi, gradJ_Psi, gradJ_Phi, gradJ_Psi)
        denominator_1 = inner_product(gradJ_Phi, gradJ_Psi, X_tilde, Y_tilde)

        beta = numerator / (denominator_1 + denominator_2)
        X = gradJ_Phi + beta * X_tilde
        Y = gradJ_Psi + beta * Y_tilde

        print(trooper.get_cost(U, T, L, x0))
    return U, trooper, x0


@app.cell
def _(L, T, U, np, plt, trooper, x0):
    def _():
        Y = trooper.simulate_FOM(U, T, L, x0)
        _, Yhat = trooper.simulate_ROM(U, T, L, x0)
        Y = np.array(Y)
        Yhat = np.array(Yhat)

        plt.plot(Y, label="FOM (n = 3)")
        plt.plot(Yhat, label="ROM (r = 2)")
        plt.legend()
        plt.xlabel(r"$t$")
        plt.ylabel(r"$y$")
        plt.title("Simple TROOP Example")
        plt.show()


    _()
    return


if __name__ == "__main__":
    app.run()
