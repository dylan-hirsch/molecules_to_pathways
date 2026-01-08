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

    random.seed(25)
    np.random.seed(25)
    return np, plt, troop


@app.cell
def _(np):
    n = 3  # FOM size
    r = 2  # ROM size

    d = 1  # input size
    m = 1  # output size

    L = 11  # number of time steps
    T = 5  # final time

    """
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
    """


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
def _(np):
    def inner_product(X1, Y1, X2, Y2):
        return np.linalg.trace(X1.T @ X2) + np.linalg.trace(Y1.T @ Y2)


    def bisection(trooper, Ux, Sx, Vx, Uy, Sy, Vy, X, Y, c1=0.4, c2=0.6):
        alpha_trooper = trooper.copy()

        J0 = trooper.get_cost()
        dJ0 = trooper.get_cost_derivative(0, Ux, Sx, Vx, Uy, Sy, Vy, X, Y)

        Phi_alpha, Psi_alpha = trooper.compute_translation_along_geodesic(
            0.01, Ux, Sx, Vx, Uy, Sy, Vy
        )
        alpha_trooper.set_Phi_Psi(Phi_alpha, Psi_alpha)

        J = alpha_trooper.get_cost()
        print((J - J0) / 0.01)
        print(dJ0)

        lb = 0
        alpha = 1
        ub = np.inf

        while ub - lb > 1e-3:
            Phi_alpha, Psi_alpha = trooper.compute_translation_along_geodesic(
                alpha, Ux, Sx, Vx, Uy, Sy, Vy
            )
            alpha_trooper.set_Phi_Psi(Phi_alpha, Psi_alpha)

            J = alpha_trooper.get_cost()
            if J > J0 + c1 * alpha * dJ0:
                ub = alpha
                alpha = 0.5 * (lb + ub)
            else:
                dJ = alpha_trooper.get_cost_derivative(
                    alpha, Ux, Sx, Vx, Uy, Sy, Vy, X, Y
                )
                if dJ < c2 * dJ0:
                    lb = alpha
                    if ub == np.inf:
                        alpha = 2 * lb
                    else:
                        alpha = 0.5 * (lb + ub)
                else:
                    break
        return alpha, alpha_trooper
    return bisection, inner_product


@app.cell
def _(L, T, bisection, d, dfdx, dgdx, f, g, inner_product, m, n, np, r, troop):
    U = lambda t: 1  # np.array([1])
    x0 = np.array([0.0, 0.0, 0.0])
    trooper = troop.troop(n, r, d, m, f, g, dfdx, dgdx, U=U, x0=x0, T=T, L=L)
    gradJ_Phi, gradJ_Psi = trooper.compute_gradient()

    X = -gradJ_Phi
    Y = -gradJ_Psi
    stopping_criterion = inner_product(gradJ_Phi, gradJ_Psi, gradJ_Phi, gradJ_Psi)

    # while stopping_criterion > 1e-8:
    for k in range(100):
        Ux, Sx, VxT = np.linalg.svd(X, full_matrices=False)
        Uy, Sy, VyT = np.linalg.svd(Y, full_matrices=False)
        Vx = VxT.T
        Vy = VyT.T

        alpha, alpha_trooper = bisection(trooper, Ux, Sx, Vx, Uy, Sy, Vy, X, Y)
        break

        X_tilde, Y_tilde = trooper.compute_parallel_translation(
            alpha, Ux, Sx, Vx, Uy, Sy, Vy, X, Y
        )
        gradJ_Phi, gradJ_Psi = trooper.compute_gradient()
        denominator_2 = inner_product(-gradJ_Phi, -gradJ_Psi, X, Y)

        trooper = alpha_trooper
        parity = trooper.standardize_representatives()
        X_tilde[:, 0] = parity * X_tilde[:, 0]
        gradJ_Phi, gradJ_Psi = trooper.compute_gradient()

        numerator = inner_product(-gradJ_Phi, -gradJ_Psi, -gradJ_Phi, -gradJ_Psi)
        denominator_1 = inner_product(-gradJ_Phi, -gradJ_Psi, X_tilde, Y_tilde)

        beta = numerator / (denominator_1 + denominator_2)
        X = -gradJ_Phi + beta * X_tilde
        Y = -gradJ_Psi + beta * Y_tilde

        print(trooper.get_cost())
    return (trooper,)


@app.cell
def _(L, T, np, plt, trooper):
    ts = np.linspace(0, T, L)
    ys = [trooper.Y(t) for t in ts]
    yhats = [trooper.Yhat(t) for t in ts]

    plt.plot(ts, ys)
    plt.plot(ts, yhats)
    return


if __name__ == "__main__":
    app.run()
