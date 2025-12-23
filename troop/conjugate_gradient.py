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
    return np, troop


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
def _(d, dfdx, dgdx, f, g, m, n, r, troop):
    trooper = troop.troop(n, r, d, m, f, g, dfdx, dgdx)
    return (trooper,)


@app.cell
def _(np):
    def inner_product(X1, Y1, X2, Y2):
        trace_Xs = np.linalg.trace(X1.T @ X2)
        trace_Ys = np.linalg.trace(Y1.T @ Y2)
        return trace_Xs + trace_Ys
    return (inner_product,)


@app.cell
def _(L, T, inner_product, np, trooper):
    U = lambda t: np.sin(2 * np.pi * t)
    x0 = np.zeros((3,))
    trooper.get_cost(U, T, L, x0, gamma=0.01)
    gradJ_Phi, gradJ_Psi = trooper.compute_gradient(U, T, L, x0, gamma=0.01)
    X0 = gradJ_Phi
    Y0 = gradJ_Psi
    stopping_criterion = inner_product(gradJ_Phi, gradJ_Psi, gradJ_Phi, gradJ_Psi)

    while stopping_criterion > 1e-8:
        # = np.linalg.svd(X0)
        stopping_criterion = stopping_criterion / 2
    return X0, Y0


@app.cell
def _(X0, Y0, np):
    UX, SX, VXT = np.linalg.svd(X0, full_matrices=False)
    UY, SY, VYT = np.linalg.svd(Y0, full_matrices=False)
    return


@app.cell
def _(np):
    def geodesic(alpha, M, U, S, V):
        return M @ (V * np.cos(alpha * S)) @ V.T + (U * np.sin(alpha @ S)) @ V.T


    def bisection(object, U, S, V):
        pass


    def parallel_translation(U, S, V):
        pass
    return


@app.cell
def _(np):
    a = np.array((1, 4, 9))
    M = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    M * a[:, None]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
