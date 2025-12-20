import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import troop_vscode as troop
    return np, plt, troop


@app.cell
def _():
    n = 3  # FOM size
    r = 2  # ROM size

    d = 1  # input size
    m = 1  # output size

    L = 100  # number of time steps
    T = 10  # final time
    return L, T, d, m, n, r


@app.cell
def _(np):
    # USER: instantiate nonlinear dynamics, f and g, and corresponding Jacobians here

    """
    B = np.random.normal(size=(n, d))
    C = np.random.normal(size=(m, n))


    def random_hurwitz(n):
        # Random eigenvalues with negative real part
        real_parts = -np.random.uniform(0.1, 2.0, size=n)
        eigs = real_parts

        # Random invertible matrix
        V = np.random.randn(n, n)
        while np.linalg.cond(V) > 1e8:
            V = np.random.randn(n, n)

        A = V @ np.diag(eigs) @ np.linalg.inv(V)
        return np.real(A)


    A = random_hurwitz(n)


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
    return dfdx, dgdx, f, g


@app.cell
def _(d, dfdx, dgdx, f, g, m, n, r, troop):
    object = troop.troop(n, r, d, m, f, g, dfdx, dgdx)
    return (object,)


@app.cell
def _(L, T, n, np, object):
    x0 = np.zeros((n,))
    U = lambda t: 2 * np.sin(t)  # np.sin(t) * np.ones((1,))
    print(object.get_mse(U, T, L, x0))
    for _ in range(100):
        object.gradient_step(U, T, L, x0, alpha=0.1)
        print(object.get_mse(U, T, L, x0))
    return U, x0


@app.cell
def _(L, T, U, np, object, x0):
    Y = object.simulate_FOM(U, T, L, x0)
    _, Yhat = object.simulate_ROM(U, T, L, x0)
    Y = np.array(Y)
    Yhat = np.array(Yhat)
    return Y, Yhat


@app.cell
def _(Y, Yhat, plt):
    plt.plot(Y)
    plt.plot(Yhat)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
