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

    import time

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


    def f(x, u):
        x1 = x[0]
        x2 = x[1]
        theta = x[2]
        return np.array([np.cos(theta), np.sin(theta), u]).reshape((3,))


    def g(x):
        x1 = x[0]
        x2 = x[1]
        return np.array([x2]).reshape((1,))


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
        return np.array([0, 1, 0]).reshape((1, 3))
    return L, T, d, dfdx, dgdx, f, g, m, n, r


@app.cell
def _(L, T, d, dfdx, dgdx, f, g, m, n, np, r, troop):
    u_fn = lambda t: 1  # np.array([1])
    x0 = np.array([0.0, 0.0, 0.0])

    trooper = troop.Trooper(
        n, r, d, m, f, g, dfdx, dgdx, u_fn=u_fn, x0=x0, T=T, L=L
    )

    trooper.conjugate_gradient(
        max_iters=100, max_step_search_iters=40, initial_step_size=0.01
    )
    return (trooper,)


@app.cell
def _(plt, trooper):
    Ys = [trooper.yhat_fn(t)[0] for t in trooper.times]
    plt.plot(trooper.times, Ys)
    plt.scatter(trooper.times, trooper.Y)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
