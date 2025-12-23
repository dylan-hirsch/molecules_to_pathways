import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import jax
    import jax.numpy as jnp
    import hj_reachability as hj
    import numpy as np

    from IPython.display import HTML
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    import dynamics
    import troop

    from scipy import integrate as ode
    from scipy.optimize import root_scalar
    from scipy.interpolate import make_interp_spline as spline

    import random

    random.seed(123)
    np.random.seed(123)

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 15}
    plt.rc("font", **font)
    return dynamics, hj, jnp, np, ode, plt, spline, troop


@app.cell
def _(mm, np):
    def dx(t, x, grad_valuess, grid, model, times):
        K1 = 0.25
        K2 = 0.25
        K3 = 0.25
        K4 = 0.25
        K5 = 0.25

        n1 = 4.0
        n2 = 4.0
        n3 = 4.0
        n4 = 2.0
        n5 = 2.0

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]

        x = np.array(x).reshape([len(x)])
        z = model.Phi.T @ (x - model.x_star)

        i = np.argmin(np.abs(times - t))
        grad_value = grid.interpolate(grad_valuess[i], state=z)
        u = model.optimal_control(z, t, grad_value)[0]

        dx1 = mm(x3, K3, n3) - x1
        dx2 = mm(x1, K1, n1) - x2
        dx3 = mm(x2, K2, n2) - x3
        dx4 = u * mm(x2, K2, n2) + mm(x5, K5, n5) - x4
        dx5 = u * mm(x3, K3, n3) + mm(x4, K4, n4) - x5

        dx = np.array([dx1, dx2, dx3, dx4, dx5])

        return dx
    return (dx,)


@app.cell
def _(np):
    K1 = 0.25
    K2 = 0.25
    K3 = 0.25
    K4 = 0.25
    K5 = 0.25

    n1 = 4.0
    n2 = 4.0
    n3 = 4.0
    n4 = 2.0
    n5 = 2.0


    def mm(x, K, n):
        return 1.0 / (1.0 + (x / K) ** n)


    def dmmdx(x, K, n):
        return -n / K * (x / K) ** (n - 1) / (1.0 + (x / K) ** n) ** 2


    def f(x, u):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]

        return np.array(
            [
                mm(x3, K3, n3) - x1,
                mm(x1, K1, n1) - x2,
                mm(x2, K2, n2) - x3,
                mm(x5, K5, n5) - x4 + u * mm(x2, K2, n2),
                mm(x4, K4, n4) - x5 + u * mm(x3, K3, n3),
            ]
        ).reshape((5,))


    def g(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        return np.array([x4 - x5]).reshape(
            1,
        )


    def dfdx(x, u):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        return np.array(
            [
                [-1, 0, dmmdx(x3, K3, n3), 0, 0],
                [dmmdx(x1, K1, n1), -1, 0, 0, 0],
                [0, dmmdx(x2, K2, n2), -1, 0, 0],
                [0, u * dmmdx(x2, K2, n2), 0, -1, dmmdx(x5, K5, n5)],
                [0, 0, u * dmmdx(x3, K3, n3), dmmdx(x4, K4, n4), -1],
            ]
        ).reshape((5, 5))


    def dgdx(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        return np.array([0, 0, 0, 1, -1]).reshape((1, 5))
    return dfdx, dgdx, f, g, mm


@app.cell
def _():
    rank = 3
    Ks = [0.25, 0.25, 0.25, 0.25, 0.25]
    ns = [4.0, 4.0, 4.0, 2.0, 2.0]
    return Ks, ns, rank


@app.cell
def _(dfdx, dgdx, f, g, np, rank, troop):
    x0 = np.array([0.2, 0.2, 0.6, 1.0, 0.0])

    L = 100  # number of time steps
    t0 = -10
    T = abs(t0)  # final time
    object = troop.troop(5, rank, 1, 1, f, g, dfdx, dgdx)
    return L, T, object, t0, x0


@app.cell
def _(
    Ks,
    L,
    T,
    dx,
    dynamics,
    hj,
    jnp,
    np,
    ns,
    object,
    ode,
    rank,
    spline,
    t0,
    x0,
):
    MSEs = []
    for _ in range(50):
        Phi = object.Phi
        Psi = object.Psi
        model = dynamics.reduced_model(
            rank=rank, Ks=Ks, ns=ns, Phi=Phi, Psi=Psi, x_star=x0
        )
        zmax = np.ones((rank,))

        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(
                -1.5 * np.sqrt(rank) * np.abs(zmax),
                1.5 * np.sqrt(rank) * np.abs(zmax),
            ),
            [51 for _ in range(rank)],
            periodic_dims=None,
        )

        xgrid = jnp.einsum("ij,...j -> ...i", Phi, grid.states) - x0
        l = xgrid[..., 3] - xgrid[..., 4]

        times = np.linspace(0.0, t0, 51)
        solver_settings = hj.SolverSettings.with_accuracy("very_high")
        V = hj.solve(solver_settings, model, grid, times, l)

        grad_valuess = [grid.grad_values(V[i, ...]) for i in range(len(times))]

        sol = ode.solve_ivp(
            lambda t, x: dx(t, x, grad_valuess, grid, model, times),
            [t0, 0.0],
            x0,
            max_step=0.1,
        )

        us = []
        for i in range(len(sol.t)):
            x = np.array(sol.y[:, i]).reshape((5,))
            z = model.Phi.T @ (x - model.x_star)
            t = sol.t[i]
            j = np.argmin(np.abs(times - t))
            grad_value = grid.interpolate(grad_valuess[j], state=z)
            val = grid.interpolate(V[j], state=z)
            u = model.optimal_control(z, t, grad_value)[0]
            us.append(u)
        us = np.array(us)

        U = spline(sol.t + abs(t0), us)
        print(object.get_mse(U, T, L, x0))

        for _ in range(10):
            object.gradient_step(U, T, L, x0, alpha=0.01)
            MSE = object.get_mse(U, T, L, x0)
            MSEs.append(MSE)
            print(MSE)
    return MSEs, sol, us


@app.cell
def _(plt, sol, t0):
    for k in range(5):
        plt.plot(sol.t + abs(t0), sol.y[k, :])
    plt.ylabel("Concentration")
    plt.xlabel(r"$t$")
    plt.title(r"FOM trajectory with controller from learned ROM")
    plt.show()
    return


@app.cell
def _(plt, sol, t0, us):
    plt.plot(sol.t + abs(t0), us)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u(t)$")
    plt.title(r"FOM trajectory with controller from learned ROM")
    plt.show()
    return


@app.cell
def _(MSEs, plt):
    plt.plot(MSEs)
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title(r"$y[u]$ vs. $\hat{y}[u]$")
    return


if __name__ == "__main__":
    app.run()
