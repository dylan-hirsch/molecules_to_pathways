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
    import scipy as sp

    from dynamics.repressiloggleator import RepressiloggleatorReducedModel
    from reducers.troop import Trooper

    from scipy import integrate as ode
    from scipy.optimize import root_scalar
    from scipy.interpolate import make_interp_spline as spline

    from IPython.display import HTML
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    import random

    random.seed(2026)
    np.random.seed(2026)

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 15}
    plt.rc("font", **font)
    return (
        RepressiloggleatorReducedModel,
        Trooper,
        hj,
        jnp,
        np,
        ode,
        plt,
        spline,
    )


@app.cell
def _(np):
    n = 5  # FOM size
    r = 3  # ROM size

    d = 1  # input size
    m = 1  # output size

    L = 11  # number of time steps
    T = 10  # final time

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

    Ks = [K1, K2, K3, K4, K5]
    ns = [n1, n2, n3, n4, n5]


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
    return Ks, L, T, d, dfdx, dgdx, f, g, m, n, ns, r


@app.cell
def _(f, np):
    def dx(t, x, grad_valuess, grid, model, times):
        x = np.array(x).reshape([len(x)])
        z = model.Derivative_Projection @ x

        i = np.argmin(np.abs(times - t))
        grad_value = grid.interpolate(grad_valuess[i], state=z)
        u = model.optimal_control(z, t, grad_value)[0]

        return f(x, u)
    return (dx,)


@app.cell
def _(L, T, Trooper, d, dfdx, dgdx, f, g, m, n, np, r):
    u_fn0 = lambda t: 1

    x0 = np.array([0.2, 0.2, 0.6, 1.0, 0.0])  # initial state

    trooper = Trooper(n, r, d, m, f, g, dfdx, dgdx, u_fn=u_fn0, x0=x0, T=T, L=L)
    return trooper, u_fn0, x0


@app.cell
def _(
    Ks,
    RepressiloggleatorReducedModel,
    T,
    dx,
    f,
    hj,
    jnp,
    np,
    ns,
    ode,
    r,
    spline,
    trooper,
    u_fn0,
    x0,
):
    ts = np.linspace(-T, 0, 100)
    sol0 = ode.solve_ivp(
        lambda t, x: f(x, u_fn0(x)),
        [-T, 0],
        x0,
        max_step=0.1,
        dense_output=True,
        atol=trooper.atol,
        rtol=trooper.rtol,
    )

    records = [(sol0.sol, [u_fn0(t) for t in ts])]


    for iter in range(10):
        print("Iteration: " + str(iter))

        trooper.conjugate_gradient(verbose=True, max_iters=100)

        Phi = trooper.Phi
        Psi = trooper.Psi
        model = RepressiloggleatorReducedModel(
            Ks=Ks, ns=ns, rank=r, Phi=Phi, Psi=Psi
        )

        zmax = np.ones((r,))

        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(
                -2.5 * 1 * np.abs(zmax),
                2.5 * 1 * np.abs(zmax),
            ),
            [51 for _ in range(r)],
            periodic_dims=None,
        )

        xgrid = jnp.einsum("ij,...j -> ...i", Phi, grid.states)
        l = xgrid[..., 3] - xgrid[..., 4]

        times = np.linspace(0.0, -T, 51)
        solver_settings = hj.SolverSettings.with_accuracy("very_high")
        V = hj.solve(solver_settings, model, grid, times, l)

        grad_valuess = [grid.grad_values(V[i, ...]) for i in range(len(times))]

        sol = ode.solve_ivp(
            lambda t, x: dx(t, x, grad_valuess, grid, model, times),
            [-T, 0.0],
            x0,
            max_step=0.1,
            dense_output=True,
            atol=trooper.atol,
            rtol=trooper.rtol,
        )

        us = []
        for t in ts:
            x = np.array(sol.sol(t)).reshape(
                5,
            )
            z = model.Derivative_Projection @ x
            j = np.argmin(np.abs(times - t))
            grad_value = grid.interpolate(grad_valuess[j], state=z)
            u = model.optimal_control(z, t, grad_value)[0]
            us.append(u)
        us = np.array(us)

        u_fn = spline(ts + T, us)
        trooper.set_u_fn(lambda t: u_fn(t))
        records.append((sol.sol, us))
    return records, ts


@app.cell
def _(plt, records, ts):
    fig, axs = plt.subplots(11, 2, figsize=(10, 40))
    for record, ax1, ax2 in zip(records, axs[:, 0], axs[:, 1]):
        state_function = record[0]
        inputs = record[1]
        labels = [r"$A$", r"$B$", r"$C$", r"$X$", r"$Y$"]
        for state_index, label in zip(range(5), labels):
            ax1.plot(ts, [state_function(t)[state_index] for t in ts], label=label)
        ax1.set_ylim([-0.1, 1.1])
        ax1.legend()
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel(r"$\mathbf{x}(t)$")
        ax2.plot(ts, inputs)
        ax2.set_ylim([-1.1, 1.1])
        ax2.set_xlabel(r"$t$")
        ax2.set_ylabel(r"$u(t)$")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
