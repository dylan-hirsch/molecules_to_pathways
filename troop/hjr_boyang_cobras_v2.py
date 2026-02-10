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

    from dynamics.dubin_5d import Dubin5dReducedModel
    from reducers.cobras import Cobra

    from scipy import integrate as ode
    from scipy.optimize import root_scalar
    from scipy.interpolate import make_interp_spline as spline
    from scipy.interpolate import interp1d as interp

    from IPython.display import HTML
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    import random
    import pickle as pkl

    random.seed(2026)
    np.random.seed(2026)

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 15}
    plt.rc("font", **font)
    return Cobra, Dubin5dReducedModel, hj, interp, jnp, np, ode, plt


@app.cell
def _(np):
    n = 5  # FOM size
    r = 3  # ROM size

    d = 2  # input size
    m = 1  # output size

    L = 10
    N = 10
    T = 2  # final time


    def f(x, u):
        theta = x[2]
        v = x[3]
        omega = x[4]

        a = u[0]
        alpha = u[1]

        return np.array(
            [v * np.cos(theta), v * np.sin(theta), omega, a, alpha]
        ).reshape((n,))


    def h(x):
        x4 = x[3]
        x5 = x[4]
        return np.array(
            [
                np.sqrt(abs(x4 - 1) ** 2 + abs(x5 - 1) ** 2),
            ]
        ).reshape((m,))


    def dfdx(x, u):
        theta = x[2]
        v = x[3]
        return np.array(
            [
                [0, 0, -v * np.sin(theta), np.cos(theta), 0],
                [0, 0, +v * np.cos(theta), np.sin(theta), 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ).reshape((n, n))


    def dhdx(x):
        x4 = x[3]
        x5 = x[4]
        return np.array(
            [
                [
                    0,
                    0,
                    0,
                    (x4 - 1) / np.sqrt(abs(x4 - 1) ** 2 + abs(x5 - 1) ** 2),
                    (x5 - 1) / np.sqrt(abs(x4 - 1) ** 2 + abs(x5 - 1) ** 2),
                ],
            ]
        ).reshape((m, n))
    return L, N, T, d, dfdx, dhdx, f, h, m, n


@app.cell
def _(f, jnp, np):
    def dx(t, x, grad_valuess, grid, model, times):
        x = np.array(x).reshape([len(x)])
        z = model.Derivative_Projection @ x

        i = np.argmin(np.abs(times - t))
        grad_value = grid.interpolate(grad_valuess[i], state=z)
        u = model.optimal_control(z, t, grad_value)

        return f(x, u)


    def value_postprocessor(t, v, l, g):
        return jnp.minimum(v, l)
        # return jnp.maximum(jnp.minimum(v, l), g)
    return (dx,)


@app.cell
def _(d, np):
    def u_fn0(t):
        return np.array([np.sin(t), np.cos(t)]).reshape((d,))


    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # initial state
    return u_fn0, x0


@app.cell
def _(
    Cobra,
    Dubin5dReducedModel,
    L,
    N,
    T,
    d,
    dfdx,
    dhdx,
    dx,
    f,
    h,
    hj,
    interp,
    jnp,
    m,
    n,
    np,
    ode,
    u_fn0,
    x0,
):
    ts = np.linspace(-T, 0, 100)
    sol0 = ode.solve_ivp(
        lambda t, x: f(x, u_fn0(t)),
        [-T, 0],
        x0,
        max_step=0.1,
        dense_output=True,
    )

    records = [(sol0.sol, [u_fn0(t) for t in ts], np.nan, np.nan)]


    u_fn = u_fn0
    for iter in range(1):
        print("Iteration: " + str(iter))

        cobra = Cobra(
            n,
            1,
            d,
            m,
            f,
            h,
            dfdx,
            dhdx,
            u_fn=u_fn,
            x0=x0,
            T=T,
            N=N,
            L=L,
        )

        Wx = cobra.X @ cobra.X.T
        Wg = cobra.Y @ cobra.Y.T
        P = cobra.Phi @ cobra.Psi.T

        metric = np.linalg.trace(Wg @ (np.eye(n) - P) @ Wx @ (np.eye(n) - P.T))

        Phi, _ = np.linalg.qr(cobra.Phi)
        Psi, _ = np.linalg.qr(cobra.Psi)
        r0 = min(Phi.shape[1], Psi.shape[1])

        PhiTPsi = Phi.T @ Psi
        U, S, V = np.linalg.svd(PhiTPsi)
        r0 = sum(S > 1e-10)
        print(r0)

        Phi = Phi[:, :r0]
        Psi = Psi[:, :r0]

        model = Dubin5dReducedModel(rank=r0, Phi=Phi, Psi=Psi)

        zmax = np.ones((r0,))

        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(
                -5 * np.abs(zmax),
                5 * np.abs(zmax),
            ),
            [51 for _ in range(r0)],
            periodic_dims=None,
        )

        xgrid = jnp.einsum("ij,...j -> ...i", Phi, grid.states)
        l = (
            jnp.sqrt(
                jnp.abs(xgrid[..., 3] - 1.0) ** 2 + abs(xgrid[..., 4] - 1.0) ** 2
            )
            - 1.0
        )
        # g = (
        #    -jnp.sqrt(
        #        jnp.abs(xgrid[..., 0] - 0.5) ** 2 * 2
        #        + jnp.abs(xgrid[..., 1] - 0.0) ** 2 / 2
        #    )
        #    + 1.0
        # )

        times = np.linspace(0.0, -T, 51)
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high",
            # value_postprocessor=lambda t, v: value_postprocessor(
            #    t, v, l, 0
            # ),  # value_postprocessor(t, v, l, g),
        )
        V = hj.solve(solver_settings, model, grid, times, l)

        grad_valuess = [grid.grad_values(V[i, ...]) for i in range(len(times))]

        sol = ode.solve_ivp(
            lambda t, x: dx(t, x, grad_valuess, grid, model, times),
            [-T, 0.0],
            x0,
            max_step=0.1,
            dense_output=True,
        )

        us = []
        for t in ts:
            x = np.array(sol.sol(t)).reshape(
                n,
            )
            z = model.Derivative_Projection @ x
            j = np.argmin(np.abs(times - t))
            grad_value = grid.interpolate(grad_valuess[j], state=z)
            u = model.optimal_control(z, t, grad_value)
            us.append(u)
        us = np.array(us)

        u_fn = interp(
            ts + T,
            us,
            kind="linear",
            bounds_error=False,
            fill_value=(us[0], us[-1]),
            axis=0,
        )

        records.append((sol.sol, us, metric, r0))
    return records, ts


@app.cell
def _(T, np, plt, records, ts):
    ts0 = np.linspace(-T, 0, 100)

    fig, axs = plt.subplots(26, 3, figsize=(10, 100))
    for record, ax1, ax2, ax3 in zip(records, axs[:, 0], axs[:, 1], axs[:, 2]):
        state_function = record[0]
        inputs = record[1]
        rec_metric = record[2]
        rec_r0 = record[3]
        labels = [r"$x$", r"$y$", r"$\theta$", r"$v$", r"$\omega$"]
        for state_index, label in zip(range(5), labels):
            ax1.plot(
                ts0, [state_function(t)[state_index] for t in ts0], label=label
            )
        ax1.hlines(0.00, min(ts0), max(ts0), linestyle="--", colors="k")
        ax1.hlines(1.25, min(ts0), max(ts0), linestyle="--", colors="k")
        ax1.set_ylim([-5, 5])
        ax1.legend(fontsize=8)
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel(r"$\mathbf{x}(t)$")
        ax2.plot(ts, inputs)
        ax2.set_ylim([-1.1, 1.1])
        ax2.set_xlabel(r"$t$")
        ax2.set_ylabel(r"$u(t)$")
        ax2.set_title(rec_r0)
        ax3.plot(
            [state_function(t)[3] for t in ts0],
            [state_function(t)[4] for t in ts0],
        )
        ax3.set_xlim([-2.1, 2.1])
        ax3.set_ylim([-2.1, 2.1])
        ax3.scatter([0.5, 1.25], [0, 0])
        ax3.set_title(f"{rec_metric:.{2}g}")
        ax3.axvline(1, linestyle="--", linewidth=2, color="k")
        ax3.axhline(1, linestyle="--", linewidth=2, color="k")
    plt.tight_layout()
    plt.savefig("/Users/dylanhirsch/Desktop/dubins_car_cobra.png")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
