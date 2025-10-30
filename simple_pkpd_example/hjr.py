import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np

    from IPython.display import HTML
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import dynamics

    import hj_reachability as hj
    from scipy import integrate as ode
    import random

    random.seed(123)
    np.random.seed(123)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 15}
    plt.rc("font", **font)
    return dynamics, hj, jnp, np, ode, plt


@app.cell
def _(dynamics, hj, jnp, np):
    model = dynamics.model()

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([0, 0, 0]), jnp.array([1, 1, 5])),
        (26, 26, 26),
        periodic_dims=None,
    )

    l = 10 * np.linalg.norm(grid.states[..., [0]] - np.array([0.53]), axis=-1) - 1
    g = np.maximum(-0.8 + grid.states[..., 1], -0.8 + grid.states[..., 2])


    def value_postprocessor(t, v, l=l, g=g):
        return jnp.maximum(jnp.minimum(v, l), g)


    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )
    return g, grid, l, model, solver_settings


@app.cell
def _(g, grid, hj, l, model, np, solver_settings):
    t0 = -8
    points_per_day = 24
    N = points_per_day * abs(t0)
    times = np.linspace(0.0, t0, N + 1)
    V = hj.solve(solver_settings, model, grid, times, np.maximum(l, g))
    return N, V, points_per_day, t0


@app.cell
def _(N, V, grid, model, np, ode, points_per_day):
    def dx(x, u):
        x1, x2, x3 = x

        dx1 = 0.5 * x2**4 / (x2**4 + 0.5**4)
        dx2 = u - 2 * x2
        dx3 = x2 - 0.3 * x3

        dx = np.array([dx1, dx2, dx3])

        return dx


    T = []
    Y = []
    us = []
    x0 = np.array([0.0, 0.0, 0.0])
    for i in range(N):
        grad_value = grid.interpolate(grid.grad_values(V[-1 - i, ...]), state=x0)
        u = model.optimal_control(x0, 0, grad_value)[0]

        sol = ode.solve_ivp(
            lambda t, x: dx(x, u),
            [i / points_per_day, (i + 1) / points_per_day],
            x0,
            max_step=0.01,
        )
        x0 = sol.y[:, -1]
        T.append(sol.t)
        Y.append(sol.y)
        us.append(u)

    t_sol = np.concatenate([t if i == 0 else t[1:] for i, t in enumerate(T)])
    y_sol = np.concatenate(
        [y if i == 0 else y[:, 1:] for i, y in enumerate(Y)], axis=1
    )
    return dx, t_sol, us, y_sol


@app.cell
def _(N, np, plt, points_per_day, t0, t_sol, us, y_sol):
    fig, axs = plt.subplots(3, 1, figsize=(6, 6))

    colors = ["blue", "magenta", "purple"]

    ax = axs[0]
    ax.plot(t_sol, y_sol[0, :], color=colors[0])
    ax.set_ylabel("[X]\n(Normalized)")
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim(0, abs(t0))
    ax.hlines(
        y=0.5,
        xmin=0,
        xmax=max(t_sol),
        color="green",
        linestyles="dashed",
        label="Therapeutic",
    )
    ax.legend(ncol=2)

    ax = axs[1]
    ax.plot(t_sol, y_sol[1, :], color=colors[1], label="Blood")
    ax.plot(t_sol, y_sol[2, :], color=colors[2], label="Tissue")
    ax.set_ylabel("[Drug]\n(Normalized)")
    ax.set_ylim([-0.05, 1.55])
    ax.set_xlim(0, abs(t0))
    ax.hlines(
        y=0.8,
        xmin=0,
        xmax=max(t_sol),
        color="red",
        linestyles="dashed",
        label="Toxic",
    )
    ax.legend(ncol=3)

    ax = axs[2]
    ax.set_xlabel("Day")
    ax.plot(np.array(range(N)) / points_per_day, us, 'black')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim(0, abs(t0))
    ax.set_ylabel("Dose \n(Normalized)")

    plt.tight_layout()
    plt.savefig("/Users/dylanhirsch/Desktop/dose.svg", transparent=True)
    plt.show()
    return


@app.cell
def _(N, dx, np, ode, plt, points_per_day, t0):
    def _():
        fig, axs = plt.subplots(3, 1, figsize=(6, 6))

        colors = ["blue", "magenta", "purple"]

        x0 = np.array([0, 0, 0])
        u = 0.64
        sol = ode.solve_ivp(
            lambda t, x: dx(x, u),
            [0, abs(t0)],
            x0,
            max_step=0.01,
        )
        t_sol = sol.t
        y_sol = sol.y

        ax = axs[0]
        ax.plot(t_sol, y_sol[0, :], color=colors[0])
        ax.set_ylabel("[X]\n(Normalized)")
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim(0, abs(t0))
        ax.hlines(
            y=0.5,
            xmin=0,
            xmax=max(t_sol),
            color="green",
            linestyles="dashed",
            label="Therapeutic",
        )
        ax.legend(ncol=2)

        ax = axs[1]
        ax.plot(t_sol, y_sol[1, :], color=colors[1], label="Blood")
        ax.plot(t_sol, y_sol[2, :], color=colors[2], label="Tissue")
        ax.set_ylabel("[Drug]\n(Normalized)")
        ax.set_ylim([-0.05, 1.55])
        ax.set_xlim(0, abs(t0))
        ax.hlines(
            y=0.8,
            xmin=0,
            xmax=max(t_sol),
            color="red",
            linestyles="dashed",
            label="Toxic",
        )
        ax.legend(ncol=3)

        ax = axs[2]
        ax.set_xlabel("Day")
        ax.plot(np.array(range(N)) / points_per_day, u * np.ones(N), 'black')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim(0, abs(t0))
        ax.set_ylabel("Dose \n(Normalized)")

        plt.tight_layout()
        plt.savefig("/Users/dylanhirsch/Desktop/bad_1.svg", transparent=True)
        plt.show()
    


    _()
    return


@app.cell
def _(N, dx, np, ode, plt, points_per_day, t0):
    def _():
        fig, axs = plt.subplots(3, 1, figsize=(6, 6))

        colors = ["blue", "magenta", "purple"]

        x0 = np.array([0, 0, 0])
        sol = ode.solve_ivp(
            lambda t, x: dx(x, 1 if t > 5.2 else 0),
            [0, abs(t0)],
            x0,
            max_step=0.01,
        )
        t_sol = sol.t
        y_sol = sol.y

        ax = axs[0]
        ax.plot(t_sol, y_sol[0, :], color=colors[0])
        ax.set_ylabel("[X]\n(Normalized)")
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim(0, abs(t0))
        ax.hlines(
            y=0.5,
            xmin=0,
            xmax=max(t_sol),
            color="green",
            linestyles="dashed",
            label="Therapeutic",
        )
        ax.legend(ncol=2)

        ax = axs[1]
        ax.plot(t_sol, y_sol[1, :], color=colors[1], label="Blood")
        ax.plot(t_sol, y_sol[2, :], color=colors[2], label="Tissue")
        ax.set_ylabel("[Drug]\n(Normalized)")
        ax.set_ylim([-0.05, 1.55])
        ax.set_xlim(0, abs(t0))
        ax.hlines(
            y=0.8,
            xmin=0,
            xmax=max(t_sol),
            color="red",
            linestyles="dashed",
            label="Toxic",
        )
        ax.legend(ncol=3)

        ax = axs[2]
        ax.set_xlabel("Day")
        ts = np.array(range(N)) / points_per_day
        ax.plot(ts, [1 if t > 5.2 else 0 for t in ts], 'black')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim(0, abs(t0))
        ax.set_ylabel("Dose \n(Normalized)")

        plt.tight_layout()
        plt.savefig("/Users/dylanhirsch/Desktop/bad_2.svg", transparent=True)
        plt.show()


    _()
    return


if __name__ == "__main__":
    app.run()
