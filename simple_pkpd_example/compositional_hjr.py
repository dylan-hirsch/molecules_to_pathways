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
    return dynamics, hj, jnp, mo, np, ode, plt


@app.cell
def _(dynamics, hj, jnp):
    gamma = 0.05
    delta = 0.5
    model = dynamics.model(gamma=gamma, delta=delta)

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([0, 0, 0]), jnp.array([1, 1, 5])),
        (26, 26, 26),
        periodic_dims=None,
    )
    return delta, gamma, grid, model


@app.cell
def _(mo):
    mo.md(r"""## Get value function for staying above therapeutic threshold""")
    return


@app.cell
def _(grid, hj, jnp, np):
    g0 = np.maximum.reduce(
        [
            np.array([0.5]) - grid.states[..., 0],
            -0.8 + grid.states[..., 1],
            -0.8 + grid.states[..., 2],
        ]
    )


    def value_postprocessor_inner(t, v, g=g0):
        return jnp.maximum(v, g)


    solver_settings0 = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor_inner
    )
    return g0, solver_settings0


@app.cell
def _(g0, grid, hj, model, np, solver_settings0):
    points_per_day = 4
    t0 = -7
    N = points_per_day * abs(t0)
    times = np.linspace(0, t0, N + 1)
    V0 = hj.solve(solver_settings0, model, grid, times, g0)
    return N, V0, points_per_day, t0, times


@app.cell
def _(mo):
    mo.md(r"""## Get value function for compositional task""")
    return


@app.cell
def _(V0, grid, hj, jnp, model, np, times):
    l = V0[-1, ...]
    g = np.maximum(-0.8 + grid.states[..., 1], -0.8 + grid.states[..., 2])


    def value_postprocessor(t, v, l=l, g=g):
        return jnp.maximum(jnp.minimum(v, l), g)


    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )

    V = hj.solve(solver_settings, model, grid, times, np.maximum(l, g))
    return (V,)


@app.cell
def _(N, V, V0, delta, gamma, grid, model, np, ode, points_per_day):
    def dx(x, u):
        x1, x2, x3 = x

        dx1 = 0.5 * x2**4 / (x2**4 + 0.5**4) - gamma * x1
        dx2 = u - 2 * x2
        dx3 = x2 - delta * x3

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
        if x0[0] > .5:
            break
    j = 0
    while i + j < 2 * N - 1:
        grad_value = grid.interpolate(grid.grad_values(V0[-1, ...]), state=x0)
        u = model.optimal_control(x0, 0, grad_value)[0]

        sol = ode.solve_ivp(
            lambda t, x: dx(x, u),
            [(j + i + 1) / points_per_day, (j + i + 2) / points_per_day],
            x0,
            max_step=0.01,
        )
        x0 = sol.y[:, -1]
        T.append(sol.t)
        Y.append(sol.y)
        us.append(u)
        j += 1


    t_sol = np.concatenate([t if i == 0 else t[1:] for i, t in enumerate(T)])
    y_sol = np.concatenate(
        [y if i == 0 else y[:, 1:] for i, y in enumerate(Y)], axis=1
    )
    return t_sol, us, y_sol


@app.cell
def _(np, plt, points_per_day, t0, t_sol, us, y_sol):
    fig, axs = plt.subplots(3, 1, figsize=(6, 6))

    colors = ["blue", "magenta", "purple"]

    ax = axs[0]
    ax.plot(t_sol, y_sol[0, :], color=colors[0])
    ax.set_ylabel("[X]\n(Normalized)")
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim(0, 2 * abs(t0))
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
    ax.set_xlim(0, 2 * abs(t0))
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
    ax.scatter(np.array(range(len(us))) / points_per_day, us)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim(0, 2 * abs(t0))
    ax.set_ylabel("Dose \n(Normalized)")

    plt.tight_layout()
    plt.savefig("/Users/dylanhirsch/Desktop/long_treatment.svg", transparent=True)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
