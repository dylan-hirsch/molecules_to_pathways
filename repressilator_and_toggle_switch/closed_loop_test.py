import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pickle

    from IPython.display import HTML
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import dynamics

    import hj_reachability as hj
    import pandas as pd

    from scipy import integrate as ode
    from scipy import signal

    import random

    random.seed(123)

    #!pip install latex

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 15}
    plt.rc("font", **font)


    COLORS = ["blue", "orange", "green", "red", "purple"]
    LABELS = ["A", "B", "C", "X", "Y"]
    return COLORS, LABELS, dynamics, hj, jnp, np, ode, pickle, plt


@app.cell
def _(pickle):
    with open(
        "/Users/dylanhirsch/Research/model_reduction/V_decoy.pkl", "rb"
    ) as file:
        V = pickle.load(file)
    return (V,)


@app.cell
def _(dynamics, hj, jnp, np):
    model = dynamics.reduced_model(rank=3)

    t0 = -25
    times = np.linspace(0.0, t0, 100)

    T = dynamics.T[:, 0:3]
    Tinv = dynamics.Tinv[0:3, :]
    zmax = np.array([0.25, 0.25, 1])  # Tinv @ np.array([1.,1.,1.,1.,1.])
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.minimum(-zmax, zmax), jnp.maximum(-zmax, zmax)),
        (100, 100, 100),
        periodic_dims=None,
    )
    return Tinv, grid, model, t0, times


@app.cell
def _(Tinv, np):
    def mm(x, K, n):
        return 1.0 / (1.0 + (abs(x) / K) ** n)


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

        i = np.argmin(np.abs(times - t))

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]

        x = np.array(x).reshape([len(x)])
        z = Tinv @ x

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
def _(V, dx, grid, model, np, ode, t0, times):
    grad_valuess = [grid.grad_values(V[i, ...]) for i in range(len(times))]
    x0 = np.array([0.1, 0.2, 0.15, 1.0, 0.0])
    sol = ode.solve_ivp(
        lambda t, x: dx(t, x, grad_valuess, grid, model, times),
        [t0, 0],
        x0,
        max_step=0.1,
    )
    return grad_valuess, sol


@app.cell
def _(
    COLORS,
    LABELS,
    Tinv,
    grad_valuess,
    grid,
    model,
    np,
    plt,
    sol,
    t0,
    times,
):
    fig, axs = plt.subplots(2, 1, figsize=(7, 10))

    ##
    ax = axs[0]

    for _ in range(5):
        ax.plot(sol.t + abs(t0), sol.y[_, :], color=COLORS[_], label=LABELS[_])
    ax.set_xlabel(r"$t$", fontsize=20)
    ax.set_ylabel("Protein Concentration (Normalized)", fontsize=15)
    ax.set_title(r"FOM under $u = \pi_r(x,t)$", fontsize=20)
    ax.legend(fontsize=15)

    ##
    ax = axs[1]

    us = []
    for i in range(len(sol.t)):
        t = sol.t[i]
        x = np.array(sol.y[:, i]).reshape([5])
        z = Tinv @ x
        j = np.argmin(np.abs(times - t))
        grad_value = grid.interpolate(grad_valuess[j], state=z)
        u = model.optimal_control(z, t, grad_value)[0]
        us.append(u)

    ax.plot(sol.t + abs(t0), us)
    ax.set_xlabel(r"$t$", fontsize=20)
    ax.set_ylabel("Control Action", fontsize=15)
    ax.set_title(r"$u = \pi_r(x,t)$", fontsize=20)

    plt.tight_layout()
    plt.savefig("/Users/dylanhirsch/Desktop/closed_loop.svg", bbox_inches="tight")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
