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
    return dynamics, hj, jnp, np


@app.cell
def _(dynamics, hj, jnp, np):
    model = dynamics.reduced_model(rank=3)

    T = dynamics.T[:, 0:3]
    Tinv = dynamics.Tinv[0:3, :]
    zmax = np.array([0.25, 0.25, 1])  # Tinv @ np.array([1.,1.,1.,1.,1.])

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.minimum(-zmax, zmax), jnp.maximum(-zmax, zmax)),
        (100, 100, 100),
        periodic_dims=None,
    )

    xgrid = T @ grid.states[..., None]
    xgrid = xgrid[..., 0]
    l = xgrid[..., 3] - xgrid[..., 4]
    return grid, l, model


@app.cell
def _(grid, hj, l, model, np):
    t0 = -25
    times = np.linspace(0.0, t0, 100)
    solver_settings = hj.SolverSettings.with_accuracy("medium")
    V = hj.solve(solver_settings, model, grid, times, l)
    return (V,)


@app.cell
def _(V):
    import pickle

    with open("/Users/dylanhirsch/Research/model_reduction/V.pkl", "wb") as file:
        pickle.dump(V, file)
    return


if __name__ == "__main__":
    app.run()
