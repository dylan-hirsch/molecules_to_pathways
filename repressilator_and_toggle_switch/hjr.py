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

    from balanced_truncation import get_lqg_bt_transform

    import hj_reachability as hj

    from scipy import integrate as ode
    from scipy.optimize import root_scalar
    import control as ct

    import random

    random.seed(123)
    np.random.seed(123)

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 15}
    plt.rc("font", **font)
    return dynamics, get_lqg_bt_transform, hj, np, root_scalar


@app.cell
def _():
    rank = 4
    Ks = [0.25, 0.25, 0.25, 0.25, 0.25]
    ns = [4.0, 4.0, 4.0, 2.0, 2.0]
    return Ks, ns, rank


@app.cell
def _(Ks, get_lqg_bt_transform, np, ns, root_scalar):
    def mm(x, K, n):
        return 1 / (1 + (x / K) ** n)


    def mm_derivative(x, K, n):
        return -n * x ** (n - 1) / K**n / (1 + (x / K) ** n) ** 2


    def represillator_root(x, K, n):
        return x ** (n + 1) + K**n * x - K**n


    def toggle_root(x, K, n):
        return K**n * (1 + (x / K) ** n) ** n * (1 - x) - x


    x_represillator = root_scalar(
        lambda x: represillator_root(x, Ks[0], ns[0]), x0=1.0
    ).root

    x_toggle = root_scalar(lambda x: toggle_root(x, Ks[3], ns[3]), x0=1.0).root

    a = mm_derivative(x_represillator, Ks[0], ns[0])
    b = mm_derivative(x_toggle, Ks[3], ns[3])
    c = mm(x_represillator, Ks[0], ns[0])

    A11 = np.array([[-1, 0, a], [a, -1, 0], [0, a, -1]])
    A22 = np.real(np.array([[-1, b], [b, -1]]))

    A = np.block([[A11, np.zeros((3, 2))], [np.zeros((2, 3)), A22]])

    B = np.array(
        [
            [0, 0, 0, c, c],
            [0.1, 0, 0, 0, 0],
            [0, 0.1, 0, 0, 0],
            [0, 0, 0.1, 0, 0],
            [0, 0, 0, 0.1, 0],
            [0, 0, 0, 0, 0.1],
        ]
    ).T

    C = np.array(
        [
            [0, 0, 0, -1 / 2, 1 / 2],
            [0, 1.0, -1.0, 0, 0],
            [0, 0, 0.1, 0, 0],
            [0.1, 0, 0, 0, 0],
            [0, 0, 0, 0.1, 0],
            [0, 0, 0, 0, 0.1],
        ]
    )

    T0, Tinv0, S0, error = get_lqg_bt_transform(A, B, C)
    T, _, _ = np.linalg.svd(T0)
    Tinv = T.T
    return T, Tinv


@app.cell
def _(Ks, T, Tinv, dynamics, hj, np, ns, rank):
    model = dynamics.reduced_model(rank=rank, Ks=Ks, ns=ns, T=T, Tinv=Tinv)


    Tr = T[:, 0:rank]
    Tinvr = Tinv[0:rank, :]
    zmax = np.ones((rank,))

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(-np.abs(zmax), np.abs(zmax)),
        (21, 21, 21, 21),
        periodic_dims=None,
    )

    xgrid = Tr @ grid.states[..., None]
    xgrid = xgrid[..., 0]
    l = xgrid[..., 3] - xgrid[..., 4]
    return grid, l, model


@app.cell
def _(grid, hj, l, model, np):
    t0 = -10
    times = np.linspace(0.0, t0, 100)
    solver_settings = hj.SolverSettings.with_accuracy("medium")
    V = hj.solve(solver_settings, model, grid, times, l)
    return V, times


@app.cell
def _(V, grid, model, times):
    import pickle

    with open("/Users/dylanhirsch/Research/model_reduction/V.pkl", "wb") as file:
        pickle.dump((V, model, grid, times), file)
    return


if __name__ == "__main__":
    app.run()
