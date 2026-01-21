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

    from dynamics.linear_system import LinearReducedModel
    from reducers.troop import Trooper

    from scipy import integrate as ode
    from scipy.optimize import root_scalar
    from scipy.interpolate import make_interp_spline as spline

    from IPython.display import HTML
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    import random

    random.seed(2025)
    np.random.seed(2025)

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 15}
    plt.rc("font", **font)
    return LinearReducedModel, Trooper, hj, jnp, np, ode, plt, sp, spline


@app.cell
def _(np, sp):
    n = 3  # FOM size
    r = 2  # ROM size

    d = 1  # input size
    m = 1  # output size

    L = 11  # number of time steps
    T = 3  # final time


    A = np.random.normal(size=(n, n))
    U, Q = sp.linalg.schur(A)
    np.fill_diagonal(U, -np.abs(np.diag(U)))
    A = Q @ U @ Q.T
    B = np.random.normal(size=(n, d))
    C = np.random.normal(size=(m, n))

    # A = np.eye(3)
    # B = np.array([1, 0, 0]).reshape((3, 1))
    # C = np.array([1, 0, 0]).reshape((1, 3))


    def f(x, u):
        xdot = A @ x + B @ u
        return xdot.reshape((n,))


    def g(x):
        y = C @ x
        return y.reshape((1,))


    def dfdx(x, u):
        return A.reshape((3, 3))


    def dgdx(x):
        return C.reshape((1, 3))
    return A, B, C, L, T, d, dfdx, dgdx, f, g, m, n, r


@app.cell
def _(d, f, np):
    def dx(t, x, grad_valuess, grid, model, times):
        x = np.array(x).reshape([len(x)])
        z = model.Derivative_Projection @ x

        i = np.argmin(np.abs(times - t))
        grad_value = grid.interpolate(grad_valuess[i], state=z)
        u = np.array(model.optimal_control(z, t, grad_value)).reshape((d,))

        return f(x, u)
    return (dx,)


@app.cell
def _(L, T, Trooper, d, dfdx, dgdx, f, g, m, n, np, r):
    u_fn0 = lambda t: np.array([1])

    x0 = np.array([0.0, 0.0, 0.0])  # initial state

    trooper = Trooper(n, r, d, m, f, g, dfdx, dgdx, u_fn=u_fn0, x0=x0, T=T, L=L)
    return trooper, u_fn0, x0


@app.cell
def _(
    A,
    B,
    C,
    LinearReducedModel,
    T,
    dx,
    f,
    hj,
    jnp,
    m,
    np,
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


    for iter in range(5):
        print("Iteration: " + str(iter))

        trooper.conjugate_gradient(verbose=True, max_iters=100)

        Phi = trooper.Phi
        Psi = trooper.Psi
        model = LinearReducedModel(A=A, Bu=B, Bd=None, rank=r, Phi=Phi, Psi=Psi)

        zmax = np.ones((r,))

        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(
                -1.5 * 15 * np.abs(zmax),
                1.5 * 15 * np.abs(zmax),
            ),
            [257 for _ in range(r)],
            periodic_dims=None,
        )

        xgrid = jnp.einsum("ij,...j -> ...i", Phi, grid.states)
        l = jnp.einsum("ij,...j -> ...i", C, xgrid)[..., 0]

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
                3,
            )
            z = model.Derivative_Projection @ x
            j = np.argmin(np.abs(times - t))
            grad_value = grid.interpolate(grad_valuess[j], state=z)
            u = model.optimal_control(z, t, grad_value)[0]
            us.append(u)
        us = np.array(us)

        u_fn = spline(ts + T, us)
        trooper.set_u_fn(
            lambda t: np.array(u_fn(t)).reshape(
                m,
            )
        )

        records.append((sol.sol, us))
    return records, times, ts


@app.cell
def _(plt, records, ts):
    fig, axs = plt.subplots(6, 2, figsize=(10, 20))
    for record, ax1, ax2 in zip(records, axs[:, 0], axs[:, 1]):
        state_function = record[0]
        inputs = record[1]
        labels = [r"$x$", r"$y$", r"$\theta$"]
        for state_index, label in zip(range(3), labels):
            ax1.plot(ts, [state_function(t)[state_index] for t in ts], label=label)
        ax1.set_ylim([-10, 10])
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


@app.cell
def _(
    A,
    B,
    C,
    LinearReducedModel,
    T,
    dx,
    hj,
    jnp,
    n,
    np,
    ode,
    times,
    trooper,
    ts,
    x0,
):
    def _():
        model0 = LinearReducedModel(
            A=A, Bu=B, Bd=None, rank=n, Phi=np.eye(3), Psi=np.eye(3)
        )

        zmax0 = np.ones((n,))

        grid0 = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(
                -5 * np.abs(zmax0),
                5 * np.abs(zmax0),
            ),
            [51 for _ in range(n)],
            periodic_dims=None,
        )

        xgrid0 = jnp.einsum("ij,...j -> ...i", np.eye(3), grid0.states)
        l0 = jnp.einsum("ij,...j -> ...i", C, xgrid0)[..., 0]

        solver_settings0 = hj.SolverSettings.with_accuracy("very_high")
        V0 = hj.solve(solver_settings0, model0, grid0, times, l0)

        grad_valuess0 = [grid0.grad_values(V0[i, ...]) for i in range(len(times))]

        sol0 = ode.solve_ivp(
            lambda t, x: dx(t, x, grad_valuess0, grid0, model0, times),
            [-T, 0.0],
            x0,
            max_step=0.1,
            dense_output=True,
            atol=trooper.atol,
            rtol=trooper.rtol,
        )

        us0 = []
        for t in ts:
            xdum = np.array(sol0.sol(t)).reshape(
                3,
            )
            zdum = model0.Derivative_Projection @ xdum
            k = np.argmin(np.abs(times - t))
            grad_value0 = grid0.interpolate(grad_valuess0[k], state=zdum)
            u0 = model0.optimal_control(zdum, t, grad_value0)[0]
            us0.append(u0)
        us0 = np.array(us0)
        return us0


    us0 = _()
    return (us0,)


@app.cell
def _(plt, ts, us0):
    plt.plot(ts, us0)
    plt.ylim([-1.1, 1.1])
    plt.show()
    return


if __name__ == "__main__":
    app.run()
