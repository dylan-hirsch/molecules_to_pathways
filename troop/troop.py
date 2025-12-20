import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.integrate import quad_vec
    from scipy.interpolate import make_interp_spline as spline
    return mo, np, quad_vec, solve_ivp, spline


@app.cell
def _():
    n = 10  # FOM size
    r = 5  # ROM size

    d = 1  # input size
    m = 1  # output size

    L = 100  # number of time steps
    T = 1  # final time
    return L, T, d, m, n, r


@app.cell
def _(d, m, n, np):
    # USER: instantiate nonlinear dynamics, f and g, and corresponding Jacobians here

    A = np.random.normal(size=(n, n))
    B = np.random.normal(size=(n, d))
    C = np.random.normal(size=(m, n))


    def f(x, u):
        return A @ x + B @ u


    def g(x):
        return C @ x


    def dfdx(x, u):
        return A


    def dgdx(x):
        return C
    return dfdx, dgdx, f, g


@app.cell
def _(mo):
    mo.md(r"""
    ## Implementation of Algorithm 4.1
    """)
    return


@app.cell
def _(L, T, f, g, n, np, r, solve_ivp):
    # Inputs to algorithm
    def U(t):
        return np.sin(10 * t) * np.ones((1,))


    def standard_representation(Phi0, Psi0):
        # We map Phi0 and Psi0 to members of their equivalence class Phi and Psi which satisfy
        # Phi' @ Phi = I_r
        # Psi' @ Psi = I_r
        # det(Psi' @ Phi) > 0
        Phi, _ = np.linalg.qr(Phi0)
        Psi, _ = np.linalg.qr(Psi0)
        if np.linalg.det(Psi.T @ Phi) < 0:
            Psi = -Psi
        return Phi, Psi


    Phi, Psi = standard_representation(
        np.random.normal(size=(n, r)), np.random.normal(size=(n, r))
    )


    def FOM_output(times, x0, U):
        sol = solve_ivp(
            lambda t, x: f(x, U(t)),
            y0=x0,
            t_span=(times[0], times[-1]),
            t_eval=times,
        )
        return [g(sol.y[:, i]) for i in range(len(times))]


    times = np.linspace(0, T, L)
    x0 = np.zeros((n,))
    Y = FOM_output(times, x0, U)
    return Phi, Psi, U, Y, standard_representation, times, x0


@app.cell
def _(dfdx, dgdx, f, m, n, np, r):
    def F_adjoint(z, u, Phi, Psi, M):
        F = M @ Psi.T @ dfdx(Phi @ z, u) @ Phi
        return F.T


    def S_adjoint_Phi(v, z, u, Phi, Psi, M):
        x = Phi @ z
        f_tilde = M @ Psi.T @ f(x, u)
        df_tilde_dz = M @ Psi.T @ dfdx(x, u) @ Phi

        return dfdx(x, u).T @ Psi @ M.T @ v.reshape((r, 1)) @ z.reshape(
            (1, r)
        ) - Psi @ M.T @ v.reshape((r, 1)) @ f_tilde.reshape((1, r))


    def S_adjoint_Psi(v, z, u, Phi, Psi, M):
        x = Phi @ z
        f_tilde = M @ Psi.T @ f(x, u)
        df_tilde_dz = M @ Psi.T @ dfdx(x, u) @ Phi

        return (f(x, u) - Phi @ f_tilde).reshape(n, 1) @ (v.reshape((1, r)) @ M)


    def H_adjoint(z, Phi, Psi, M):
        H = dgdx(Phi @ z) @ Phi
        return H.T


    def T_adjoint_Phi(w, z, u, Phi, Psi, M):
        return (dgdx(Phi @ z).T @ w.reshape((m, 1))) @ z.reshape((1, r))


    def T_adjoint_Psi(w, z, u, Phi, Psi, M):
        return np.zeros((n, r))


    def grad_z_adjoint_Phi(v, z, Phi, Psi, M):
        return -Psi @ M.T @ v.reshape((r, 1)) @ z.reshape((1, r))


    def grad_z_adjoint_Psi(x0, v, z, Phi, Psi, M):
        return (x0 - Phi @ z).reshape((n, 1)) @ v.reshape((1, r)) @ M


    def interpolate(t, t0, t1, z0, z1):
        return ((t1 - t) * z0 + (t - t0) * z1) / (t1 - t0)
    return (
        F_adjoint,
        H_adjoint,
        S_adjoint_Phi,
        S_adjoint_Psi,
        T_adjoint_Phi,
        T_adjoint_Psi,
        grad_z_adjoint_Phi,
        grad_z_adjoint_Psi,
    )


@app.cell
def _(
    F_adjoint,
    H_adjoint,
    T_adjoint_Phi,
    T_adjoint_Psi,
    f,
    g,
    m,
    r,
    solve_ivp,
    spline,
):
    def ROM_trajectory(times, x0, U, Phi, Psi, M):
        sol = solve_ivp(
            lambda t, z: M @ (Psi.T @ f(Phi @ z, U(t))),
            y0=M @ (Psi.T @ x0),
            t_span=(times[0], times[-1]),
        )

        Z = spline(sol.t, sol.y.T)
        Yhat = [g(Phi @ Z(time)) for time in times]

        return Z, Yhat


    def init_grad(times, U, Y, Yhat, Z, Phi, Psi, M):
        error = (Yhat[-1] - Y[-1]).reshape(m, 1)
        z = Z(times[-1])
        u = U(times[-1])
        gradJ_Phi = T_adjoint_Phi(error, z, u, Phi, Psi, M)
        gradJ_Psi = T_adjoint_Psi(error, z, u, Phi, Psi, M)

        return gradJ_Phi, gradJ_Psi


    def init_dual(times, Y, Yhat, Z, Phi, Psi, M):
        error = (Yhat[-1] - Y[-1]).reshape(m, 1)
        z = Z(times[-1])
        p = H_adjoint(z, Phi, Psi, M) @ error

        return p.reshape((r,))


    def dual_dynamics(p, z, u, Phi, Psi, M):
        dp = -F_adjoint(z, u, Phi, Psi, M).T @ p
        return dp
    return ROM_trajectory, dual_dynamics, init_dual, init_grad


@app.cell
def _(
    H_adjoint,
    L,
    ROM_trajectory,
    S_adjoint_Phi,
    S_adjoint_Psi,
    T_adjoint_Phi,
    T_adjoint_Psi,
    dual_dynamics,
    grad_z_adjoint_Phi,
    grad_z_adjoint_Psi,
    init_dual,
    init_grad,
    np,
    quad_vec,
    solve_ivp,
    spline,
):
    def troop(times, x0, Phi, Psi, U, Y, gamma=0.01):
        M = np.linalg.inv(Psi.T @ Phi)

        # Assemble and simulate ...
        Z, Yhat = ROM_trajectory(times, x0, U, Phi, Psi, M)

        # Initialize the gradient ...
        gradJ_Phi, gradJ_Psi = init_grad(times, U, Y, Yhat, Z, Phi, Psi, M)

        # Compute adjoint variable at final time ... (we use p in place of lambda)
        p = init_dual(times, Y, Yhat, Z, Phi, Psi, M)

        # For l in ...
        for l in reversed(range(L - 1)):
            tlplus1 = times[l + 1]
            tl = times[l]

            # Solve the adjoint equation ...
            dual_sol = solve_ivp(
                lambda t, p_dummy: dual_dynamics(
                    p_dummy,
                    Z(t),
                    U(t),
                    Phi,
                    Psi,
                    M,
                ),
                [tlplus1, tl],
                p,
                t_eval=np.linspace(tl, tlplus1, L)[::-1],
            )
            taus = dual_sol.t[::-1]
            P = spline(taus, dual_sol.y[:, ::-1].T)

            # Compute the integral component ...
            gradJ_Phi = (
                gradJ_Phi
                + quad_vec(
                    lambda t: S_adjoint_Phi(P(t), Z(t), U(t), Phi, Psi, M),
                    tl,
                    tlplus1,
                )[0]
            )
            gradJ_Psi = (
                gradJ_Psi
                + quad_vec(
                    lambda t: S_adjoint_Psi(P(t), Z(t), U(t), Phi, Psi, M),
                    tl,
                    tlplus1,
                )[0]
            )

            # Add lth element of the sum ...
            error = Yhat[l] - Y[l]
            gradJ_Phi = gradJ_Phi + T_adjoint_Phi(error, Z(tl), U(tl), Phi, Psi, M)
            gradJ_Phi = gradJ_Phi + T_adjoint_Psi(error, Z(tl), U(tl), Phi, Psi, M)

            # Add "jump" to adjoint ...
            p = p + H_adjoint(Z(tl), Phi, Psi, M) @ error

        # add gradient due to initial condition
        gradJ_Phi = gradJ_Phi + grad_z_adjoint_Phi(p, Z(0), Phi, Psi, M)
        gradJ_Psi = gradJ_Psi + grad_z_adjoint_Psi(x0, p, Z(0), Phi, Psi, M)

        # normalize by trajectory length
        gradJ_Phi = gradJ_Phi / L
        gradJ_Psi = gradJ_Psi / L

        # Add regularization
        gradJ_Phi = gradJ_Phi + gamma * 2 * (Phi - Psi @ M.T)
        gradJ_Psi = gradJ_Psi + gamma * 2 * (Psi - Phi @ M)

        return gradJ_Phi, gradJ_Psi
    return (troop,)


@app.cell
def _(Phi, Psi, U, Y, np, standard_representation, times, troop, x0):
    def _(Phi, Psi):
        alpha = 0.01
        for i in range(100):
            gradJ_Phi, gradJ_Psi = troop(times, x0, Phi, Psi, U, Y)
            print(
                np.linalg.norm(gradJ_Phi, "fro"), np.linalg.norm(gradJ_Psi, "fro")
            )
            Phi -= alpha * gradJ_Phi
            Psi -= alpha * gradJ_Psi
            Phi, Psi = standard_representation(Phi, Psi)
        return Phi, Psi


    Phi_fin, Psi_fin = _(Phi, Psi)
    return


@app.cell
def _(tos):
    tos
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
