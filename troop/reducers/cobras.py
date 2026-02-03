import numpy as np
from scipy.integrate import solve_ivp


class Cobra:
    def __init__(
        self,
        n,
        r,
        d,
        m,
        f,
        g,
        Df,
        Dg,
        u_fn,
        sg=1,
        x0=None,
        T=1.0,
        N=10,
        L=10,
        atol=1.0e-7,
        rtol=1.0e-4,
        solver_method="RK45",
    ):
        self.n = n
        self.r = r
        self.d = d
        self.m = m

        self.f = f
        self.g = g
        self.Df = Df
        self.Dg = Dg

        self.Phi = None
        self.Psi = None

        self.atol = atol
        self.rtol = rtol
        self.solver_method = solver_method

        self.u_fn = u_fn
        self.sg = sg

        if x0 is None:
            x0 = np.zeros((self.n,))
        self.x0 = x0

        self.T = T
        self.N = N
        self.L = L
        self.times = np.linspace(0, T, N + L + 1)

        self.x = None
        self.X = None
        self.simulate_fom()

        self.Y = None
        self.sample_gradients()

        self.factorized_covariance_balancing()
        # self.z_fn = None
        # self.simulate_rom()

    def fom_dynamics(self, t, x):
        return self.f(x, self.u_fn(t))

    def simulate_fom(self):
        """
        Docstring for simulate_fom

        Simulate the full-order model (FOM) dynamics.
        """
        sol = solve_ivp(
            self.fom_dynamics,
            y0=self.x0,
            t_span=(self.times[0], self.times[-1]),
            method=self.solver_method,
            atol=self.atol,
            rtol=self.rtol,
            dense_output=True,
        )

        self.x = sol.sol
        self.X = np.array([sol.sol(t) for t in self.times]).T

    def rom_dynamics(self, t, z):
        return self.Psi.T @ self.f(self.Phi @ z, self.u_fn(t))

    def simulate_rom(self):
        """
        Docstring for simulate_rom

        Simulate the reduced-order model (ROM) dynamics.
        """
        sol = solve_ivp(
            self.rom_dynamics,
            y0=self.Psi.T @ self.x0,
            t_span=(self.times[0], self.times[-1]),
            dense_output=True,
            method=self.solver_method,
            atol=self.atol,
            rtol=self.rtol,
        )
        self.z_fn = sol.sol

    def yhat_fn(self, t):
        return self.g(self.Phi @ self.z_fn(t))

    def adjoint_dynamics(self, t, g_eta, tf):
        return self.Df(self.x(tf - t), self.u_fn(tf - t)).T @ g_eta

    def solve_adjoint(self, eta, tf):
        sol = solve_ivp(
            lambda t, g_eta: self.adjoint_dynamics(t, g_eta, tf),
            y0=self.Dg(self.x(tf)).T @ eta,
            t_span=(0, tf),
            dense_output=True,
            method=self.solver_method,
            atol=self.atol,
            rtol=self.rtol,
        )
        return sol.sol

    def sample_gradients(self):
        cols = []
        for _ in range(self.sg):
            eta = (self.L + 1) * np.random.normal(
                size=(self.m,)
            )  # Sample from N(0, (L+1)I)
            t = np.random.choice(self.N + 1)
            tau = np.random.choice(self.L + 1)
            tf = t + tau
            g_eta = self.solve_adjoint(eta, tf)
            tau_min = max(0, tf - self.N)
            tau_max = min(self.L, tf)
            Yi = (
                np.array([g_eta(k) for k in range(tau_min, tau_max + 1)]).T
                / (1 + tau_max - tau_min) ** 0.5
            )
            cols.append(Yi)
        self.Y = np.hstack(cols) / np.sqrt(self.sg)

    def factorized_covariance_balancing(self):
        U, S, Vt = np.linalg.svd(self.Y.T @ self.X, full_matrices=False)
        V = Vt.T
        Phi = self.X @ V[:, : self.r] @ np.diag(np.sqrt(1 / S[: self.r]))
        Psi = self.Y @ U[:, : self.r] @ np.diag(np.sqrt(1 / S[: self.r]))
        self.Phi = Phi
        self.Psi = Psi
