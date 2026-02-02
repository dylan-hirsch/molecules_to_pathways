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
        x0=None,
        T=1.0,
        N=10,
        L=10,
        gamma=0.001,
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

        if x0 is None:
            x0 = np.zeros((self.n,))
        self.x0 = x0

        self.T = T
        self.N = N
        self.times = np.linspace(0, T, N + L + 1)

        self.gamma = gamma

        self.Y = None
        self.simulate_fom()

        self.z_fn = None
        self.simulate_rom()

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
            t_eval=self.times,
            method=self.solver_method,
            atol=self.atol,
            rtol=self.rtol,
        )

        self.Y = [
            self.g(
                sol.y[:, i].reshape(
                    self.n,
                )
            )
            for i in range(len(self.times))
        ]

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

    def factorized_covariance_balancing(self, X, Y):
        U, S, Vt = np.linalg.svd(Y.T @ X, full_matrices=False)
        V = Vt.T
        Phi = X @ V[:, : self.r] @ np.diag(np.sqrt(1 / S[: self.r]))
        Psi = Y @ U[:, : self.r] @ np.diag(np.sqrt(1 / S[: self.r]))
        self.Phi = Phi
        self.Psi = Psi
