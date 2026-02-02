import time

import numpy as np
from scipy.integrate import quad_vec, solve_ivp


class Trooper:
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
        Phi=None,
        Psi=None,
        u_fn=None,
        x0=None,
        A=None,
        Y=None,
        z_fn=None,
        T=1.0,
        L=101,
        gamma=0.001,
        atol=1.0e-7,
        rtol=1.0e-4,
        solver_method="RK45",
    ):
        """
        Initialize the Trooper class with the given parameters.

        Parameters:
        n (int): Dimension of the full-order model.
        r (int): Dimension of the reduced-order model.
        d (int): Dimension of the control input.
        m (int): Dimension of the output.
        f (function): Function representing the full-order model dynamics.
        g (function): Function representing the output mapping.
        Df (function): Function representing the Jacobian of f.
        Dg (function): Function representing the Jacobian of g.
        Phi (np.ndarray): Initial basis matrix for the state space.
        Psi (np.ndarray): Initial basis matrix for the test space.
        U (function): Control input function of time.
        x0 (np.ndarray): Initial condition for the full-order model.
        A (np.ndarray): Psi.T @ Phi inverse matrix if available (to avoid recomputing).
        Y (list): Full-order model trajectory samples.
        Yhat (function): Reduced-order model trajectory function.
        Z (function): Reduced-order model state function.
        T (float): Final time for the simulation.
        L (int): Number of time samples.
        gamma (float): Regularization parameter.
        atol (float): Absolute tolerance for the ODE solver.
        rtol (float): Relative tolerance for the ODE solver.
        solver_method (str): Method for the ODE solver.
        """

        self.n = n
        self.r = r
        self.d = d
        self.m = m

        self.f = f
        self.g = g
        self.Df = Df
        self.Dg = Dg

        self.Phi = Phi
        self.Psi = Psi
        self.A = A
        self.standardize_representatives()

        self.atol = atol
        self.rtol = rtol
        self.solver_method = solver_method

        if u_fn is None:

            def u_fn(_):
                return np.zeros((self.d,))

        self.u_fn = u_fn

        if x0 is None:
            x0 = np.zeros((self.n,))
        self.x0 = x0

        self.times = np.linspace(0, T, L)
        self.gamma = gamma

        if Y is None:
            self.Y = None
            self.simulate_fom()
        else:
            self.Y = Y.copy()

        if z_fn is None:
            self.z_fn = None
            self.simulate_rom()
        else:
            self.z_fn = z_fn

        self.gradJ_Phi = None
        self.gradJ_Psi = None
        self.parity = None

    def standardize_representatives(self):
        """
        Docstring for standardize_representatives

        We map the Trooper's current Phi and Psi to new Phi and Psi,
        which are in the equivalence class of the current variables,
        but which also satisfy
        # Phi' @ Phi = I_r
        # Psi' @ Psi = I_r
        # det(Psi' @ Phi) > 0
        """

        if self.Phi is None:
            self.Phi = np.random.normal(size=(self.n, self.r))

        PhiTPhi = self.Phi.T @ self.Phi

        if np.linalg.norm(PhiTPhi - np.eye(self.r)) > 1e-9:
            self.Phi, _ = np.linalg.qr(self.Phi)

        if self.Psi is None:
            self.Psi = np.random.normal(size=(self.n, self.r))

        PsiTPsi = self.Psi.T @ self.Psi
        if np.linalg.norm(PsiTPsi - np.eye(self.r)) > 1e-9:
            self.Psi, _ = np.linalg.qr(self.Psi)

        PsiTPhi = self.Psi.T @ self.Phi
        sign, magnitude = np.linalg.slogdet(PsiTPhi)
        if magnitude < -12:
            raise ValueError("Psi.T @ Phi is close to singular.")
        elif sign < 0:
            self.parity = -1
            self.Psi[:, 0] = -self.Psi[:, 0]
        else:
            self.parity = 1

        if self.A is None or np.linalg.norm(self.A @ PsiTPhi - np.eye(self.r)) > 1e-9:
            self.A = np.linalg.inv(PsiTPhi)

    ### Simulations of FOM and ROM

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

    def simulate_rom(self):
        """
        Docstring for simulate_rom

        Simulate the reduced-order model (ROM) dynamics.
        """
        sol = solve_ivp(
            self.rom_dynamics,
            y0=self.A @ (self.Psi.T @ self.x0),
            t_span=(self.times[0], self.times[-1]),
            dense_output=True,
            method=self.solver_method,
            atol=self.atol,
            rtol=self.rtol,
        )
        self.z_fn = sol.sol

    def fom_dynamics(self, t, x):
        return self.f(x, self.u_fn(t))

    def rom_dynamics(self, t, z):
        return self.A @ (self.Psi.T @ self.f(self.Phi @ z, self.u_fn(t)))

    def yhat_fn(self, t):
        return self.g(self.Phi @ self.z_fn(t))

    ## Setters
    def set_phi_psi(self, Phi, Psi):
        """
        Docstring for set_phi_psi

        Set new values for Phi and Psi,
        then standardize them and resimulate the ROM.
        """
        self.Phi = Phi
        self.Psi = Psi
        self.standardize_representatives()
        self.simulate_rom()

    def set_u_fn(self, u_fn):
        """
        Docstring for set_u_fn

        Set new values for u_fn,
        then resimulate the FOM.
        """
        self.u_fn = u_fn
        self.simulate_fom()
        self.simulate_rom()

    ### Adjoint calculations

    def f_adjoint(self, z, u):
        df_tilde_dz = self.A @ self.Psi.T @ self.Df(self.Phi @ z, u) @ self.Phi
        return df_tilde_dz.T

    def s_adjoint_phi(self, v, z, u):
        x = self.Phi @ z
        f_tilde = self.A @ self.Psi.T @ self.f(x, u)
        return self.Df(x, u).T @ self.Psi @ self.A.T @ v.reshape(
            (self.r, 1)
        ) @ z.reshape((1, self.r)) - self.Psi @ self.A.T @ v.reshape(
            (self.r, 1)
        ) @ f_tilde.reshape((1, self.r))

    def s_adjoint_psi(self, v, z, u):
        x = self.Phi @ z
        f_tilde = self.A @ self.Psi.T @ self.f(x, u)

        return (self.f(x, u) - self.Phi @ f_tilde).reshape(self.n, 1) @ (
            v.reshape((1, self.r)) @ self.A
        )

    def h_adjoint(self, z):
        H = self.Dg(self.Phi @ z) @ self.Phi
        return H.T

    def t_adjoint_phi(self, w, z):
        return (self.Dg(self.Phi @ z).T @ w.reshape((self.m, 1))) @ z.reshape(
            (1, self.r)
        )

    def t_adjoint_psi(self):
        return np.zeros((self.n, self.r))

    def grad_z_adjoint_phi(self, v, z):
        return -self.Psi @ self.A.T @ v.reshape((self.r, 1)) @ z.reshape((1, self.r))

    def grad_z_adjoint_psi(self, v, z):
        return (
            (self.x0 - self.Phi @ z).reshape((self.n, 1))
            @ v.reshape((1, self.r))
            @ self.A
        )

    ### Initialization of gradient and dual

    def init_grad(self):
        """
        Docstring for init_grad

        Initialize the gradient variables gradJ_Phi and gradJ_Psi
        used in the adjoint-based gradient computation."""

        t = self.times[-1]
        error = (self.yhat_fn(t) - self.Y[-1]).reshape(
            self.m,
        )
        z = self.z_fn(t)
        self.gradJ_Phi = self.t_adjoint_phi(error, z)
        self.gradJ_Psi = self.t_adjoint_psi()

    def init_dual(self):
        t = self.times[-1]
        error = (self.yhat_fn(t) - self.Y[-1]).reshape(
            self.m,
        )
        z = self.z_fn(t)
        p = self.h_adjoint(z) @ error
        return p.reshape((self.r,))

    ### Dual dynamics

    def calculate_dual_dynamics(self, p, z, u):
        dp = -self.f_adjoint(z, u) @ p
        return dp

    ### Calculate gradients

    def compute_gradient(self):
        """
        Docstring for compute_gradient

        Compute the gradients of the cost function with respect to Phi and Psi
        using adjoint-based methods.
        """
        # Initialize the gradient ...
        self.init_grad()
        # Compute adjoint variable at final time ... (we use p in place of lambda)
        p = self.init_dual()

        # For l in ...
        # for l in reversed(range(len(self.times) - 1)):
        for i in range(len(self.times) - 1)[::-1]:
            tl = self.times[i]
            tlplus1 = self.times[i + 1]

            # Solve the adjoint equation ...
            dual_sol = solve_ivp(
                lambda t, q: self.calculate_dual_dynamics(
                    q,
                    self.z_fn(t),
                    self.u_fn(t),
                ),
                [tlplus1, tl],
                p,
                dense_output=True,
                method=self.solver_method,
                atol=self.atol,
                rtol=self.rtol,
            )

            def p_fn(t, sol=dual_sol):
                return sol.sol(t)

            # Compute the integral component ...
            self.gradJ_Phi += quad_vec(
                lambda t: self.s_adjoint_phi(p_fn(t), self.z_fn(t), self.u_fn(t)),
                tl,
                tlplus1,
            )[0]
            self.gradJ_Psi += quad_vec(
                lambda t: self.s_adjoint_psi(p_fn(t), self.z_fn(t), self.u_fn(t)),
                tl,
                tlplus1,
            )[0]

            # Add lth element of the sum ...
            error = (self.yhat_fn(tl) - self.Y[i]).reshape(
                self.m,
            )
            self.gradJ_Phi += self.t_adjoint_phi(error, self.z_fn(tl))
            self.gradJ_Psi += self.t_adjoint_psi()
            # Add "jump" to adjoint ...
            p = p_fn(tl) + self.h_adjoint(self.z_fn(tl)) @ error

        # add gradient due to initial condition
        self.gradJ_Phi += self.grad_z_adjoint_phi(p, self.z_fn(0))
        self.gradJ_Psi += self.grad_z_adjoint_psi(p, self.z_fn(0))

        # normalize by trajectory length
        self.gradJ_Phi /= len(self.times)
        self.gradJ_Psi /= len(self.times)

        # Add regularization
        self.gradJ_Phi += self.gamma * 2 * (self.Phi - self.Psi @ self.A.T)
        self.gradJ_Psi += self.gamma * 2 * (self.Psi - self.Phi @ self.A)

        Phi_error = np.linalg.norm(self.Phi.T @ self.gradJ_Phi) / np.linalg.norm(
            self.gradJ_Phi
        )
        Psi_error = np.linalg.norm(self.Psi.T @ self.gradJ_Psi) / np.linalg.norm(
            self.gradJ_Phi
        )
        print("Phi error: " + str(Phi_error))
        print("Psi error: " + str(Psi_error))

        # Project onto tangent spaces
        self.project_onto_tangent_space()

        Phi_error = np.linalg.norm(self.Phi.T @ self.gradJ_Phi) / np.linalg.norm(
            self.gradJ_Phi
        )
        Psi_error = np.linalg.norm(self.Psi.T @ self.gradJ_Psi) / np.linalg.norm(
            self.gradJ_Phi
        )

    def project_onto_tangent_space(self):
        """
        Docstring for project_onto_tangent_space

        Project matrix Upsilon onto the tangent space of the Grassman manifold
        at state Theta.
        """
        self.gradJ_Phi = self.gradJ_Phi - self.Phi @ self.Phi.T @ self.gradJ_Phi
        self.gradJ_Psi = self.gradJ_Psi - self.Psi @ self.Psi.T @ self.gradJ_Psi

    # Geodesic calculations
    def inner_product(self, X1, Y1, X2, Y2):
        """
        Docstring for inner_product

        Compute the product Grassman inner product between (X1, Y1) and (X2, Y2).
        """

        return np.sum(X1 * X2) + np.sum(Y1 * Y2)

    def gradient_norm(self):
        return np.sqrt(
            self.inner_product(
                self.gradJ_Phi,
                self.gradJ_Psi,
                self.gradJ_Phi,
                self.gradJ_Psi,
            )
        )

    def geodesic(self, alpha, Ux, Sx, Vx, Uy, Sy, Vy):
        """
        Docstring for geodesic

        Compute the geodesic on the product Grassman manifold.

        :param alpha: step size along the geodesic
        :param Ux: from SVD of search direction in Phi (n x r).
        :param Sx: from SVD of search direction in Phi (r x r). May contain zeros.
        :param Vx: from SVD of search direction in Phi (r x r).
        :param Uy: from SVD of search direction in Psi (n x r).
        :param Sy: from SVD of search direction in Psi (r x r). May contain zeros.
        :param Vy: from SVD of search direction in Psi (r x r).
        """
        Phi_alpha = (
            self.Phi @ (Vx * np.cos(alpha * Sx)) @ Vx.T
            + (Ux * np.sin(alpha * Sx)) @ Vx.T
        )
        Psi_alpha = (
            self.Psi @ (Vy * np.cos(alpha * Sy)) @ Vy.T
            + (Uy * np.sin(alpha * Sy)) @ Vy.T
        )
        return Phi_alpha, Psi_alpha

    def parallel_translate(self, alpha, Ux, Sx, Vx, Uy, Sy, Vy, X, Y):
        """
        Docstring for parallel_translate

        Parallel transport of (X, Y) along its own geodesic.

        :param self: Description
        :param alpha: Description
        :param Ux: Description
        :param Sx: Description
        :param Vx: Description
        :param Uy: Description
        :param Sy: Description
        :param Vy: Description
        :param X: Description
        :param Y: Description
        """
        Xtilde = (
            (-self.Phi @ (Vx * np.sin(alpha * Sx)) + Ux * np.cos(alpha * Sx)) @ Ux.T @ X
        )
        Ytilde = (
            (-self.Psi @ (Vy * np.sin(alpha * Sy)) + Uy * np.cos(alpha * Sy)) @ Uy.T @ Y
        )
        return Xtilde, Ytilde

    def get_cost_derivative(self, X, Y):
        self.compute_gradient()
        cost_derivative = self.inner_product(self.gradJ_Phi, self.gradJ_Psi, X, Y)

        # compute gradient error
        Ux, Sx, VxT = np.linalg.svd(X, full_matrices=False)
        Uy, Sy, VyT = np.linalg.svd(Y, full_matrices=False)
        Vx = VxT.T
        Vy = VyT.T

        normalizer = np.sqrt(
            self.inner_product(
                X,
                Y,
                X,
                Y,
            )
        )

        alpha = 1e-3 / normalizer

        Phi_alpha, Psi_alpha = self.geodesic(alpha, Ux, Sx, Vx, Uy, Sy, Vy)

        alpha_trooper = self.copy()
        alpha_trooper.set_phi_psi(Phi_alpha, Psi_alpha)
        J_plus = alpha_trooper.get_cost()

        true_cost_derivative = (J_plus - self.get_cost()) / alpha
        print("True cost derivative: " + str(true_cost_derivative))

        return cost_derivative

    # Gradient descent

    def bisection(
        self,
        Ux,
        Sx,
        Vx,
        Uy,
        Sy,
        Vy,
        X,
        Y,
        init_step_size,
        c1=0.01,
        c2=0.1,
        max_step_search_iters=50,
        verbose=False,
        profile=False,
    ):
        alpha_trooper = self.copy()

        J0 = alpha_trooper.get_cost()
        if profile:
            print("J0: " + str(J0))

        dJ0 = alpha_trooper.get_cost_derivative(X, Y)
        if profile:
            print("dJ0 magnitude: " + str(np.linalg.norm(dJ0)))

        normalizer = np.sqrt(
            self.inner_product(
                X,
                Y,
                X,
                Y,
            )
        )

        if profile:
            print("Normalizer: " + str(normalizer))

        lb = 0
        alpha = init_step_size / normalizer
        ub = np.inf
        if verbose:
            print("bisection iteration: 0")
            print("Lower bound: " + str(lb))
            print("Step size: " + str(alpha))
            print("Upper bound: " + str(ub))

        for _ in range(max_step_search_iters):
            start = time.time()
            Phi_alpha, Psi_alpha = self.geodesic(alpha, Ux, Sx, Vx, Uy, Sy, Vy)
            end = time.time()
            if profile:
                print("Geodesic calculation: " + str(end - start))

            start = time.time()
            alpha_trooper.set_phi_psi(Phi_alpha, Psi_alpha)
            end = time.time()
            if profile:
                print("Set phi psi: " + str(end - start))

            start = time.time()
            J = alpha_trooper.get_cost()
            end = time.time()
            if profile:
                print("Get cost: " + str(end - start))

            if J0 + c1 * alpha * dJ0 < J:
                ub = alpha
                alpha = 0.5 * (lb + ub)
            else:
                start = time.time()
                Xtilde, Ytilde = self.parallel_translate(
                    alpha, Ux, Sx, Vx, Uy, Sy, Vy, X, Y
                )
                end = time.time()
                if profile:
                    print("Parallel Tanslate: " + str(end - start))

                start = time.time()
                alpha_trooper.compute_gradient()
                end = time.time()
                if profile:
                    print("Compute gradient: " + str(end - start))

                start = time.time()
                dJ = alpha_trooper.get_cost_derivative(Xtilde, Ytilde)
                end = time.time()
                if profile:
                    print("Get cost derivative: " + str(end - start))

                if dJ < c2 * dJ0:
                    lb = alpha
                    alpha = 2 * lb if ub == np.inf else 0.5 * (lb + ub)
                else:
                    break
            if verbose:
                print("bisection iteration: " + str(_))
                print("Lower bound: " + str(lb))
                print("Step size: " + str(alpha))
                print("Upper bound: " + str(ub))

        alpha_trooper.compute_gradient()

        return alpha, alpha_trooper, normalizer

    def backtracking(
        self,
        Ux,
        Sx,
        Vx,
        Uy,
        Sy,
        Vy,
        X,
        Y,
        c1=0.1,
        c2=0.5,
        max_iters=30,
        verbose=False,
    ):
        alpha_trooper = self.copy()

        J0 = alpha_trooper.get_cost()
        dJ0 = alpha_trooper.get_cost_derivative(X, Y)

        normalizer = np.sqrt(
            self.inner_product(
                X,
                Y,
                X,
                Y,
            )
        )

        alpha = 0.1 / normalizer

        for iter in range(max_iters):
            Phi_alpha, Psi_alpha = self.geodesic(alpha, Ux, Sx, Vx, Uy, Sy, Vy)

            alpha_trooper.set_phi_psi(Phi_alpha, Psi_alpha)

            J = alpha_trooper.get_cost()

            if J0 + c1 * alpha * dJ0 < J:
                alpha *= c2
            else:
                break
        if verbose:
            print(
                "cost: "
                + str(J0)
                + "; dJ0: "
                + str(dJ0)
                + "; backtracking steps: "
                + str(iter)
            )

        alpha_trooper.compute_gradient()

        return alpha, alpha_trooper, iter == max_iters - 1

    def conjugate_gradient(
        self,
        max_iters=100,
        tol=1e-4,
        c1=0.1,
        c2=0.5,
        backtracking_iters=30,
        verbose=False,
    ):
        self.compute_gradient()
        gradient_norm = self.gradient_norm()
        X = -self.gradJ_Phi
        Y = -self.gradJ_Psi

        iter = 0

        print("Initial cost: " + str(self.get_cost()))
        print("Initial gradient norm: " + str(gradient_norm))
        while gradient_norm > tol and iter < max_iters:
            Ux, Sx, VxT = np.linalg.svd(X, full_matrices=False)
            Uy, Sy, VyT = np.linalg.svd(Y, full_matrices=False)
            Vx = VxT.T
            Vy = VyT.T

            alpha, alpha_trooper, reached_max_iters = self.backtracking(
                Ux,
                Sx,
                Vx,
                Uy,
                Sy,
                Vy,
                X,
                Y,
                c1,
                c2,
                max_iters=backtracking_iters,
                verbose=verbose,
            )

            if reached_max_iters:
                print("Backtracking reached maximum step search iterations.")
                break

            denominator_2 = self.inner_product(self.gradJ_Phi, self.gradJ_Psi, X, Y)

            X_tilde, Y_tilde = self.parallel_translate(
                alpha, Ux, Sx, Vx, Uy, Sy, Vy, X, Y
            )

            self.inherit(alpha_trooper)
            del alpha_trooper

            X_tilde[:, 0] = self.parity * X_tilde[:, 0]

            denominator_1 = self.inner_product(
                self.gradJ_Phi, self.gradJ_Psi, X_tilde, Y_tilde
            )

            gradient_norm = self.gradient_norm()
            beta = gradient_norm**2 / (denominator_1 - denominator_2)

            X = -self.gradJ_Phi + beta * X_tilde
            Y = -self.gradJ_Psi + beta * Y_tilde

            iter = iter + 1
        print("Final cost: " + str(self.get_cost()))
        print("Final gradient norm: " + str(gradient_norm))

    # Cost functions
    def get_mse(self):
        error = 0.0
        for i in range(len(self.times)):
            t = self.times[i]
            error += np.linalg.norm(self.Y[i] - self.yhat_fn(t)) ** 2
        return error / len(self.times)

    def get_cost(self):
        _, logabsdet = np.linalg.slogdet(self.Psi.T @ self.Phi)
        reg = -2 * self.gamma * logabsdet
        return 0.5 * self.get_mse() + reg

    # Copiers
    def copy(self):
        new_trooper = Trooper(
            n=self.n,
            r=self.r,
            d=self.d,
            m=self.m,
            f=self.f,
            g=self.g,
            Df=self.Df,
            Dg=self.Dg,
            Phi=self.Phi,
            Psi=self.Psi,
            gamma=self.gamma,
            u_fn=self.u_fn,
            x0=self.x0,
            T=self.times[-1],
            L=len(self.times),
            Y=self.Y,
            yhat_fn=self.yhat_fn,
            z_fn=self.z_fn,
            A=self.A,
            atol=self.atol,
            rtol=self.rtol,
            solver_method=self.solver_method,
        )
        return new_trooper

    def inherit(self, other):
        self.Phi = other.Phi
        self.Psi = other.Psi
        self.A = other.A
        self.Y = other.Y
        self.yhat_fn = other.yhat_fn
        self.z_fn = other.z_fn
        self.gradJ_Phi = other.gradJ_Phi
        self.gradJ_Psi = other.gradJ_Psi
        self.parity = other.parity
