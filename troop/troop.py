import random
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quad_vec

random.seed(23)
np.random.seed(23)

class troop:
    
    def __init__(self, n, r, d, m, f, g, Df, Dg, Phi = None, Psi = None, U = None, x0 = None, T = 1.0, L = 100, gamma = 0.001):
        
        self.n = n
        self.r = r
        self.d = d
        self.m = m

        self.f = f
        self.g = g
        self.Df = Df
        self.Dg = Dg

        if Phi is None:
            Phi = np.random.normal(size=(self.n, self.r))
        elif abs(np.linalg.det(Phi.T @ Phi)) < 1e-12:
            raise ValueError("Phi must be full rank.")
        self.Phi = Phi

        if Psi is None:
            Psi = np.random.normal(size=(self.n, self.r))
        elif abs(np.linalg.det(Psi.T @ Psi)) < 1e-12:
            raise ValueError("Psi must be full rank.")
        self.Psi = Psi

        self.standardize_representatives()

        if U is None:
            U = lambda t: np.zeros((self.d,))
        self.U = U

        if x0 is None:
            x0 = np.zeros((self.n,))
        self.x0 = x0

        self.times = np.linspace(0, T, L)
        self.gamma = gamma

        self.Y = None
        self.simulate_FOM()

        self.Yhat = None
        self.Z = None
        self.simulate_ROM()


    def standardize_representatives(self):
        # We map Phi0 and Psi0 to members of their equivalence class Phi and Psi which satisfy
        # Phi' @ Phi = I_r
        # Psi' @ Psi = I_r
        # det(Psi' @ Phi) > 0
        self.Phi, _ = np.linalg.qr(self.Phi)
        self.Psi, _ = np.linalg.qr(self.Psi)

        parity = 1
        if np.linalg.det(self.Psi.T @ self.Phi) < 0:
            self.Psi[:, 0] = -self.Psi[:, 0]
            parity = -1
        
        self.M = np.linalg.inv(self.Psi.T @ self.Phi)
        return parity

    ### Simulations of FOM and ROM

    def simulate_FOM(self):
        
        # Simulate FOM
        sol = solve_ivp(lambda t, x: self.f(x, self.U(t)), 
                        y0 = self.x0,
                        t_span=(self.times[0], self.times[-1]),
                        dense_output=True)
        
        self.Y = lambda t: self.g(sol.sol(t))


    def simulate_ROM(self):

        # Simulate ROM
        sol = solve_ivp(
            lambda t, z: self.M @ (self.Psi.T @ self.f(self.Phi @ z, self.U(t))),
            y0=self.M @ (self.Psi.T @ self.x0),
            t_span=(self.times[0], self.times[-1]),
            dense_output=True)
        self.Z = lambda t: sol.sol(t)
        self.Yhat = lambda t: self.g(self.Phi @ self.Z(t))

    ### Adjoint calculations

    def F_adjoint(self, z, u):
        df_tilde_dz = self.M @ self.Psi.T @ self.Df(self.Phi @ z, u) @ self.Phi
        return df_tilde_dz.T

    def S_adjoint_Phi(self, v, z, u):
        x = self.Phi @ z
        f_tilde = self.M @ self.Psi.T @ self.f(x, u)
        return self.Df(x, u).T @ self.Psi @ self.M.T @ v.reshape((self.r, 1)) @ z.reshape((1, self.r)) \
            - self.Psi @ self.M.T @ v.reshape((self.r, 1)) @ f_tilde.reshape((1, self.r))

    def S_adjoint_Psi(self, v, z, u):
        x = self.Phi @ z
        f_tilde = self.M @ self.Psi.T @ self.f(x, u)

        return (self.f(x, u) - self.Phi @ f_tilde).reshape(self.n, 1) @ (v.reshape((1, self.r)) @ self.M)

    def H_adjoint(self, z):
        H = self.Dg(self.Phi @ z) @ self.Phi
        return H.T

    def T_adjoint_Phi(self, w, z):
        return (self.Dg(self.Phi @ z).T @ w.reshape((self.m, 1))) @ z.reshape((1, self.r))

    def T_adjoint_Psi(self):
        return np.zeros((self.n, self.r))

    def grad_z_adjoint_Phi(self, v, z):
        return -self.Psi @ self.M.T @ v.reshape((self.r, 1)) @ z.reshape((1, self.r))

    def grad_z_adjoint_Psi(self, v, z):
        return (self.x0 - self.Phi @ z).reshape((self.n, 1)) @ v.reshape((1, self.r)) @ self.M

    ### Initialization
        
    def init_grad(self):
        t = self.times[-1]
        error = (self.Yhat(t) - self.Y(t)).reshape(self.m, 1)
        z = self.Z(t)
        gradJ_Phi = self.T_adjoint_Phi(error, z)
        gradJ_Psi = self.T_adjoint_Psi()

        return gradJ_Phi, gradJ_Psi

    def init_dual(self):
        t = self.times[-1]
        error = (self.Yhat(t) - self.Y(t)).reshape(self.m, 1)
        z = self.Z(t)
        p = self.H_adjoint(z) @ error

        return p.reshape((self.r,))

    ### Dual dynamics

    def calculate_dual_dynamics(self, p, z, u):
        dp = -self.F_adjoint(z, u) @ p
        return dp
    
    ### Calculate gradients

    def compute_gradient(self):

        # Initialize the gradient ...
        gradJ_Phi, gradJ_Psi = self.init_grad()
        # Compute adjoint variable at final time ... (we use p in place of lambda)
        p = self.init_dual()

        # For l in ...
        #for l in reversed(range(len(self.times) - 1)):
        for tl, tlplus1 in zip(reversed(self.times[:-1]), reversed(self.times[1:])):
            # Solve the adjoint equation ...
            dual_sol = solve_ivp(
                lambda t, q: self.calculate_dual_dynamics(
                    q,
                    self.Z(t),
                    self.U(t),
                ),
                [tlplus1, tl],
                p,
                dense_output=True)
            P = lambda t: dual_sol.sol(t)

            # Compute the integral component ...
            gradJ_Phi += quad_vec(
                    lambda t: self.S_adjoint_Phi(P(t), self.Z(t), self.U(t)),
                    tl,
                    tlplus1,
                )[0]
            gradJ_Psi += quad_vec(
                    lambda t: self.S_adjoint_Psi(P(t), self.Z(t), self.U(t)),
                    tl,
                    tlplus1,
                )[1]

            # Add lth element of the sum ...
            error = self.Yhat(tl) - self.Y(tl)
            gradJ_Phi += self.T_adjoint_Phi(error, self.Z(tl))
            gradJ_Psi += self.T_adjoint_Psi()

            # Add "jump" to adjoint ...
            p = P(tl) + self.H_adjoint(self.Z(tl)) @ error

        # add gradient due to initial condition
        gradJ_Phi += self.grad_z_adjoint_Phi(p, self.Z(0))
        gradJ_Psi += self.grad_z_adjoint_Psi(p, self.Z(0))

        # normalize by trajectory length
        gradJ_Phi = gradJ_Phi / len(self.times)
        gradJ_Psi = gradJ_Psi / len(self.times)

        # Add regularization
        gradJ_Phi += 0 * self.gamma * 2 * (self.Phi - self.Psi @ self.M.T)
        gradJ_Psi += self.gamma * 2 * (self.Psi - self.Phi @ self.M)

        # Project onto tangent spaces
        gradJ_Phi = self.project_onto_tangent_space(self.Phi, gradJ_Phi)
        gradJ_Psi = self.project_onto_tangent_space(self.Psi, gradJ_Psi)

        return gradJ_Phi, gradJ_Psi

    def project_onto_tangent_space(self, Theta, Upsilon):
        # Project matrix Upsilon onto the tangent space of the Grassman manifold at state Theta
        return Upsilon - Theta @ Theta.T @ Upsilon

    def conjugate_gradient(self):
        pass

    def get_mse(self):
        
        error = 0.0
        for t in self.times:
            error += np.linalg.norm(self.Y(t) - self.Yhat(t))**2
        error = np.sqrt(error / len(self.times))
        return error
    
    def get_cost(self):

        reg = -2 * self.gamma * np.log(np.linalg.det(self.Psi.T @ self.Phi))
        return self.get_mse() / len(self.times) + reg