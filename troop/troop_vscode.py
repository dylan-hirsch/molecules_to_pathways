import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quad_vec
from scipy.interpolate import make_interp_spline as spline

class troop:
    
    def __init__(self, n_states, n_reduced_states, n_inputs, n_outputs, f, g, Df, Dg, Phi = None, Psi = None):
        
        self.n = n_states
        self.r = n_reduced_states
        self.d = n_inputs
        self.m = n_outputs

        self.f = f
        self.g = g
        self.Df = Df
        self.Dg = Dg

        if Phi is None:
            Phi = np.random.normal(size=(self.n, self.r))
        elif abs(np.linalg.det(Phi.T)) < 1e-12:
            raise ValueError("Phi must be full rank.")
        self.Phi = Phi

        if Psi is None:
            Psi = np.random.normal(size=(self.n, self.r))
        elif abs(np.linalg.det(Psi.T)) < 1e-12:
            raise ValueError("Psi must be full rank.")
        self.Psi = Psi

        self.standardize_representatives()
        self.M = np.linalg.inv(self.Psi.T @ self.Phi)

    def standardize_representatives(self):
        # We map Phi0 and Psi0 to members of their equivalence class Phi and Psi which satisfy
        # Phi' @ Phi = I_r
        # Psi' @ Psi = I_r
        # det(Psi' @ Phi) > 0
        self.Phi, _ = np.linalg.qr(self.Phi)
        self.Psi, _ = np.linalg.qr(self.Psi)
        if np.linalg.det(self.Psi.T @ self.Phi) < 0:
            self.Psi = -self.Psi

    ### Simulations of FOM and ROM

    def simulate_FOM(self, U, T, L, x0):
        times = np.linspace(0, T, L)
        sol = solve_ivp(lambda t, x: self.f(x, U(t)), 
                        y0 = x0, t_span=(times[0], 
                        times[-1]), t_eval = times)
        return [self.g(sol.y[:, i]) for i in range(len(times))]

    def simulate_ROM(self, U, T, L, x0):
        times = np.linspace(0, T, L)
        sol = solve_ivp(
            lambda t, z: self.M @ (self.Psi.T @ self.f(self.Phi @ z, U(t))),
            y0=self.M @ (self.Psi.T @ x0),
            t_span=(times[0], times[-1]),
        )
        Z = spline(sol.t, sol.y.T)
        Yhat = [self.g(self.Phi @ Z(time)) for time in times]

        return Z, Yhat
    
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

    def grad_z_adjoint_Psi(self, x0, v, z):
        return (x0 - self.Phi @ z).reshape((self.n, 1)) @ v.reshape((1, self.r)) @ self.M

    ### Initialization
        
    def init_grad(self, Y, Yhat, Z, T):
        error = (Yhat[-1] - Y[-1]).reshape(self.m, 1)
        z = Z(T)
        gradJ_Phi = self.T_adjoint_Phi(error, z)
        gradJ_Psi = self.T_adjoint_Psi()

        return gradJ_Phi, gradJ_Psi

    def init_dual(self, Y, Yhat, Z, T):
        error = (Yhat[-1] - Y[-1]).reshape(self.m, 1)
        z = Z(T)
        p = self.H_adjoint(z) @ error

        return p.reshape((self.r,))

    ### Dual dynamics

    def calculate_dual_dynamics(self, p, z, u):
        dp = -self.F_adjoint(z, u).T @ p
        return dp
    
    ### Calculate gradients

    def compute_gradient(self, U, T, L, x0, gamma=0.01):

        # Simulate FOM ...
        Y = self.simulate_FOM(U, T, L, x0)

        # Assemble and simulate ...
        Z, Yhat = self.simulate_ROM(U, T, L, x0)

        # Initialize the gradient ...
        gradJ_Phi, gradJ_Psi = self.init_grad(Y, Yhat, Z, T)

        # Compute adjoint variable at final time ... (we use p in place of lambda)
        p = self.init_dual(Y, Yhat, Z, T)

        # For l in ...
        times = np.linspace(0, T, L)
        for l in reversed(range(L - 1)):
            tlplus1 = times[l + 1]
            tl = times[l]

            # Solve the adjoint equation ...
            dual_sol = solve_ivp(
                lambda t, p_dummy: self.calculate_dual_dynamics(
                    p_dummy,
                    Z(t),
                    U(t),
                ),
                [tlplus1, tl],
                p,
                t_eval=np.linspace(tl, tlplus1, L)[::-1],
            )
            taus = dual_sol.t[::-1]
            P = spline(taus, dual_sol.y[:, ::-1].T)

            # Compute the integral component ...
            gradJ_Phi += quad_vec(
                    lambda t: self.S_adjoint_Phi(P(t), Z(t), U(t)),
                    tl,
                    tlplus1,
                )[0]
            gradJ_Psi += quad_vec(
                    lambda t: self.S_adjoint_Psi(P(t), Z(t), U(t)),
                    tl,
                    tlplus1,
                )[1]

            # Add lth element of the sum ...
            error = Yhat[l] - Y[l]
            gradJ_Phi += self.T_adjoint_Phi(error, Z(tl))
            gradJ_Psi += self.T_adjoint_Psi()

            # Add "jump" to adjoint ...
            p += self.H_adjoint(Z(tl)) @ error

        # add gradient due to initial condition
        gradJ_Phi += self.grad_z_adjoint_Phi(p, Z(0))
        gradJ_Psi += self.grad_z_adjoint_Psi(x0, p, Z(0))

        # normalize by trajectory length
        gradJ_Phi = gradJ_Phi / L
        gradJ_Psi = gradJ_Psi / L

        # Add regularization
        gradJ_Phi += gamma * 2 * (self.Phi - self.Psi @ self.M.T)
        gradJ_Psi += gamma * 2 * (self.Psi - self.Phi @ self.M)

        return gradJ_Phi, gradJ_Psi
    
    def gradient_step(self, U, T, L, x0, gamma=0.01, alpha=0.01):
        gradJ_Phi, gradJ_Psi = self.compute_gradient(U, T, L, x0, gamma)

        self.Phi -= alpha * gradJ_Phi
        self.Psi -= alpha * gradJ_Psi

        self.standardize_representatives()
        self.M = np.linalg.inv(self.Psi.T @ self.Phi)

    def get_mse(self, U, T, L, x0):

        Y = self.simulate_FOM(U, T, L, x0)
        _, Yhat = self.simulate_ROM(U, T, L, x0)

        error = 0.0
        for y, yhat in zip(Y, Yhat):
            error += np.linalg.norm(y - yhat)**2
        error = np.sqrt(error / len(Y))
        return error
        



        