import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad_vec
from scipy.interpolate import make_interp_spline as spline

n = 10  # FOM size
r = 5  # ROM size

d = 1  # input size
m = 1  # output size

L = 100  # number of time steps
T = 1  # final time

class troop:
    
    def __init__(self, times, x0, U, n_states, n_reduced_states, n_inputs, n_outputs, n_samples, t_final, f, g, Df, Dg, Phi = None, Psi = None):
        self.times = times
        self.x0 = x0
        self.U = U
        
        self.n = n_states
        self.r = n_reduced_states
        self.d = n_inputs
        self.m = n_outputs
        self.L = n_samples
        self.T = t_final

        self.f = f
        self.g = g
        self.Df = Df
        self.Dg = Dg

        if Phi is None:
            Phi = np.random.normal(size=(self.n, self.r))
        if Psi is None:
            Psi = np.random.normal(size=(self.n, self.r))

        self.standardize_representatives()

    def standardize_representatives(self):
        # We map Phi0 and Psi0 to members of their equivalence class Phi and Psi which satisfy
        # Phi' @ Phi = I_r
        # Psi' @ Psi = I_r
        # det(Psi' @ Phi) > 0
        self.Phi, _ = np.linalg.qr(self.Phi)
        self.Psi, _ = np.linalg.qr(self.Psi)
        if np.linalg.det(self.Psi.T @ self.Phi) < 0:
            self.Psi = -self.Psi

    def simulate_FOM(self):
        sol = solve_ivp(lambda t, x: self.f(x, self.U(t)), 
                        y0 = self.x0, t_span=(self.times[0], 
                        self.times[-1]), t_eval = self.times)
        return [self.g(sol.y[:, i]) for i in range(len(self.times))]



    

    

    