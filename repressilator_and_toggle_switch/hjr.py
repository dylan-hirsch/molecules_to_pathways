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

    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    font = {'size': 15}
    plt.rc('font', **font)
    return dynamics, hj, jnp, np, ode, plt


@app.cell
def _(dynamics, hj, jnp, np):
    model = dynamics.reduced_model(rank = 3)

    T = dynamics.T[:,0:3]
    Tinv = dynamics.Tinv[0:3,:]
    zmax = np.array([.25,.25,1]) #Tinv @ np.array([1.,1.,1.,1.,1.])

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(jnp.minimum(-zmax,zmax),
                                                                               jnp.maximum(-zmax,zmax)),
                                                                   (100, 100, 100),
                                                                   periodic_dims=None)

    xgrid = T @ grid.states[...,None]
    xgrid = xgrid[...,0]
    l = np.linalg.norm(xgrid[...,[3]] - xgrid[...,[4]], axis = -1)
    return Tinv, grid, l, model


@app.cell
def _(grid, hj, l, model, np):
    t0 = -10
    times = np.linspace(0., t0, 100)
    solver_settings = hj.SolverSettings.with_accuracy("medium")
    V = hj.solve(solver_settings, model, grid, times, l)
    return V, t0, times


@app.cell
def _(V):
    import pickle
    with open('/Users/dylanhirsch/Research/model_reduction/V_decoy.pkl', 'wb') as file:
        pickle.dump(V, file)
    return


@app.cell
def _(Tinv, V, grid, model, np, ode, t0, times):
    def mm(x, K, n):
        return 1. / (1. + (abs(x)/K)**n)

    def dx(t, x, grad_valuess, grid, model, times):

        K1 = 0.25
        K2 = 0.25
        K3 = 0.25
        K4 = 0.25
        K5 = 0.25

        n1 = 4.
        n2 = 4.
        n3 = 4.
        n4 = 2.
        n5 = 2.
    
        i = np.argmin(np.abs(times - t))

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
    
        x = np.array(x).reshape([len(x)])
        z = Tinv @ x
    
        grad_value = grid.interpolate(grad_valuess[i], state = z)
    
        u = model.optimal_control(z, t, grad_value)[0]

        dx1 = mm(x3,K3,n3) - x1
        dx2 = mm(x1,K1,n1) - x2
        dx3 = mm(x2,K2,n2) - x3
        dx4 = u*mm(x2,K2,n2) + mm(x5,K5,n5) - x4
        dx5 = u*mm(x3,K3,n3) + mm(x4,K4,n4) - x5
    
        dx = np.array([dx1, dx2, dx3, dx4, dx5])
    
        return dx

    grad_valuess = [grid.grad_values(V[i,...]) for i in range(len(times))]
    x0 = np.array([.1,.2,.15,1.,0.])  
    sol = ode.solve_ivp(lambda t,x:dx(t, x, grad_valuess, grid, model, times), 
                        [t0, 0], x0, max_step = .1)
    return grad_valuess, sol


@app.cell
def _(plt, sol):
    for _ in range(5):
        plt.plot(sol.t,sol.y[_,:])

    plt.show()
    return


@app.cell
def _(Tinv, grad_valuess, grid, model, np, plt, sol, times):
    us = []
    for i in range(len(sol.t)):
        t = sol.t[i]
        x = np.array(sol.y[:,i]).reshape([5])
        z = Tinv @ x
        j = np.argmin(np.abs(times - t))
        grad_value = grid.interpolate(grad_valuess[j], state = z) 
        u = model.optimal_control(z, t, grad_value)[0]
        us.append(u)

    plt.plot(us)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
