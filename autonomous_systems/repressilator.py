import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    return np, plt, solve_ivp


@app.cell
def _(np, u):
    def repressilator(X):
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]

        alpha = 3
    
        dx1 = alpha / (1 + x3**4) - x1
        dx2 = alpha / (1 + x1**4) - x2
        dx3 = alpha / (1 + x2**4) - x3
        return np.array([dx1, dx2, dx3])

    def toggle_switch(X):
        x4 = X[0]
        x5 = X[1]

        beta = 3
    
        dx4 = beta / (1 + x5**4) - x4
        dx5 = beta / (1 + x4**4) - x5
        return np.array([dx4, dx5])

    def repressilator_and_toggle_switch(X):
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]
        x4 = X[3]
        x5 = X[4]

        alpha = 3
        beta = 3

        dx1 = alpha / (1 + x3**4) - x1
        dx2 = alpha / (1 + x1**4) - x2
        dx3 = alpha / (1 + x2**4) - x3
        dx4 = (u * x2 + (1 - u) * x3) * beta / (1 + x5**4) - x4
        dx5 = beta / (1 + x4**4) - x5
    
        return np.array([dx1, dx2, dx3, dx4, dx5])
    return repressilator_and_toggle_switch, toggle_switch


@app.cell
def _(np, plt, repressilator_and_toggle_switch, solve_ivp):
    def _():

        X0 = np.random.uniform(size = (3,))
        tspan = [0,100]
        sol = solve_ivp(lambda t,X: repressilator_and_toggle_switch(X), tspan, X0, max_step=.01)

        for i in range(3):
            plt.plot(sol.t,sol.y[i,:])

        plt.show()

    _()
    return


@app.cell
def _(np, plt, solve_ivp, toggle_switch):
    def _():

        X0 = np.array([1,0])
        tspan = [0,100]
        sol = solve_ivp(lambda t,X: toggle_switch(X), tspan, X0, max_step=.01)

        for i in range(2):
            plt.plot(sol.t,sol.y[i,:])

        plt.show()

    _()
    return


@app.cell
def _(np, plt, solve_ivp, toggle_switch):
    def _():

        X0 = np.array([0,0,0,1,0])
        tspan = [0,100]
        sol = solve_ivp(lambda t,X: toggle_switch(X), tspan, X0, max_step=.01)

        for i in range(2):
            plt.plot(sol.t,sol.y[i,:])

        plt.show()

    _()
    return


if __name__ == "__main__":
    app.run()
