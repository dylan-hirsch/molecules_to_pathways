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
def _():
    COLORS = ["blue", "orange", "green", "red", "purple"]
    LABELS = ["A", "B", "C", "X", "Y"]
    FONTSIZE1 = 20
    FONTSIZE2 = 15
    FONTSIZE3 = 15

    TSPAN = [0, 25]
    K1 = 1 / 4
    N1 = 4
    K2 = 1 / 4
    N2 = 2
    return (
        COLORS,
        FONTSIZE1,
        FONTSIZE2,
        FONTSIZE3,
        K1,
        K2,
        LABELS,
        N1,
        N2,
        TSPAN,
    )


@app.cell
def _(K1, K2, N1, N2, np):
    def repressilator(X):
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]

        dx1 = 1 / (1 + (x3 / K1) ** N1) - x1
        dx2 = 1 / (1 + (x1 / K1) ** N1) - x2
        dx3 = 1 / (1 + (x2 / K1) ** N1) - x3

        return np.array([dx1, dx2, dx3])


    def toggle_switch(X):
        x4 = X[0]
        x5 = X[1]

        dx4 = 1 / (1 + (x5 / K2) ** N2) - x4
        dx5 = 1 / (1 + (x4 / K2) ** N2) - x5

        return np.array([dx4, dx5])


    def repressilator_and_toggle_switch(X, u):
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]
        x4 = X[3]
        x5 = X[4]

        theta = 3

        dx1 = 1 / (1 + (x3 / K1) ** N1) - x1
        dx2 = 1 / (1 + (x1 / K1) ** N1) - x2
        dx3 = 1 / (1 + (x2 / K1) ** N1) - x3
        dx4 = u / (1 + (x2 / K2) ** N1) + 1 / (1 + (x5 / K2) ** N2) - x4
        dx5 = u / (1 + (x3 / K2) ** N1) + 1 / (1 + (x4 / K2) ** N2) - x5

        return np.array([dx1, dx2, dx3, dx4, dx5])
    return repressilator, repressilator_and_toggle_switch, toggle_switch


@app.cell
def _(
    COLORS,
    FONTSIZE1,
    FONTSIZE2,
    FONTSIZE3,
    LABELS,
    TSPAN,
    np,
    plt,
    repressilator,
    solve_ivp,
):
    def _():
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        X0 = np.random.uniform(size=(3,))
        sol = solve_ivp(lambda t, X: repressilator(X), TSPAN, X0, max_step=0.01)

        for i in range(3):
            ax.plot(sol.t, sol.y[i, :], color=COLORS[i], label=LABELS[i])

        ax.set_xlabel("Time (cell generations)", fontsize=FONTSIZE2)
        ax.set_ylabel("Protein Concentration (Normalized)", fontsize=FONTSIZE2)
        ax.set_title("Repressilator Dynamics", fontsize=FONTSIZE1)
        ax.legend(fontsize=FONTSIZE3)
        ax.set_ylim([-0.05, 1.05])

        plt.savefig(
            "/Users/dylanhirsch/Desktop/repressilator.svg", bbox_inches="tight"
        )
        plt.show()


    _()
    return


@app.cell
def _(
    COLORS,
    FONTSIZE1,
    FONTSIZE2,
    FONTSIZE3,
    LABELS,
    TSPAN,
    np,
    plt,
    solve_ivp,
    toggle_switch,
):
    def _():
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        ax = axs[0]

        ax.set_xlabel("Time (cell generations)", fontsize=FONTSIZE2)
        ax.set_ylabel("Protein Concentration (Normalized)", fontsize=FONTSIZE2)
        ax.set_title("Toggle Switch (X starts high)", fontsize=FONTSIZE1)
        ax.set_ylim([-0.05, 1.05])

        X0 = np.array([1, 0])
        sol = solve_ivp(lambda t, X: toggle_switch(X), TSPAN, X0, max_step=0.01)

        for i in range(2):
            ax.plot(sol.t, sol.y[i, :], color=COLORS[i + 3], label=LABELS[i + 3])

        ax.legend(fontsize=FONTSIZE3)

        ax = axs[1]

        ax.set_xlabel("Time (cell generations)", fontsize=FONTSIZE2)
        ax.set_ylabel("Protein Concentration (Normalized)", fontsize=FONTSIZE2)
        ax.set_title("Toggle Switch (Y starts high)", fontsize=FONTSIZE1)
        ax.set_ylim([-0.05, 1.05])

        X0 = np.array([0, 1])
        sol = solve_ivp(lambda t, X: toggle_switch(X), TSPAN, X0, max_step=0.01)

        for i in range(2):
            ax.plot(sol.t, sol.y[i, :], color=COLORS[i + 3], label=LABELS[i + 3])

        ax.legend(fontsize=FONTSIZE3)

        plt.savefig(
            "/Users/dylanhirsch/Desktop/toggle_switch.svg", bbox_inches="tight"
        )
        plt.show()


    _()
    return


@app.cell
def _(
    COLORS,
    FONTSIZE1,
    FONTSIZE2,
    FONTSIZE3,
    LABELS,
    TSPAN,
    np,
    plt,
    repressilator_and_toggle_switch,
    solve_ivp,
):
    def _():
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        us = [0, 0.5, 1]

        for j in range(3):
            ax = axs[j]

            X0 = np.concatenate([np.random.uniform(size=(3,)), np.array([1, 0])])
            sol = solve_ivp(
                lambda t, X: repressilator_and_toggle_switch(X, u=us[j]),
                TSPAN,
                X0,
                max_step=0.01,
            )

            for i in range(5):
                ax.plot(sol.t, sol.y[i, :], color=COLORS[i], label=LABELS[i])

            ax.set_xlabel("Time (cell generations)", fontsize=FONTSIZE2)
            ax.set_ylabel("Protein Concentration (Normalized)", fontsize=FONTSIZE2)
            ax.set_title("u = " + str(us[j]), fontsize=FONTSIZE1)

            ax.legend(fontsize=FONTSIZE3)
            ax.set_ylim([-0.05, 2.05])

        plt.savefig(
            "/Users/dylanhirsch/Desktop/not_working.svg", bbox_inches="tight"
        )
        plt.show()


    _()
    return


@app.cell
def _(
    COLORS,
    FONTSIZE1,
    FONTSIZE2,
    FONTSIZE3,
    LABELS,
    TSPAN,
    np,
    plt,
    repressilator_and_toggle_switch,
    solve_ivp,
):
    def _():
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        X0 = np.concatenate([np.random.uniform(size=(3,)), np.array([1, 0])])
        sol = solve_ivp(
            lambda t, X: repressilator_and_toggle_switch(
                X, u=1 if X[1] > X[2] else 0
            ),
            TSPAN,
            X0,
            max_step=0.01,
        )

        for i in range(5):
            ax.plot(sol.t, sol.y[i, :], color=COLORS[i], label=LABELS[i])

        ax.set_xlabel("Time (cell generations)", fontsize=FONTSIZE2)
        ax.set_ylabel("Protein Concentration (Normalized)", fontsize=FONTSIZE2)
        ax.set_title("u = [B > C]", fontsize=FONTSIZE1)

        ax.legend(fontsize=FONTSIZE3)

        plt.savefig("/Users/dylanhirsch/Desktop/working.svg", bbox_inches="tight")

        plt.show()


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
