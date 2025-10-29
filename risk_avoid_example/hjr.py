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

    import random

    random.seed(123)
    np.random.seed(123)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    font = {"size": 15}
    plt.rc("font", **font)
    return dynamics, hj, jnp, np, plt


@app.cell
def _(dynamics, hj, jnp, np):
    model = dynamics.model()

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([-10, 0]), jnp.array([+10, +10])),
        (50, 50),
        periodic_dims=None,
    )

    l1 = np.linalg.norm(grid.states[..., [0]] - np.array([0.0]), axis=-1) - 1
    l2 = 5 * np.linalg.norm(grid.states[..., [1]] - np.array([1.0]), axis=-1) - 1
    g1 = 2 - np.linalg.norm(grid.states[..., [0]] - np.array([-5.0]), axis=-1)
    g2 = 2 - np.linalg.norm(grid.states[..., [0]] - np.array([+5.0]), axis=-1)
    g3 = 2.5 - np.linalg.norm(grid.states[..., [1]] - np.array([2.5]), axis=-1)


    l = np.minimum(np.maximum(l1, l2), 5.0)
    g = np.maximum(np.minimum(g1, g3), np.minimum(g2, g3))


    def value_postprocessor(t, v, l=l, g=g):
        return jnp.maximum(jnp.minimum(v, l), g)


    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )
    return g, grid, l, model, solver_settings


@app.cell
def _(np):
    np.maximum(np.array([0, 0, 0.5]), 1)
    return


@app.cell
def _(g, grid, hj, l, model, np, solver_settings):
    t0 = -10
    times = np.linspace(0.0, t0, 100)
    V0 = hj.solve(solver_settings, model, grid, times, np.maximum(l, g))
    return (V0,)


@app.cell
def _(V0, np, plt):
    #!/usr/bin/env python3
    """
    Animate a time-varying value function V(t, x, y) as a 3D surface, with
    obstacle and target surfaces overlaid.

    Inputs:
    - V: np.ndarray, shape (nt, nx, ny)
    - X, Y: 1D arrays for grid coordinates (optional). If None, indices are used.
    - obstacle: either None, shape (nx, ny), or shape (nt, nx, ny)
    - target:   either None, shape (nx, ny), or shape (nt, nx, ny)

    Usage:
    - Fill in/load your arrays at the bottom under `if __name__ == "__main__":`.
    - Optionally set `save="movie.mp4"` or `"movie.gif"` to export.
    """

    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)
    from typing import Optional, Tuple, Union

    Array3 = np.ndarray
    Array2 = np.ndarray
    Array1 = np.ndarray


    def _ensure_time_axis(
        A: Optional[Union[Array3, Array2]], nt: int
    ) -> Optional[Array3]:
        """Broadcast a 2D array to (nt, nx, ny); pass through 3D; None stays None."""
        if A is None:
            return None
        if A.ndim == 2:
            return np.broadcast_to(A, (nt,) + A.shape)
        if A.ndim == 3:
            return A
        raise ValueError("Array must be 2D (nx, ny) or 3D (nt, nx, ny).")


    def animate_value_surface(
        V: Array3,
        X: Optional[Array1] = None,
        Y: Optional[Array1] = None,
        obstacle: Optional[Union[Array3, Array2]] = None,
        target: Optional[Union[Array3, Array2]] = None,
        fps: int = 20,
        elev: float = 30,
        azim: float = -60,
        zlim: Optional[Tuple[float, float]] = None,
        downsample: int = 1,
        save: Optional[str] = None,
        dpi: int = 120,
    ):
        """
        Create and (optionally) save an animation of V(t,x,y) with obstacle/target overlays.
        """
        if V.ndim != 3:
            raise ValueError("V must be a 3D array with shape (nt, nx, ny).")
        nt, nx, ny = V.shape

        # Downsample for speed if requested
        ds = max(1, int(downsample))
        Vd = V[:, ::ds, ::ds]
        nx_d, ny_d = Vd.shape[1:]
        if X is None:
            Xd = np.arange(0, nx, ds)
        else:
            if X.ndim != 1 or len(X) != nx:
                raise ValueError("X must be 1D of length nx.")
            Xd = X[::ds]
        if Y is None:
            Yd = np.arange(0, ny, ds)
        else:
            if Y.ndim != 1 or len(Y) != ny:
                raise ValueError("Y must be 1D of length ny.")
            Yd = Y[::ds]

        # Broadcast obstacle/target to time axis and downsample
        Od = _ensure_time_axis(obstacle, nt)
        Td = _ensure_time_axis(target, nt)
        if Od is not None:
            Od = Od[:, ::ds, ::ds]
        if Td is not None:
            Td = Td[:, ::ds, ::ds]

        # Build meshgrid for plotting
        Xg, Yg = np.meshgrid(Xd, Yd, indexing="ij")  # (nx_d, ny_d)

        # Figure/axes setup
        fig = plt.figure(figsize=(8, 6), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        # Z limits: default spans all data (including obstacle/target if present)
        if zlim is None:
            zmin = np.nanmin(Vd)
            zmax = np.nanmax(Vd)
            if Od is not None:
                zmin = min(zmin, np.nanmin(Od))
                zmax = max(zmax, np.nanmax(Od))
            if Td is not None:
                zmin = min(zmin, np.nanmin(Td))
                zmax = max(zmax, np.nanmax(Td))
            pad = 0.05 * (zmax - zmin + 1e-12)
            zlim = (zmin - pad, zmax + pad)
        ax.set_zlim(*zlim)

        # For legend: create dummy handles with desired colors
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D([0], [0], color="lightblue", lw=4, label=r"$V(x,y,t)$"),
        ]
        if Od is not None:
            legend_handles.append(
                Line2D([0], [0], color="red", lw=4, label=r"$g(x,y)$")
            )
        if Td is not None:
            legend_handles.append(
                Line2D([0], [0], color="green", lw=4, label=r"$\ell(x,y)$")
            )
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0, 1))
        ax.invert_xaxis()

        # Draw initial frame
        surf_value = [None]
        surf_obst = [None]
        surf_targ = [None]

        def draw_frame(tidx):
            # Clear previously drawn surfaces (only the surfaces, keep axes/labels)
            for coll in [surf_value[0], surf_obst[0], surf_targ[0]]:
                if coll is not None:
                    coll.remove()

            # Value surface
            surf_value[0] = ax.plot_surface(
                Xg,
                Yg,
                Vd[tidx],
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=True,
                alpha=0.9,
                color="lightblue",
            )

            # Obstacle (red), slightly transparent so overlaps are visible
            if Od is not None:
                surf_obst[0] = ax.plot_surface(
                    Xg,
                    Yg,
                    Od[tidx],
                    rstride=1,
                    cstride=1,
                    linewidth=0.2,
                    antialiased=True,
                    alpha=0.5,
                    color="red",
                )

            # Target (green)
            if Td is not None:
                surf_targ[0] = ax.plot_surface(
                    Xg,
                    Yg,
                    Td[tidx],
                    rstride=1,
                    cstride=1,
                    linewidth=0.2,
                    antialiased=True,
                    alpha=0.5,
                    color="green",
                )

        # Initialize with first frame
        draw_frame(0)

        def update(tidx):
            draw_frame(tidx)
            return []

        anim = FuncAnimation(
            fig, update, frames=nt, interval=1000 / fps, blit=False, repeat=True
        )

        if save:
            if save.lower().endswith(".gif"):
                anim.save(save, writer=PillowWriter(fps=fps))
                print(f"Saved GIF to {save}")
            else:
                # For MP4, you need ffmpeg installed in your PATH
                anim.save(save, writer="ffmpeg", fps=fps, dpi=dpi)
                print(f"Saved MP4 to {save}")
        else:
            plt.tight_layout()
            plt.show()

        return anim


    # Example synthetic grid and data (replace with your arrays)
    nt, nx, ny = V0.shape
    x = np.linspace(-10.0, 10.0, nx)
    y = np.linspace(0.0, 10.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Example double obstacle: lower/upper surfaces (static here)
    lower_obstacle = np.maximum(
        np.minimum(2 - np.abs(X - 5), 2.5 - np.abs(Y - 2.5)),
        np.minimum(2 - np.abs(X + 5), 2.5 - np.abs(Y - 2.5)),
    )  # red
    target_surface = np.minimum(
        np.maximum(np.abs(X) - 1, 5 * np.abs(Y - 1) - 1.0), 5.0
    )  # green

    # ======== Call the animator ========
    animation = animate_value_surface(
        V0,
        X=x,
        Y=y,
        obstacle=lower_obstacle,  # can also pass (nt, nx, ny) array
        target=target_surface,  # can also pass (nt, nx, ny) array
        fps=20,
        elev=30,
        azim=-60,
        downsample=1,  # speed-up for big grids; set to 1 for full res
        save=None,  # e.g., "value_movie.mp4" or "value_movie.gif"
        dpi=120,
    )

    writer = PillowWriter(fps=20)
    animation.save("/Users/dylanhirsch/Desktop/test.gif", writer=writer)
    return


if __name__ == "__main__":
    app.run()
