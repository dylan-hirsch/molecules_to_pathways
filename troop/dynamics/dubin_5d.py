import jax.numpy as jnp
from hj_reachability import dynamics, sets


class Dubin5dReducedModel(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        Phi,
        Psi,
        control_mode="min",
        disturbance_mode="max",
        control_space=None,
        disturbance_space=None,
        rank=3,
        uMax=+1.0,
        uMin=-1.0,
        dMax=0.00,
        dMin=0.00,
    ):
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        self.Phi = jnp.asarray(Phi)
        self.Psi = jnp.asarray(Psi)
        self.Derivative_Projection = jnp.linalg.inv(Psi.T @ Phi) @ self.Psi.T
        self.rank = rank

        if control_space is None:
            control_space = sets.Box(
                jnp.array([self.uMin] * 2), jnp.array([self.uMax] * 2)
            )
        if disturbance_space is None:
            disturbance_space = sets.Box(
                jnp.array([self.dMin] * 5),
                jnp.array([self.dMax] * 5),
            )
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        xstate = self.Phi @ state.reshape([self.rank, 1])
        x, y, theta, v, omega = xstate
        x = x[0]
        y = y[0]
        theta = theta[0]
        v = v[0]
        omega = omega[0]

        fx = jnp.array(
            [[v * jnp.cos(theta)], [v * jnp.sin(theta)], [omega], [0.0], [0.0]]
        )

        fz = self.Derivative_Projection @ fx

        return fz.reshape([self.rank])

    def control_jacobian(self, state, time):
        # xstate = self.Phi @ state.reshape([self.rank, 1]) + self.x_star
        # x, y, theta = xstate
        # x = x[0]
        # y = y[0]
        # theta = theta[0]

        gux = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        guz = self.Derivative_Projection @ gux

        return guz

    def disturbance_jacobian(self, state, time):
        # xstate = self.Phi @ state.reshape([self.rank, 1]) + self.x_star
        # x, y, theta = xstate
        # x = x[0]
        # y = y[0]
        # theta = theta[0]

        gdx = jnp.zeros((5, 5))

        gdz = self.Derivative_Projection @ gdx

        return gdz
