import jax.numpy as jnp
from hj_reachability import dynamics, sets


class Dubin4dReducedModel(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        Phi,
        Psi,
        control_mode="min",
        disturbance_mode="max",
        control_space=None,
        disturbance_space=None,
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
        self.Psi = self.Psi @ jnp.linalg.inv(Phi.T @ Psi)
        self.rank = self.Phi.shape[1]

        if control_space is None:
            control_space = sets.Box(jnp.array([self.uMin]), jnp.array([self.uMax]))
        if disturbance_space is None:
            disturbance_space = sets.Box(
                jnp.array([self.dMin] * 4),
                jnp.array([self.dMax] * 4),
            )
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        xstate = self.Phi @ state.reshape([self.rank, 1])
        x, y, theta, omega = xstate
        x = x[0]
        y = y[0]
        theta = theta[0]
        omega = omega[0]

        fx = jnp.array([[jnp.cos(theta)], [jnp.sin(theta)], [omega], [0.0]])

        fz = self.Psi.T @ fx

        return fz.reshape([self.rank])

    def control_jacobian(self, state, time):
        # xstate = self.Phi @ state.reshape([self.rank, 1]) + self.x_star
        # x, y, theta = xstate
        # x = x[0]
        # y = y[0]
        # theta = theta[0]

        gux = jnp.array([[0.0], [0.0], [0.0], [1.0]])

        guz = self.Psi.T @ gux

        return guz

    def disturbance_jacobian(self, state, time):
        # xstate = self.Phi @ state.reshape([self.rank, 1]) + self.x_star
        # x, y, theta = xstate
        # x = x[0]
        # y = y[0]
        # theta = theta[0]

        gdx = jnp.zeros((4, 4))

        gdz = self.Psi.T @ gdx

        return gdz
