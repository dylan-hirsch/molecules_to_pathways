import jax.numpy as jnp
from hj_reachability import dynamics, sets


class LinearReducedModel(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        Phi,
        Psi,
        A,
        Bu=None,
        Bd=None,
        control_mode="min",
        disturbance_mode="max",
        control_space=None,
        disturbance_space=None,
        control_center=(0.0,),
        control_radius=+1.0,
        disturbance_center=(0.0,),
        disturbance_radius=+0.0,
    ):
        self.Phi = jnp.asarray(Phi)
        self.Psi = jnp.asarray(Psi)
        self.Psi = self.Psi @ jnp.linalg.inv(Phi.T @ Psi)
        self.rank = self.Phi.shape[1]

        self.A = jnp.array(A)
        n, _ = A.shape

        if Bu is None:
            self.Bu = jnp.zeros((n, 1))
        else:
            self.Bu = jnp.array(Bu)

        if Bd is None:
            self.Bd = jnp.zeros((n, 1))
        else:
            self.Bd = jnp.array(Bd)

        if control_space is None:
            control_space = sets.Ball(jnp.array(control_center), control_radius)
        if disturbance_space is None:
            disturbance_space = sets.Ball(
                jnp.array(disturbance_center), disturbance_radius
            )

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        xstate = self.Phi @ state.reshape([self.rank, 1])

        fx = self.A @ xstate

        fz = self.Psi.T @ fx

        return fz.reshape([self.rank])

    def control_jacobian(self, state, time):
        gux = self.Bu

        guz = self.Psi.T @ gux

        return guz.reshape([self.rank, self.Bu.shape[-1]])

    def disturbance_jacobian(self, state, time):
        gdx = self.Bd

        gdz = self.Psi.T @ gdx

        return gdz.reshape([self.rank, self.Bd.shape[-1]])
