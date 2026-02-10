import jax.numpy as jnp
from hj_reachability import dynamics, sets


class RepressiloggleatorReducedModel(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        Phi,
        Psi,
        Ks=(0.25,) * 5,
        ns=(4.0,) * 3 + (2.0,) * 2,
        control_mode="min",
        disturbance_mode="max",
        control_space=None,
        disturbance_space=None,
        uMax=1.0,
        uMin=0.0,
        dMax=0.00,
        dMin=0.00,
    ):
        self.Phi = jnp.asarray(Phi)
        self.Psi = jnp.asarray(Psi)
        self.Psi = self.Psi @ jnp.linalg.inv(Phi.T @ Psi)
        self.rank = self.Phi.shape[1]

        if control_space is None:
            control_space = sets.Box(jnp.array([uMin]), jnp.array([uMax]))
        if disturbance_space is None:
            disturbance_space = sets.Box(
                jnp.array([dMin, dMin, dMin, dMin, dMin]),
                jnp.array([dMax, dMax, dMax, dMax, dMax]),
            )

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

        self.Ks = Ks
        self.ns = ns

    def mm(self, x, K, n):
        return 1.0 / (1.0 + (jnp.abs(x) / K) ** n)

    def open_loop_dynamics(self, state, time):
        xstate = self.Phi @ state.reshape([self.rank, 1])

        x1, x2, x3, x4, x5 = xstate

        x1 = x1[0]
        x2 = x2[0]
        x3 = x3[0]
        x4 = x4[0]
        x5 = x5[0]

        K1 = self.Ks[0]
        K2 = self.Ks[1]
        K3 = self.Ks[2]
        K4 = self.Ks[3]
        K5 = self.Ks[4]

        n1 = self.ns[0]
        n2 = self.ns[1]
        n3 = self.ns[2]
        n4 = self.ns[3]
        n5 = self.ns[4]

        fx = jnp.array(
            [
                [self.mm(x3, K3, n3) - x1],
                [self.mm(x1, K1, n1) - x2],
                [self.mm(x2, K2, n2) - x3],
                [self.mm(x5, K5, n5) - x4],
                [self.mm(x4, K4, n4) - x5],
            ]
        )

        fz = self.Psi.T @ fx

        return fz.reshape([self.rank])

    def control_jacobian(self, state, time):
        xstate = self.Phi @ state.reshape([self.rank, 1])

        _, x2, x3, _, _ = xstate
        x2 = x2[0]
        x3 = x3[0]

        K2 = self.Ks[1]
        K3 = self.Ks[2]

        n2 = self.ns[1]
        n3 = self.ns[2]

        gux = jnp.array(
            [[0.0], [0.0], [0.0], [self.mm(x2, K2, n2)], [self.mm(x3, K3, n3)]]
        )

        guz = self.Psi.T @ gux

        return guz.reshape([self.rank, 1])

    def disturbance_jacobian(self, state, time):
        xstate = self.Phi @ state.reshape([self.rank, 1])

        x1, x2, x3, x4, x5 = xstate
        x1 = x1[0]
        x2 = x2[0]
        x3 = x3[0]
        x4 = x4[0]
        x5 = x5[0]

        K1 = self.Ks[0]
        K2 = self.Ks[1]
        K3 = self.Ks[2]
        K4 = self.Ks[3]
        K5 = self.Ks[4]

        n1 = self.ns[0]
        n2 = self.ns[1]
        n3 = self.ns[2]
        n4 = self.ns[3]
        n5 = self.ns[4]

        gdx = jnp.array(
            [
                [self.mm(x3, K3, n3), 0.0, 0.0, 0.0, 0.0],
                [0.0, self.mm(x1, K1, n1), 0.0, 0.0, 0.0],
                [0.0, 0.0, self.mm(x2, K2, n2), 0.0, 0.0],
                [0.0, 0.0, 0.0, self.mm(x5, K5, n5), 0.0],
                [0.0, 0.0, 0.0, 0.0, self.mm(x4, K4, n4)],
            ]
        )

        gdz = self.Psi.T @ gdx

        return gdz.reshape([self.rank, 5])
