import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets

## This is the dynamics file for a biomolecular circuit consisting of a repressilator and a toggle switch.
## The structure of the biological circuit is (x1 --| x2 --| x3 --| x1) and (x4--|x5--|x4)

T = jnp.array([[-.70, .97, .70], [-.60,-.10,-.85], [-.46,-.53,.28]])
Tinv = jnp.linalg.pinv(T)

## MAPK BT (Nonlinear)
class mapk_nonlinear_bt(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None,
                 uMax=1.,
                 uMin=0.,
                 dMax=1.,
                 dMin=0.):
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        self.T = T
        self.Tinv = Tinv

        if control_space is None:
            control_space = sets.Box(jnp.array([self.uMin]), jnp.array([self.uMax]))
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([self.dMin, self.dMin, self.dMin]),
                                         jnp.array([self.dMax, self.dMax, self.dMax]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        xstate = self.T @ state.reshape([3,1])
        x1, x2, x3 = xstate
        x1 = x1[0]
        x2 = x2[0]
        x3 = x3[0]
        x4 = x4[0]
        x5 = x5[0]
        fx = self.Tinv @ jnp.array([[0.],
                                    [0.],
                                    [0.],
                                    [0.],
                                    [0.]])
        return fx.reshape([5])

    def control_jacobian(self, state, time):
        xstate = self.T @ state.reshape([3,1])
        x1, x2, x3 = xstate
        x1 = x1[0]
        x2 = x2[0]
        x3 = x3[0]
        x4 = x4[0]
        x5 = x5[0]
        gu = self.Tinv @ jnp.array([[0.],
                                    [0.],
                                    [0.],
                                    [0.],
                                    [0.]])
        return gu

    def disturbance_jacobian(self, state, time):
        xstate = self.T @ state.reshape([3,1])
        x1, x2, x3 = xstate
        x1 = x1[0]
        x2 = x2[0]
        x3 = x3[0]
        x4 = x4[0]
        x5 = x5[0]
        gd = self.Tinv @ jnp.array([[0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.]])
        return gd
    
class mapk_nonlinear_reduced_bt(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None,
                 uMax=1.,
                 uMin=0.,
                 dMax=1.,
                 dMin=0.):
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        self.T = T[:,[0,1,2]]
        self.Tinv = Tinv[[0,1,2],:]

        if control_space is None:
            control_space = sets.Box(jnp.array([self.uMin]), jnp.array([self.uMax]))
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([self.dMin, self.dMin, self.dMin]),
                                         jnp.array([self.dMax, self.dMax, self.dMax]))
        
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        xstate = self.T @ state.reshape([3,1])
        x1, x2, x3 = xstate
        x1 = x1[0]
        x2 = x2[0]
        x3 = x3[0]
        x4 = x4[0]
        x5 = x5[0]

        fx = self.Tinv @ jnp.array([[0.],
                                    [0.],
                                    [0.],
                                    [0.],
                                    [0.]])

        return fx.reshape([3])

    def control_jacobian(self, state, time):
        xstate = self.T @ state.reshape([2,1])
        x1, x2, x3 = xstate
        x1 = x1[0]
        x2 = x2[0]
        x3 = x3[0]
        x4 = x4[0]
        x5 = x5[0]

        gu = self.Tinv @ jnp.array([[0.],
                                    [0.],
                                    [0.],
                                    [0.],
                                    [0.]])
        
        return gu

    def disturbance_jacobian(self, state, time):
        xstate = self.T @ state.reshape([2,1])
        x1, x2, x3 = xstate
        x1 = x1[0]
        x2 = x2[0]
        x3 = x3[0]
        x4 = x4[0]
        x5 = x5[0]

        gd = self.Tinv @ jnp.array([[0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.]])
        
        return gd