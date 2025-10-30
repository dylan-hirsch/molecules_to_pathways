import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets

##  (Nonlinear) Reduced Model for the Toggle Switch and Repressilator 
class model(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None,
                 uMax=1.,
                 uMin=0.,
                 dMax=1.,
                 dMin=1.):

        if control_space is None:
            control_space = sets.Box(jnp.array([uMin]), jnp.array([uMax]))
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([dMin]), jnp.array([dMax]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x1, x2, x3 = state
        return jnp.array([.5 * x2**4 / (x2**4 + .5**4), - 2 * x2, x2 - .3 * x3])

    def control_jacobian(self, state, time):

        gu = jnp.array([[0.,],
                        [1.,],
                        [0.]])
        
        return gu

    def disturbance_jacobian(self, state, time):
        _, _, x3 = state
        gd = jnp.array([[0.],
                        [0.],
                        [0.]])
        
        return gd