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
                 uMin=-1.,
                 dMax=1.,
                 dMin=0):

        if control_space is None:
            center = (uMax + uMin) / 2.
            radius = (uMax - uMin) / 2.
            control_space = sets.Ball(jnp.array([center, center]), radius)
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([dMin]),
                                         jnp.array([dMax]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        
        return jnp.array([0., 0.])

    def control_jacobian(self, state, time):

        gu = jnp.array([[1., 0.],
                        [0., 1.]])
        
        return gu

    def disturbance_jacobian(self, state, time):

        gd = jnp.array([[1.],
                        [0.]])
        
        return gd