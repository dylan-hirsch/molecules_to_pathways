import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets

## This is the dynamics file for a biomolecular circuit consisting of a repressilator and a toggle switch.
## The structure of the biological circuit is (x1 --| x2 --| x3 --| x1) and (x4--|x5--|x4)
## There is also activation from x2 to x4 and x3 to x5

T = jnp.array([[0.     , 0.    , -2.5820, 0.     , -1.8257],
               [0.     , 0.    ,  1.2910,  2.2339, -1.8257],
               [0.     , 0.    ,  1.2910, -2.2339, -1.8257],
               [-0.2659, 4.7168,  0.    , 0.     , 0.],
               [ 0.2659, 4.7168,  0.    , 0.     , 0.]])
Tinv = jnp.linalg.pinv(T)

K1 = 0.25
K2 = 0.25
K3 = 0.25
K4 = 0.25
K5 = 0.25

n1 = 4.
n2 = 4.
n3 = 4.
n4 = 2.
n5 = 2.

def mm(x, K, n):
    return 1. / (1. + (abs(x)/K)**n)

##  (Nonlinear) Reduced Model for the Toggle Switch and Repressilator 
class reduced_model(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None,
                 rank=3,
                 uMax=1.,
                 uMin=0.,
                 dMax=1.05,
                 dMin=.95):
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        self.T = T[:,0:rank]
        self.Tinv = Tinv[0:rank,:]
        self.rank = rank

        if control_space is None:
            control_space = sets.Box(jnp.array([self.uMin]), jnp.array([self.uMax]))
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([self.dMin, self.dMin, self.dMin, self.dMin, self.dMin]),
                                         jnp.array([self.dMax, self.dMax, self.dMax, self.dMax, self.dMax]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        xstate = self.T @ state.reshape([self.rank,1])
        x1, x2, x3, x4, x5 = xstate
        x1 = x1[0]
        x2 = x2[0]
        x3 = x3[0]
        x4 = x4[0]
        x5 = x5[0]

        fx = self.Tinv @ jnp.array([[mm(x3, K3, n3) - x1],
                                    [mm(x1, K1, n1) - x2],
                                    [mm(x2, K2, n2) - x3],
                                    [mm(x5, K5, n5) - x4],
                                    [mm(x4, K4, n4) - x5]])
        
        return fx.reshape([self.rank])

    def control_jacobian(self, state, time):
        xstate = self.T @ state.reshape([self.rank,1])
        x1, x2, x3, x4, x5 = xstate
        x1 = x1[0]
        x2 = x2[0]
        x3 = x3[0]
        x4 = x4[0]
        x5 = x5[0]

        gu = self.Tinv @ jnp.array([[0.],
                                    [0.],
                                    [0.],
                                    [mm(x2, K2, n2)],
                                    [mm(x3, K3, n3)]])
        
        return gu

    def disturbance_jacobian(self, state, time):
        xstate = self.T @ state.reshape([self.rank,1])
        x1, x2, x3, x4, x5 = xstate
        x1 = x1[0]
        x2 = x2[0]
        x3 = x3[0]
        x4 = x4[0]
        x5 = x5[0]

        gd = self.Tinv @ jnp.array([[mm(x3, K3, n3), 0., 0., 0., 0.],
                                    [0., mm(x1, K1, n1), 0., 0., 0.],
                                    [0., 0., mm(x2, K2, n2), 0., 0.],
                                    [0., 0., 0., mm(x5, K5, n5), 0.],
                                    [0., 0., 0., 0., mm(x4, K4, n4)]])
        
        return gd