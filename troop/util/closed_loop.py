import math

import numpy as np
import scipy as sp


class ClosedLoopTrajectory:
    """
    Docstring for ClosedLoopTrajectory

    This class represents a closed-loop trajectory for a sample-and-hold controller
    designed from the output of hjpy.
    """

    def __init__(
        self, model, grid, times, value_function, initial_state, steps=100, **kwargs
    ):
        """
        Docstring for __init__

        :param self: ClosedLoopTrajctory object
        :param model: dynamics object from hjpy
        :param grid: spatial grid object from hjpy
        :param times: times over which the value function was computed
        :param value_function: value function tensor
        :param initial_state: numpy vector representing the initial state
        :param steps: (int) number of sample and hold steps
        :param **kwags: optional arguments to be passed to scipy.integrate.solve_ivp
        """
        self._model = model
        self._grid = grid
        if times[0] < times[-1]:
            self._times = times
            self._V = value_function
        elif times[-1] < times[0]:
            self._times = times[::-1]
            self._V = value_function[::-1, ...]
        else:
            raise ValueError("Time axis is improper.")

        self._initial_state = np.array(initial_state)
        self._steps = max(int(steps), 1)

        self._u = None
        self._d = None

        self._us = [None] * self._steps
        self._ds = [None] * self._steps
        self._sols = [None] * self._steps

        self._solve_ivp(**kwargs)

    def x(self, t):
        return self._sols[self._get_sol_index(t)](t)

    def u(self, t):
        return self._us[self._get_sol_index(t)]

    def d(self, t):
        return self._ds[self._get_sol_index(t)]

    def gradient(self, t):
        return self._gradient(t, self.x(t))

    def value(self, t):
        return self._value(t, self.x(t))

    def _gradient(self, t, state):
        j, k = self._get_time_indexes(t)

        if j == k:
            gradient = self._grid.interpolate(
                self._grid.grad_values(self._V[j, ...]), state=state
            )

        else:
            gradient_left = self._grid.interpolate(
                self._grid.grad_values(self._V[j, ...]), state=state
            )
            gradient_right = self._grid.interpolate(
                self._grid.grad_values(self._V[k, ...]), state=state
            )

            gradient = (
                (t - self._times[j]) * gradient_left
                + (self._times[k] - t) * gradient_right
            ) / (self._times[k] - self._times[j])

        return gradient

    def _value(self, t, state):
        j, k = self._get_time_indexes(t)

        if j == k:
            value = self._grid.interpolate(self._V[j, ...], state=state)

        else:
            value_left = self._grid.interpolate(self._V[j, ...], state=state)
            value_right = self._grid.interpolate(self._V[k, ...], state=state)

            value = (
                (t - self._times[j]) * value_left + (self._times[k] - t) * value_right
            ) / (self._times[k] - self._times[j])

        return value

    def _dynamics(self, time, state):
        return self._model.__call__(state, self._u, self._d, time)

    def _get_time_indexes(self, t):
        i = np.searchsorted(self._times, t, side="right") - 1
        if t == self._times[i]:
            return i, i
        else:
            return i, i + 1

    def _get_sol_index(self, t):
        if t == 0:
            return -1
        else:
            return math.floor((1 - t / self._times[0]) * self._steps)

    def _solve_ivp(self, **kwargs):
        state = self._initial_state
        for i in range(self._steps):
            t = ((self._steps - i) / self._steps) * self._times[0]
            t_plus = ((self._steps - i - 1) / self._steps) * self._times[0]

            gradient = self._gradient(t, state)

            self._u = self._model.optimal_control(state, t, gradient)
            self._d = self._model.optimal_disturbance(state, t, gradient)

            sol = sp.integrate.solve_ivp(
                self._dynamics, [t, t_plus], state, dense_output=True, **kwargs
            )

            state = sol.sol(t_plus)
            self._sols[i] = sol.sol
            self._us[i] = self._u
            self._ds[i] = self._d
