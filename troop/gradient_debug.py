import numpy as np
import troop_v2 as troop

""""
n = 3  # FOM size
r = 2  # ROM size

d = 1  # input size
m = 1  # output size

L = 100  # number of time steps
T = 10  # final time

def f(x, u):
    x1 = x[0]
    x2 = x[1]
    theta = x[2]
    return np.array([np.cos(theta), np.sin(theta), u]).reshape((3,))


def g(x):
    x1 = x[0]
    x2 = x[1]
    return np.sqrt(np.array([x1**2 + x2**2])).reshape((1,))


def dfdx(x, u):
    x1 = x[0]
    x2 = x[1]
    theta = x[2]
    return np.array(
        [[0, 0, -np.sin(theta)], [0, 0, np.cos(theta)], [0, 0, 0]]
    ).reshape((3, 3))


def dgdx(x):
    x1 = x[0]
    x2 = x[1]
    return np.array([2 * x1, 2 * x2, 0]).reshape((1, 3))
"""

n = 3  # FOM size
r = 2  # ROM size

d = 1  # input size
m = 1  # output size

L = 100  # number of time steps
T = 10  # final time

A = np.array([[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -5.0]])
B = np.array([[1.0], [1.0], [1.0]])
C = np.array([[1.0, 1.0, 1.0]])

def f(x, u):
    return A @ x + B @ u


def g(x):
    return C @ x


def dfdx(x, u):
    return A

def dgdx(x):
    return C

trooper = troop.troop(n, r, d, m, f, g, dfdx, dgdx, U = lambda t: np.ones((1,)), T = 5.0, L = 11)
grad_Phi, grad_Psi = trooper.compute_gradient()