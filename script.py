from fem.fem_solver import FEMSolver
from fem.ode import ODE
from fem.boundary_conditions import *
import numpy as np
import matplotlib.pyplot as plt

def a_func(x):
    return -1

def b_func(x):
    return 0

def c_func(x):
    return 1

def f_func(x):
    return 1

ode = ODE(a_func, b_func, c_func, f_func)

bc1 = LeftDirichletBC(0)
bc2 = RightDirichletBC(0)

solution = FEMSolver(ode,bc1,bc2,0)
solution.plot_solution()

def u_exact(x):
    e = np.e
    return ((1/e-1)/(e-1/e)*e**x-(e-1)/(e-1/e)*e**(-x))+1 # Solved analytically
xes = np.linspace(0,1,200)
plt.plot(xes, [u_exact(x) for x in xes], label = "Exact Solution")
plt.legend()