"""This script demonstrates the usage of the Finite Element Method (FEM) solver application.

It initializes an ordinary differential equation (ODE) and sets up boundary conditions for a
one-dimensional FEM problem. It then solves the FEM problem using the FEMSolver class and plots
the numerical solution alongside the exact solution, if available.

Functions:
  - a_func(x): Returns the coefficient 'a' in the ODE.
  - b_func(x): Returns the coefficient 'b' in the ODE.
  - c_func(x): Returns the coefficient 'c' in the ODE.
  - f_func(x): Returns the function 'f' in the ODE.
  - u_exact(x): Returns the exact solution of the ODE, if available.

Dependencies:
  - numpy (np)
  - matplotlib.pyplot (plt)
  - fem.fem_solver.FEMSolver
  - fem.ode.ODE
  - fem.boundary_conditions.LeftDirichletBC
  - fem.boundary_conditions.RightDirichletBC
  - fem.boundary_conditions.LeftRobinBC
  - fem.boundary_conditions.RightRobinBC

The script begins by defining the coefficients and functions necessary to construct the ODE. 
It then creates an instance of the ODE class and displays its equation. Following this,
left and right boundary conditions are specified using the LeftDirichletBC and RightDirichletBC
classes, respectively, and their equations are displayed. Next, the script initializes the FEMSolver
with the ODE and boundary conditions, solves the problem, and plots the numerical solution. 
Additionally, if available, the exact solution is plotted for comparison.

For more information on the individual functions and classes, refer to their respective docstrings.
"""

import numpy as np
import matplotlib.pyplot as plt
from fem.fem_solver import FEMSolver
from fem.ode import ODE
from fem.boundary_conditions import LeftDirichletBC, RightDirichletBC, LeftRobinBC, RightRobinBC

def a_func(x):
    return -1

def b_func(x):
    return 0

def c_func(x):
    return 1

def f_func(x):
    return 1

ode = ODE(a_func, b_func, c_func, f_func)
ode.display()

bc1 = LeftDirichletBC(0)
bc2 = RightDirichletBC(0)
bc1.display()
bc2.display()

solution = FEMSolver(ode,bc1,bc2,0)
solution.plot_solution()

def u_exact(x):
    """Returns the exact solution of the ODE to compare with the finite element solution."""
    e = np.e
    return ((1/e-1)/(e-1/e)*e**x-(e-1)/(e-1/e)*e**(-x))+1 # Solved analytically

xes = np.linspace(0,1,200)
plt.plot(xes, [u_exact(x) for x in xes], label = "Exact Solution")
plt.legend()
