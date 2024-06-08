"""This module contains functions to test the FEMSolver class using various boundary conditions and problem setups."""

import numpy as np
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

def test_homogenous_dirichlet_linear():
    bc1 = LeftDirichletBC(0)
    bc2 = RightDirichletBC(0)
    solution = FEMSolver(ode,bc1,bc2,0)
    uh = solution.solve()
    U = np.array([0.04131624, 0.07302963, 0.09545784, 0.10882553, 0.1132666, 
                  0.10882553, 0.09545784, 0.07302963, 0.04131624])
    assert np.allclose(U,uh)

def test_homogenous_dirichlet_quadratic():
    bc1 = LeftDirichletBC(0)
    bc2 = RightDirichletBC(0)
    solution = FEMSolver(ode,bc1,bc2,1)
    uh = solution.solve()
    U = np.array([ 0.04131624,  0.07302963,  0.09545784,  0.10882553,  0.1132666 ,
        0.10882553,  0.09545784,  0.07302963,  0.04131624, -0.99900117,
       -0.99900016, -0.99900104, -0.99900016, -0.99900117, -0.99900117,
       -0.99900016, -0.99900104, -0.99900016, -0.99900117])
    assert np.allclose(U,uh)

def test_robin_linear():
    bc1 = LeftRobinBC(1,1)
    bc2 = RightRobinBC(0,1)
    solution = FEMSolver(ode,bc1,bc2,0)
    uh = solution.solve()
    U = np.array([-0.18368291, -0.10784322, -0.04310046,  0.01119389,  0.05558367,
        0.09051352,  0.11633332,  0.1333017 ,  0.14158863,  0.14127712,
        0.13236404])
    assert np.allclose(U,uh)

def test_robin_quadratic():
    bc1 = LeftRobinBC(1,1)
    bc2 = RightRobinBC(0,1)
    solution = FEMSolver(ode,bc1,bc2,1)
    uh = solution.solve()
    U = np.array([-0.18368291, -0.10784322, -0.04310046,  0.01119389,  0.05558367,
        0.09051352,  0.11633332,  0.1333017 ,  0.14158863,  0.14127712,
        0.13236404, -0.99900117, -0.99900016, -0.99900104, -0.99900016,
       -0.99900117, -0.99900117, -0.99900016, -0.99900104, -0.99900016,
       -0.99900117])
    assert np.allclose(U,uh)

def test_robinplusdiri_linear():
    bc1 = LeftRobinBC(2)
    bc2 = RightDirichletBC(5)
    solution = FEMSolver(ode,bc1,bc2,0)
    uh = solution.solve()
    U = np.array([2.06848945, 2.27417471, 2.49262298, 2.72602241, 2.97671087,
       3.24719944, 3.54019752, 3.85863999, 4.20571657, 4.58490384])
    assert np.allclose(U,uh)

def test_diriplusrobin_linear():
    bc1 = LeftDirichletBC(1)
    bc2 = RightRobinBC(3)
    solution = FEMSolver(ode,bc1,bc2,0)
    uh = solution.solve()
    U = np.array([1.19467899, 1.39130801, 1.59185665, 1.79833374, 2.01280749,
       2.23742622, 2.47443988, 2.72622255, 2.99529626, 3.28435624])
    assert np.allclose(U,uh)
