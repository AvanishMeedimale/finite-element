"""Finite Element Method Solver

This module provides a class 'FEMSolver' for solving differential equations using 
the finite element method.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from .boundary_conditions import LeftDirichletBC, RightDirichletBC
from .basis_functions import phi, psi

class FEMSolver:
    """Solver for Differential Equations using Finite Element Method
    
    Attributes:
        nodes: Array of nodes for the finite element mesh.
        quadratic: Quadratic term used in calculations.
        is_left_dirichlet: Flag indicating if the left boundary condition is Dirichlet.
        is_right_dirichlet: Flag indicating if the right boundary condition is Dirichlet.
        start_range: Start index for the linear range.
        linear_end_range: End index for the linear range.
        linear_range: Range of linear elements.
    """
    def __init__(self, ode, leftbc, rightbc, poly_order=0, N=9):
        """Initialize FEMSolver
        
        Args:
            ode: Object representing the ordinary differential equation.
            leftbc: Boundary condition object representing the left boundary condition.
            rightbc: Boundary condition object representing the right boundary condition.
            poly_order: Polynomial order for the chosen basis functions (default 0 for linear and 1 for quadratic).
            N: Number of elements (default 9).
        
        """
        self.ode = ode
        self.leftbc = leftbc
        self.rightbc = rightbc
        self.poly_order = poly_order # 0 for linear and 1 for quadratic
        self.N = N

        self.nodes = np.linspace(0, 1, self.N + 2)
        self.quadratic = self.poly_order * (self.N + 1)
        self.is_left_dirichlet = isinstance(self.leftbc, LeftDirichletBC)
        self.is_right_dirichlet = isinstance(self.rightbc, RightDirichletBC)
        self.start_range = 1 if self.is_left_dirichlet else 0
        self.linear_end_range = self.N + 1 if self.is_right_dirichlet else self.N + 2
        self.linear_range = range(self.start_range, self.linear_end_range)

    def solve(self):
        """Solve the differential equation using Finite Element Method
        
        Returns:
            Array: Solution to the linear system containing stress matrix A and load vector f
        
        """
        k = self.N + 2 + self.quadratic
        k -= int(self.is_left_dirichlet) + int(self.is_right_dirichlet)

        quadratic_range = range(self.start_range, self.linear_end_range + self.quadratic)

        def bilinear_form(u,i,v,j):
            return integrate.quad(lambda x: -1*self.ode.a(x)*u(i,x,self.nodes,deriv=True)*v(j,x,self.nodes,deriv=True) 
                                  + self.ode.b(x)*u(i,x,self.nodes,deriv=True)*v(j,x,self.nodes) 
                                  + self.ode.c(x)*u(i,x,self.nodes)*v(j,x,self.nodes), 0, 1)[0]

        def linear_functional(v,i):
            return integrate.quad(lambda x: self.ode.f(x)*v(i,x, self.nodes), 0, 1)[0]

        def sigma(i, x, nodes, deriv=False):
            if i < self.linear_end_range:
                return phi(i, x, nodes, deriv=deriv)
            return psi(i + 1 - self.linear_end_range, x, nodes, deriv=deriv)

        A = np.zeros((k, k))
        f_array = np.empty(k)

        for index, i in enumerate(quadratic_range):
            A[index, index] = bilinear_form(sigma, i, sigma, i)
            if index < len(quadratic_range) - 1:
                A[index, index + 1] = bilinear_form(sigma, i + 1, sigma, i)
                A[index + 1, index] = bilinear_form(sigma, i, sigma, i + 1)
            f_array[index] = linear_functional(sigma, i)

        apply_kwargs = {
            "N": self.N,
            "nodes": self.nodes,
            "f_array": f_array,
            "A": A,
            "ode": self.ode,
            "poly": sigma,
            "last": len(self.linear_range) - 1,
            "bilinear_form": bilinear_form
        }

        self.leftbc.apply(**apply_kwargs)
        self.rightbc.apply(**apply_kwargs)

        U = np.linalg.solve(A, f_array)
        return U

    def plot_solution(self):
        """Plot the solution obtained from solving the differential equation"""

        U = self.solve()

        def u_h(x):
            approx = 0
            for index, i in enumerate(self.linear_range):
                approx += U[index] * phi(i, x, self.nodes)
            if self.quadratic != 0:
                for i in range(1, self.N + 2):
                    approx += U[len(self.linear_range) + i - 1] * psi(i, x, self.nodes)
            if self.is_left_dirichlet:
                approx += self.leftbc.leftbc * phi(0, x, self.nodes)
            if self.is_right_dirichlet:
                approx += self.rightbc.rightbc * phi(self.N + 1, x, self.nodes)
            return approx

        xes = np.linspace(0,1,200)
        plt.plot(xes, [u_h(x) for x in xes], label = "FE Solution")
