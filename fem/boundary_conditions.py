"""Boundary Conditions for Finite Element Method

This module provides classes for defining various boundary conditions for the finite element method.

"""

import sympy as sp
from .utils import phi
from .printer import display_equation

class DirichletBC:
    """Dirichlet Boundary Condition
    
    Attributes:
        leftbc: Value of the boundary condition at the left boundary.
        rightbc: Value of the boundary condition at the right boundary.

    """
    def __init__(self, g0=None, g1=None):
        self.leftbc = g0
        self.rightbc = g1

class LeftDirichletBC(DirichletBC):
    """Left Dirichlet Boundary Condition
    
    Attributes:
        leftbc: Value of the boundary condition at the left boundary.

    """
    def __init__(self, g0):
        super().__init__(g0=g0)

    def display(self):
        """Display the left boundary condition using LaTeX formatting."""
        y = sp.Function('y')
        display_equation(y(0), self.leftbc) # pylint: disable=E1102 

    def apply(self, **kwargs):
        """Apply the left boundary condition."""
        f_array = kwargs.get("f_array")
        poly = kwargs.get("poly")
        bilinear_form = kwargs.get("bilinear_form")

        def u_hat(i,x,nodes,deriv=False):
            if not deriv:
                return self.leftbc*phi(0,x,nodes)
            return self.leftbc*phi(0,x,nodes,deriv=True)

        f_array[0] += -bilinear_form(u_hat,1,poly,1)

class RightDirichletBC(DirichletBC):
    """Right Dirichlet Boundary Condition
    
    Attributes:
        rightbc: Value of the boundary condition at the right boundary.

    """

    def __init__(self, g1):
        super().__init__(g1=g1)

    def display(self):
        """Display the right boundary condition using LaTeX formatting."""
        y = sp.Function('y')
        display_equation(y(1), self.rightbc) # pylint: disable=E1102 

    def apply(self, **kwargs):
        """Apply the right boundary condition."""
        N = kwargs.get("N")
        f_array = kwargs.get("f_array")
        poly = kwargs.get("poly")
        last = kwargs.get("last")
        bilinear_form = kwargs.get("bilinear_form")

        def u_hat(i,x,nodes,deriv=False):
            if not deriv:
                return self.rightbc*phi(N+1, x, nodes)
            return self.rightbc*phi(N+1, x, nodes, deriv=True)

        f_array[last] += -bilinear_form(u_hat, N, poly, N)

class RobinBC:
    """Robin Boundary Condition
    
    Attributes:
        leftbc: Value of the boundary condition at the left boundary.
        rightbc: Value of the boundary condition at the right boundary.
        beta: Parameter beta.

    """
    def __init__(self, g0=None, g1=None, beta=0):
        self.leftbc = g0
        self.rightbc = g1
        self.beta = beta

class LeftRobinBC(RobinBC):
    """Left Robin Boundary Condition
    
    Attributes:
        leftbc: Value of the boundary condition at the left boundary.
        beta: Parameter beta.

    """
    def __init__(self, g0, beta=0):
        super().__init__(g0=g0, beta=beta)

    def display(self):
        """Display the left boundary condition using LaTeX formatting."""
        x = sp.symbols('x')
        y = sp.Function('y')(x) # pylint: disable=E1102
        lhs = -sp.diff(y, x).subs(x, 0) + self.beta*y.subs(x, 0)
        display_equation(lhs, -1*self.leftbc)

    def apply(self, **kwargs):
        """Apply the left boundary condition."""
        f_array = kwargs.get("f_array")
        A = kwargs.get("A")
        ode = kwargs.get("ode")

        A[0,0] += -ode.a(0)*self.beta
        f_array[0] += ode.a(0)*self.leftbc

class RightRobinBC(RobinBC):
    """Right Robin Boundary Condition
    
    Attributes:
        rightbc: Value of the boundary condition at the right boundary.
        beta: Parameter beta.

    """
    def __init__(self, g1, beta=0):
        super().__init__(g1=g1, beta=beta)

    def display(self):
        """Display the left boundary condition using LaTeX formatting."""
        x = sp.symbols('x')
        y = sp.Function('y')(x) # pylint: disable=E1102
        lhs = sp.diff(y, x).subs(x, 1) + self.beta*y.subs(x, 1)
        display_equation(lhs, self.rightbc)

    def apply(self, **kwargs):
        """Apply the right boundary condition."""
        f_array = kwargs.get("f_array")
        A = kwargs.get("A")
        ode = kwargs.get("ode")
        last = kwargs.get("last")

        A[last,last] += -ode.a(1)*self.beta
        f_array[last] += -ode.a(1)*self.rightbc