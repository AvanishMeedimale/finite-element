"""Boundary Conditions for Finite Element Method

This module provides classes for defining various boundary conditions for the finite element method.

Classes:
    DirichletBC: Base class for Dirichlet boundary conditions.
    LeftDirichletBC: Implements the left Dirichlet boundary condition.
    RightDirichletBC: Implements the right Dirichlet boundary condition.
    RobinBC: Base class for Robin boundary conditions.
    LeftRobinBC: Implements the left Robin boundary condition.
    RightRobinBC: Implements the right Robin boundary condition.
"""

import sympy as sp
from .basis_functions import phi
from .printer import display_equation

class DirichletBC:
    """Base class for Dirichlet boundary conditions.
    
    Attributes:
        leftbc: Value of the boundary condition at the left boundary.
        rightbc: Value of the boundary condition at the right boundary.

    Public Methods:
        __init__(g0=None, g1=None): Initializes the boundary conditions.
        display(): Display the left and right boundary conditions using LaTeX formatting.
    """
    def __init__(self, g0=0, g1=0):
        """Initialize the Dirichlet boundary conditions.
        
        Args:
            g0: Value at the left boundary (default is 0).
            g1: Value at the right boundary (default is 0).
        """
        self.leftbc = g0
        self.rightbc = g1

    def display(self):
        """Display the left and right boundary conditions using LaTeX formatting."""
        y = sp.Function('y')
        display_equation(y(0), self.leftbc) # pylint: disable=E1102
        display_equation(y(1), self.rightbc) # pylint: disable=E1102

class LeftDirichletBC(DirichletBC):
    """Implements the left Dirichlet boundary condition.
    
    Attributes:
        leftbc: Value of the boundary condition at the left boundary.

    Public Methods:
        __init__(g0): Initializes the left Dirichlet boundary condition.
        display(): Display the left boundary condition using LaTeX formatting.
        apply(**kwargs): Apply the left boundary condition to the stress matrix and load vector.
    """
    def __init__(self, g0):
        """Initialize the left Dirichlet boundary condition.
        
        Args:
            g0: Value at the left boundary.
        """
        super().__init__(g0=g0)

    def display(self):
        """Display the left boundary condition using LaTeX formatting."""
        y = sp.Function('y')
        display_equation(y(0), self.leftbc) # pylint: disable=E1102

    def apply(self, **kwargs):
        """Apply the left boundary condition to the stress matrix and load vector.
        
        Args:
            f_array: Array representing the load vector.
            poly: Polynomial order for the chosen basis functions. 0 represents linear and 
                  1 represents quadratic (default is 0).
            bilinear_form: Bilinear form of the differential equation's weak form.
        """
        f_array = kwargs.get("f_array")
        poly = kwargs.get("poly")
        bilinear_form = kwargs.get("bilinear_form")

        def u_hat(i,x,nodes,deriv=False):
            if not deriv:
                return self.leftbc*phi(0,x,nodes)
            return self.leftbc*phi(0,x,nodes,deriv=True)

        f_array[0] += -bilinear_form(u_hat,1,poly,1)

class RightDirichletBC(DirichletBC):
    """Implements the right Dirichlet boundary condition.
    
    Attributes:
        rightbc: Value of the boundary condition at the right boundary.

    Public Methods:
        __init__(g1): Initializes the right Dirichlet boundary condition.
        display(): Display the right boundary condition using LaTeX formatting.
        apply(**kwargs): Apply the right boundary condition to the stress matrix and load vector.
    """
    def __init__(self, g1):
        super().__init__(g1=g1)

    def display(self):
        """Display the right boundary condition using LaTeX formatting."""
        y = sp.Function('y')
        display_equation(y(1), self.rightbc) # pylint: disable=E1102 

    def apply(self, **kwargs):
        """Apply the right boundary condition to the stress matrix and load vector.
        
        Args:
            N: Number of elements.
            f_array: Array representing the load vector.
            poly: Polynomial order for the chosen basis functions. 0 represents linear and 
                  1 represents quadratic (default is 0).
            last: Final index value of the linear basis functions in the load vector.
            bilinear_form: Bilinear form of the differential equation's weak form.
        """
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
    """Base class for Robin boundary conditions.
    
    Attributes:
        leftbc: Value of the boundary condition at the left boundary.
        rightbc: Value of the boundary condition at the right boundary.

    Public Methods:
        __init__(g0=None, g1=None): Initializes the boundary conditions.
        display(): Display the left and right boundary conditions using LaTeX formatting.
    """
    def __init__(self, g0=0, g1=0, beta=0):
        """Initialize the Dirichlet boundary conditions.
        
        Args:
            g0: Value at the left boundary (default is 0).
            g1: Value at the right boundary (default is 0).
            beta: Robin condition parameter beta (default is 0 where it becomes Neumann condition).
        """
        self.leftbc = g0
        self.rightbc = g1
        self.beta = beta

    def display(self):
        """Display the left and right boundary conditions using LaTeX formatting."""
        x = sp.symbols('x')
        y = sp.Function('y')(x) # pylint: disable=E1102
        lhs_left = -sp.diff(y, x).subs(x, 0) + self.beta*y.subs(x, 0)
        display_equation(lhs_left, -1*self.leftbc)
        lhs_right = sp.diff(y, x).subs(x, 1) + self.beta*y.subs(x, 1)
        display_equation(lhs_right, self.rightbc)

class LeftRobinBC(RobinBC):
    """Implements the left Robin boundary condition.
    
    Attributes:
        leftbc: Value of the boundary condition at the left boundary.
        beta: Robin condition parameter beta.

    Public Methods:
        __init__(g0, beta=0): Initializes the left Robin boundary condition.
        display(): Display the left boundary condition using LaTeX formatting.
        apply(**kwargs): Apply the left boundary condition to the stress matrix and load vector.
    """
    def __init__(self, g0, beta=0):
        """Initialize the left Robin boundary condition.
        
        Args:
            g0: Value at the left boundary.
            beta: Robin condition parameter beta (default is 0 where it becomes Neumann condition).
        """
        super().__init__(g0=g0, beta=beta)

    def display(self):
        """Display the left boundary condition using LaTeX formatting."""
        x = sp.symbols('x')
        y = sp.Function('y')(x) # pylint: disable=E1102
        lhs = -sp.diff(y, x).subs(x, 0) + self.beta*y.subs(x, 0)
        display_equation(lhs, -1*self.leftbc)

    def apply(self, **kwargs):
        """Apply the right boundary condition to the stress matrix and load vector.
        
        Args:
            f_array: Array representing the load vector.
            A: Matrix representing the stress matrix.
            ode: Object representing the ordinary differential equation.
        """
        f_array = kwargs.get("f_array")
        A = kwargs.get("A")
        ode = kwargs.get("ode")

        A[0,0] += -ode.a(0)*self.beta
        f_array[0] += ode.a(0)*self.leftbc

class RightRobinBC(RobinBC):
    """Right Robin Boundary Condition
    
    Attributes:
        rightbc: Value of the boundary condition at the right boundary.
        beta: Robin condition parameter beta.
    """
    def __init__(self, g1, beta=0):
        """
        Initialize the right Robin boundary condition.
        
        Args:
            g1: Value at the right boundary.
            beta: Robin condition parameter beta (default is 0 where it becomes Neumann condition).
        """
        super().__init__(g1=g1, beta=beta)

    def display(self):
        """Display the left boundary condition using LaTeX formatting."""
        x = sp.symbols('x')
        y = sp.Function('y')(x) # pylint: disable=E1102
        lhs = sp.diff(y, x).subs(x, 1) + self.beta*y.subs(x, 1)
        display_equation(lhs, self.rightbc)

    def apply(self, **kwargs):
        """Apply the right boundary condition to the stress matrix and load vector.
        
        Args:
            f_array: Array representing the load vector.
            A: Matrix representing the stress matrix.
            ode: Object representing the ordinary differential equation.
            last: Final index value of the linear basis functions in the load vector.
        """
        f_array = kwargs.get("f_array")
        A = kwargs.get("A")
        ode = kwargs.get("ode")
        last = kwargs.get("last")

        A[last,last] += -ode.a(1)*self.beta
        f_array[last] += -ode.a(1)*self.rightbc
