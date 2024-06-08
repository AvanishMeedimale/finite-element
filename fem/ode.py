"""Module Docstring
s
"""

import sympy as sp
from .printer import display_equation

class ODE:
    """TBA"""
    def __init__(self, a, b, c, f):
        """Input parameters:
        a
        b
        c
        f
        """
        self.a = a
        self.b = b
        self.c = c
        self.f = f

    def display(self):
        """Display the differential equation using LaTeX formatting."""
        x = sp.symbols('x')
        y = sp.Function('y')(x) # pylint: disable=E1102

        # Construct the ODE in terms of sympy expressions
        ode_lhs = self.a(x)*sp.diff(y, x, x) + self.b(x)*sp.diff(y, x) + self.c(x)*y
        ode_rhs = self.f(x)
        display_equation(ode_lhs, ode_rhs)
