"""Module for defining Ordinary Differential Equations (ODEs)

This module provides a class 'ODE' to define and display ordinary differential equations (ODEs).
"""

import sympy as sp
from .printer import display_equation

class ODE:
    """Ordinary Differential Equation (ODE)

    This class represents an ordinary differential equation of the form:
    a(x) * y''(x) + b(x) * y'(x) + c(x) * y(x) = f(x).

    Attributes:
        a (function): Coefficient function for y''(x).
        b (function): Coefficient function for y'(x).
        c (function): Coefficient function for y(x).
        f (function): Right-hand side function of the ODE.
    """
    def __init__(self, a, b, c, f):
        """
        Initialize the ODE with given coefficient functions.

        Args:
            a (function): Coefficient function for y''(x).
            b (function): Coefficient function for y'(x).
            c (function): Coefficient function for y(x).
            f (function): Right-hand side function of the ODE.
        """
        self.a = a
        self.b = b
        self.c = c
        self.f = f

    def display(self):
        """Display the differential equation using LaTeX formatting."""
        x = sp.symbols('x')
        y = sp.Function('y')(x)  # pylint: disable=E1102

        # Construct the ODE in terms of sympy expressions
        ode_lhs = self.a(x) * sp.diff(y, x, x) + self.b(x) * sp.diff(y, x) + self.c(x) * y
        ode_rhs = self.f(x)
        display_equation(ode_lhs, ode_rhs)
