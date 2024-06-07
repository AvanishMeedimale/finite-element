import sympy as sp
from .printer import display_equation

class ODE:
    def __init__(self, a, b, c, f):
        self.a = a
        self.b = b
        self.c = c
        self.f = f

    def display(self):
        """
        Display the differential equation using LaTeX formatting.
        """
        x = sp.symbols('x')
        y = sp.Function('y')(x)

        # Construct the ODE in terms of sympy expressions
        ode_lhs = self.a(x)*sp.diff(y, x, x) + self.b(x)*sp.diff(y, x) + self.c(x)*y
        ode_rhs = self.f(x)
        display_equation(ode_lhs, ode_rhs)
