import sympy as sp

class ODE:
    def __init__(self, a, b, c, f):
        self.a = a
        self.b = b
        self.c = c
        self.f = f

    def display_equation(self):
        """
        Display the differential equation using LaTeX formatting.
        """
        x = sp.symbols('x')
        y = sp.Function('y')(x)
        
        # Construct the ODE in terms of sympy expressions
        ode_lhs = self.a(x)*sp.diff(y, x, x) + self.b(x)*sp.diff(y, x) + self.c(x)*y
        ode_rhs = self.f(x)
        equation = sp.Eq(ode_lhs, ode_rhs)
        
        try:
            from IPython.display import display, Math
            display(Math(sp.latex(equation)))
        
        except ImportError:
            print(f"LaTeX equation: {sp.latex(equation)}")
    