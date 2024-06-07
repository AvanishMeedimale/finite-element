import sympy as sp

try:
    from IPython.display import display
except ImportError:
    def display(equation):
        print(f"LaTeX equation: {sp.latex(equation)}")

from sympy.printing.latex import LatexPrinter
from sympy.core.function import UndefinedFunction, AppliedUndef

class MyLatexPrinter(LatexPrinter):
    def _print_Derivative(self, expr):
        # override default print for single variable derivative
        # basically if derivative w.r.t 1 variable, show y''', else default _print_Derivative
        function, *vars = expr.args
        if isinstance(type(function), UndefinedFunction) and len(vars) == 1:
            return r'%s%s' % (function.func.__name__, "'"*vars[0][1])
        return super()._print_Derivative(expr)

    def _print_Function(self, expr, exp=None):
        # same as above but for functions, to display y instead of default y(x)
        vars = expr.args
        if isinstance(expr, AppliedUndef) and len(vars) == 1 and isinstance(vars[0], sp.Symbol):
            if exp is None:
                return expr.func.__name__
            else:
                return r'%s^{%s}' % (expr.func.__name__, exp)
        return super()._print_Function(expr, exp)

def display_equation(lhs, rhs):
    equation = sp.Eq(lhs, rhs)
    display(equation)
