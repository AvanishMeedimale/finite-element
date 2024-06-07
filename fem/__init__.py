from sympy import init_printing
from .printer import MyLatexPrinter
init_printing(latex_printer=lambda *args, **kwargs: MyLatexPrinter().doprint(*args))
