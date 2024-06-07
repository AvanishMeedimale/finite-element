import sympy as sp
from .utils import phi
from .printer import display_equation

class DirichletBC:
    def __init__(self, g0=None, g1=None):
        self.leftbc = g0
        self.rightbc = g1

class LeftDirichletBC(DirichletBC):
    def __init__(self, g0):
        super().__init__(g0=g0)

    def display(self):
        """
        Display the left boundary condition using LaTeX formatting.
        """
        y = sp.Function('y')
        display_equation(y(0), self.leftbc)

    def apply(self, nodes, f_array, A, ode, poly=phi, last=None, bilinear_form=None):
        N = len(nodes) - 2
        def u_hat(i,x,nodes,deriv=False):
            if not deriv:
                return self.leftbc*phi(0,x,nodes)
            else:
                return self.leftbc*phi(0,x,nodes,deriv=True)

        f_array[0] += -bilinear_form(u_hat,1,poly,1)

class RightDirichletBC(DirichletBC):
    def __init__(self, g1):
        super().__init__(g1=g1)

    def display(self):
        """
        Display the right boundary condition using LaTeX formatting.
        """
        y = sp.Function('y')
        display_equation(y(1), self.rightbc)

    def apply(self, nodes, f_array, A, ode, poly=phi, last=None, bilinear_form=None):
        N = len(nodes) - 2
        def u_hat(i,x,nodes,deriv=False):
            if not deriv:
                return self.rightbc*phi(N+1,x,nodes)
            else:
                return self.rightbc*phi(N+1,x,nodes,deriv=True)

        f_array[last] += -bilinear_form(u_hat,N,poly,N)


class RobinBC:
    def __init__(self, g0=None, g1=None, beta=0):
        self.leftbc = g0
        self.rightbc = g1
        self.beta = beta

class LeftRobinBC(RobinBC):
    def __init__(self, g0, beta=0):
        super().__init__(g0=g0, beta=beta)

    def display(self):
        """
        Display the left boundary condition using LaTeX formatting.
        """
        x = sp.symbols('x')
        y = sp.Function('y')(x) # pylint: disable=E1102
        lhs = -sp.diff(y, x).subs(x, 0) + self.beta*y.subs(x, 0)
        display_equation(lhs, -1*self.leftbc)

    def apply(self, nodes, f_array, A, ode, poly, last, bilinear_form):
        A[0,0] += -ode.a(0)*self.beta
        f_array[0] += ode.a(0)*self.leftbc

class RightRobinBC(RobinBC):
    def __init__(self, g1, beta=0):
        super().__init__(g1=g1, beta=beta)

    def display(self):
        """
        Display the left boundary condition using LaTeX formatting.
        """
        x = sp.symbols('x')
        y = sp.Function('y')(x) # pylint: disable=E1102
        lhs = sp.diff(y, x).subs(x, 1) + self.beta*y.subs(x, 1)
        display_equation(lhs, self.rightbc)

    def apply(self, nodes, f_array, A, ode, poly, last, bilinear_form):
        A[last,last] += -ode.a(1)*self.beta
        f_array[last] += -ode.a(1)*self.rightbc