from scipy import integrate
import sympy as sp
from fem.utils import phi

class DirichletBC:
    def __init__(self, g0=None, g1=None):
        self.leftbc = g0
        self.rightbc = g1

class LeftDirichletBC(DirichletBC):
    def __init__(self, g0):
        super().__init__(g0=g0)

    def display_equation(self):
        """
        Display the left boundary condition using LaTeX formatting.
        """
        y = sp.Function('y')
        equation = sp.Eq(y(0), self.leftbc) # pylint: disable=E1102 
        
        try:
            from IPython.display import display, Math
            display(Math(sp.latex(equation)))
        except ImportError:
            print(f"LaTeX equation: {sp.latex(equation)}")

    def apply(self, nodes, f_array, A=None, ode=None, poly=phi, last=None):
        N = len(nodes) - 2
        def bilinear_form(u,i,v,j):
            return integrate.quad(lambda x: -1*ode.a(x)*u(i,x,nodes,deriv=True)*v(j,x,nodes,deriv=True) + ode.b(x)*u(i,x,nodes,deriv=True)*v(j,x,nodes) + ode.c(x)*u(i,x,nodes)*v(j,x,nodes), 0, 1)[0]
        
        def u_hat(i,x,nodes,deriv=False):
            if deriv == False:
                return self.leftbc*phi(0,x,nodes)
            if deriv:
                return self.leftbc*phi(0,x,nodes,deriv=True)
        
        f_array[0] += -bilinear_form(u_hat,1,poly,1)

class RightDirichletBC(DirichletBC):
    def __init__(self, g1):
        super().__init__(g1=g1)

    def display_equation(self):
        """
        Display the right boundary condition using LaTeX formatting.
        """
        y = sp.Function('y')
        equation = sp.Eq(y(1), self.rightbc) # pylint: disable=E1102 
        
        try:
            from IPython.display import display, Math
            display(Math(sp.latex(equation)))
        except ImportError:
            print(f"LaTeX equation: {sp.latex(equation)}")

    def apply(self, nodes, f_array, A=None, ode=None, poly=phi, last=None):
        N = len(nodes) - 2
        def bilinear_form(u,i,v,j):
            return integrate.quad(lambda x: -1*ode.a(x)*u(i,x,nodes,deriv=True)*v(j,x,nodes,deriv=True) + ode.b(x)*u(i,x,nodes,deriv=True)*v(j,x,nodes) + ode.c(x)*u(i,x,nodes)*v(j,x,nodes), 0, 1)[0]
        def u_hat(i,x,nodes,deriv=False):
            if deriv == False:
                return self.rightbc*phi(N+1,x,nodes)
            if deriv:
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

    def display_equation(self):
        """
        Display the left boundary condition using LaTeX formatting.
        """
        x = sp.symbols('x')
        y = sp.Function('y')(x) # pylint: disable=E1102 
        lhs = -sp.diff(y, x).subs(x, 0) + self.beta*y.subs(x, 0)
        equation = sp.Eq(lhs, -1*self.leftbc)
        
        try:
            from IPython.display import display, Math
            display(Math(sp.latex(equation)))
        except ImportError:
            print(f"LaTeX equation: {sp.latex(equation)}")
 
    def apply(self, nodes, f_array, A, ode, poly=None, last=None):
        A[0,0] += -ode.a(0)*self.beta
        f_array[0] += ode.a(0)*self.leftbc

class RightRobinBC(RobinBC):
    def __init__(self, g1, beta=0):
        super().__init__(g1=g1, beta=beta)

    def display_equation(self):
        """
        Display the left boundary condition using LaTeX formatting.
        """
        x = sp.symbols('x')
        y = sp.Function('y')(x) # pylint: disable=E1102 
        lhs = sp.diff(y, x).subs(x, 1) + self.beta*y.subs(x, 1)
        equation = sp.Eq(lhs, self.rightbc)
        
        try:
            from IPython.display import display, Math
            display(Math(sp.latex(equation)))
        except ImportError:
            print(f"LaTeX equation: {sp.latex(equation)}")

    def apply(self, nodes, f_array, A, ode, poly, last):
        A[last,last] += -ode.a(1)*self.beta
        f_array[last] += -ode.a(1)*self.rightbc