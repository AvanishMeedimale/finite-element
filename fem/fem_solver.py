import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from .boundary_conditions import *
from .utils import *

class FEMSolver:
    def __init__(self, ode, leftbc, rightbc, poly_order=0, N=9):
        self.ode = ode
        self.leftbc = leftbc
        self.rightbc = rightbc
        self.poly_order = poly_order # 0 for linear and 1 for quadratic
        self.N = N

    def solve(self):
        quadratic = self.poly_order*(self.N+1)
        k = self.N+2 + quadratic
        k -= int(isinstance(self.leftbc, LeftDirichletBC)) + int(isinstance(self.rightbc, RightDirichletBC))
        testvariable = self.N+1 if isinstance(self.rightbc, RightDirichletBC) else self.N+2
        size = range(1 if isinstance(self.leftbc, LeftDirichletBC) else 0, self.N+1 if isinstance(self.rightbc, RightDirichletBC) else self.N+2) # (1,N+1) if dirichlet (0,N+2) if robin
        size2 = range(1 if isinstance(self.leftbc, LeftDirichletBC) else 0, self.N if isinstance(self.rightbc, RightDirichletBC) else self.N+1) # (1,N) if dirichlet (0,N+1) if robin
        size3 = range(1 if isinstance(self.leftbc, LeftDirichletBC) else 0, self.N+1 + quadratic if isinstance(self.rightbc, RightDirichletBC) else self.N+2 + quadratic) # (1,2N+2) if dirichlet (0,2N+3) if robin
        size4 = range(1 if isinstance(self.leftbc, LeftDirichletBC) else 0, self.N + quadratic if isinstance(self.rightbc, RightDirichletBC) else self.N+1 + quadratic) # (1,2N+1) if dirichlet (0,2N+2) if robin

        nodes = np.linspace(0, 1, self.N + 2)

        self.quadratic = quadratic
        self.nodes = nodes
        self.size = size

        def bilinear_form(u,i,v,j):
            return integrate.quad(lambda x: -1*self.ode.a(x)*u(i,x,nodes,deriv=True)*v(j,x,nodes,deriv=True) + self.ode.b(x)*u(i,x,nodes,deriv=True)*v(j,x,nodes) + self.ode.c(x)*u(i,x,nodes)*v(j,x,nodes), 0, 1)[0]

        def linear_functional(v,i):
            return integrate.quad(lambda x: self.ode.f(x)*v(i,x,nodes), 0, 1)[0]

        def sigma(i,x,nodes,deriv=False):
            if i < testvariable:
                return phi(i,x,nodes,deriv=deriv)
            else:
                return psi(i+1-testvariable,x,nodes,deriv=deriv)

        A = np.zeros((k, k))
        f_array = np.empty(k)

        for index, i in enumerate(size3):
            A[index,index] = bilinear_form(sigma,i,sigma,i) # [0,N-1] Dirichlet with [1,N] phi , [0,N+1] robin with same phi

        for index, i in enumerate(size4):
            A[index,index+1] = bilinear_form(sigma,i+1,sigma,i)
            A[index+1,index] = bilinear_form(sigma,i,sigma,i+1)

        for index, i in enumerate(size3):
            f_array[index] = linear_functional(sigma,i)


        self.leftbc.apply(nodes, f_array, A, self.ode, phi if quadratic == 0 else sigma, len(size)-1, bilinear_form)
        self.rightbc.apply(nodes, f_array, A, self.ode, phi if quadratic == 0 else sigma, len(size)-1, bilinear_form)

        U = np.linalg.solve(A, f_array)
        return U

    def plot_solution(self):
        U = self.solve()
        def u_h(x):
            approx = 0
            for index, i in enumerate(self.size):
                approx += U[index]*phi(i,x,self.nodes)
            if self.quadratic != 0:
                for i in range(1,self.N+2):
                    approx += U[len(self.size)+i-1]*psi(i,x,self.nodes)
            if isinstance(self.leftbc, LeftDirichletBC):
                approx += self.leftbc.leftbc*phi(0,x,self.nodes)
            if isinstance(self.rightbc, RightDirichletBC):
                approx += self.rightbc.rightbc*phi(self.N+1,x,self.nodes)
            return approx

        xes = np.linspace(0,1,200)
        plt.plot(xes, [u_h(x) for x in xes], label = "FE Solution")