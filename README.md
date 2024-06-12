# finite-element
This repository implements the Finite Element Method (FEM) to numerically approximate solutions to certain elliptic Partial Differential Equations, following the methodology defined in Professor Iain Smears' course, MATH0092: Variational Methods for Partial Differential Equations.

To familiarise yourself with how to method works and the core ideas on how it has been implemented, please refer to [**Finite_Element_Method_Tutorial.ipynb**](Finite_Element_Method_Tutorial.ipynb)
</br>
</br>
To use the application, please use [**script.py**](script.py) and adjust the parameters according to your equation and boundary conditions.

_Note: The current implementation handles the one-dimensional case (Ordinary Differential Equations) and supports Dirichlet, Robin, and mixed boundary conditions. The basis functions are implemented up to piecewise quadratic functions (P2)._

# Installation
This project requires numpy, scipy and matplotlib to be installed (and optionally sympy). It can be installed by running the following command in the terminal:
```
pip install pyfemsolver
```
To import all the classes and functions, use `import fem` and then proceed similar to the demonstration in [**script.py**](script.py)
