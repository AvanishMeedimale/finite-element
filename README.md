# Welcome to pyFEMSolver!
This is a python package that implements the Finite Element Method (FEM) to numerically approximate solutions to Partial Differential Equations (PDEs), following the methodology taught in Professor Iain Smears' course - MATH0092: Variational Methods for PDEs. It supports linear and non-linear second order differential equations in 1D (ODEs) and Dirichlet, Robin, and mixed boundary conditions. It also includes basis functions up to piecewise quadratic functions (P2) for a quadratic convergence to the exact solution.

- **Tutorial:** To get started and understand the methodology and implementation details, please refer to [**Finite_Element_Method_Tutorial.ipynb**](Finite_Element_Method_Tutorial.ipynb)
- **Usage:** To use the application, you can adjust the parameters in [**script.py**](script.py) according to your specific differential equation and boundary conditions.

## Installation
This project requires `NumPy`, `SciPy`, and `Matplotlib` to be installed, and optionally `SymPy` for displaying the equations in LaTeX. You can install it via pip:
```
pip install pyfemsolver
```
To import all the classes and functions, refer to the example in [**script.py**](script.py) for usage instructions.

## Limitations
The implementation follows the structure of the UCL course, which has imposed certain conditions required for a guaranteed solution. This conditions are necessary to invoke the Lax-Milgram theorem and guarantee a solution using the Finite Element Method:
- The differential equation must be _elliptic_.
- PDE coefficients must be sufficiently smooth ($L^{\infty}$ for LHS and $L^{2}$ for RHS).
- The bilinear form of the PDE's weak formulation must be coercive.
