"""This module provides linear and quadratic piecewise affine finite element basis functions.

These functions are essential components for solving partial differential equations (PDEs) using
the finite element method (FEM). The linear function, phi, represents the linear piecewise affine basis functions,
while the quadratic function, psi, represents the quadratic piecewise affine basis functions.
"""

import logging
import os

# Set up logging
handler = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
logger.addHandler(handler)

def phi(i, x, nodes, deriv=False):
    """
    Linear piecewise affine function finite element basis.

    Args:
        i (int): Integer between 0 and N+1 indicating the basis function index.
        x (float): Real number between 0 and 1 representing the evaluation point.
        nodes (list of float): List of nodal points.
        deriv (bool): Indicates whether to return the basis function or its derivative (default is False).

    Returns:
        float: Value of the basis function or its derivative at x.

    Raises:
        TypeError: If 'i' is not an integer.
        ValueError: If 'i' is not between 0 and N+1.
    """
    try:
        N = len(nodes) - 2
        h = 1 / (N + 1)

        if not isinstance(i, int):
            raise TypeError(f"Variable 'i' must be an integer, got {i}")

        if not 0 <= i <= N + 1:
            raise ValueError(f"Variable 'i' must be an integer between 0 and {N + 1}, got {i}")

        if not deriv:
            if i == 0:
                if nodes[0] <= x <= nodes[1]:
                    return 1 - (x - nodes[i]) / h
                return 0
            if i == N + 1:
                if nodes[-2] <= x <= nodes[-1]:
                    return 1 + (x - nodes[i]) / h
                return 0
            if nodes[i - 1] <= x <= nodes[i]:
                return 1 + (x - nodes[i]) / h
            if nodes[i] < x <= nodes[i + 1]:
                return 1 - (x - nodes[i]) / h
            return 0

        if i == 0:
            if nodes[0] <= x <= nodes[1]:
                return -1 / h
            return 0
        if i == N + 1:
            if nodes[-2] <= x <= nodes[-1]:
                return 1 / h
            return 0
        if nodes[i - 1] <= x <= nodes[i]:
            return 1 / h
        if nodes[i] < x <= nodes[i + 1]:
            return -1 / h
        return 0

    except TypeError as err:
        logger.error("TypeError: %s", err)
    except ValueError as err:
        logger.error("ValueError: %s", err)
    except Exception as err:
        logger.error("Error: %s", err)

def psi(i, x, nodes, deriv=False):
    """
    Quadratic piecewise affine function finite element basis.

    Args:
        i (int): Integer between 1 and N+1 indicating the basis function index.
        x (float): Real number between 0 and 1 representing the evaluation point.
        nodes (list of float): List of nodal points.
        deriv (bool): Indicates whether to return the basis function or its derivative (default is False).

    Returns:
        float: Value of the basis function or its derivative at x.

    Raises:
        TypeError: If 'i' is not an integer.
        ValueError: If 'i' is not between 1 and N+1.
    """
    try:
        N = len(nodes) - 2

        if not isinstance(i, int):
            raise TypeError(f"Variable 'i' must be an integer, got {i}")

        if not 1 <= i <= N + 1:
            raise ValueError(f"Variable 'i' must be an integer between 1 and {N + 1}, got {i}")

        if not deriv:
            if nodes[i - 1] <= x <= nodes[i]:
                return (x - nodes[i - 1]) * (x - nodes[i]) / 2
            return 0

        if nodes[i - 1] <= x <= nodes[i]:
            return x - (nodes[i - 1] + nodes[i]) / 2

        return 0

    except TypeError as err:
        logger.error("TypeError: %s", err)
    except ValueError as err:
        logger.error("ValueError: %s", err)
    except Exception as err:
        logger.error("Error: %s", err)
