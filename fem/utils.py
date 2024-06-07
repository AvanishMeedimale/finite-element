# Implement the finite element basis functions
import logging
import os

handler = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
logger.addHandler(handler)

def phi(i, x, nodes, deriv=False):
    """Linear piecewise affine function finite element basis
    i: integer between 0 and N+1
    x: real number between 0 and 1
    deriv: True or False to indicate whether function is basis or its derivative
    """
    try:
        N=len(nodes)-2
        h = 1/(N+1)
        if not isinstance(i, int):
            raise TypeError(f"Variable 'i' must be an integer, got {i}")
        if not (0 <= i <= N+1):
            raise ValueError(f"Variable 'i' must be an integer between 0 and {N+1}, got {i}")

        if not deriv:
            if i == 0:
                if nodes[0] <= x <= nodes[1]:
                    return 1 - (x-nodes[i])/h
                else:
                    return 0
            elif i == N+1:
                if nodes[-2] <= x <= nodes[-1]:
                    return 1 + (x-nodes[i])/h
                else:
                    return 0
            else:
                if nodes[i-1] <= x <= nodes[i]:
                    return 1 + (x-nodes[i])/h
                elif nodes[i] < x <= nodes[i+1]:
                    return 1 - (x-nodes[i])/h
                else:
                    return 0

        else:
            if i == 0:
                if nodes[0] <= x <= nodes[1]:
                    return -1/h
                else:
                    return 0
            elif i == N+1:
                if nodes[-2] <= x <= nodes[-1]:
                    return 1/h
                else:
                    return 0
            else:
                if nodes[i-1] <= x <= nodes[i]:
                    return 1/h
                elif nodes[i] < x <= nodes[i+1]:
                    return -1/h
                else:
                    return 0

    except TypeError as err:
        logger.error(f"TypeError: {err}")
    except ValueError as err:
        logger.error(f"ValueError: {err}")
    except Exception as err:
        logger.error(f"Error: {err}")

def psi(i, x, nodes, deriv=False):
    """Quadratic piecewise affine function finite element basis
    i: integer between 1 and N+1
    x: real number between 0 and 1
    deriv: True or False to indicate whether function is basis or its derivative
    """
    try:
        N=len(nodes)-2
        if not isinstance(i, int):
            raise TypeError(f"Variable 'i' must be an integer, got {i}")
        if not (1 <= i <= N+1):
            raise ValueError(f"Variable 'i' must be an integer between 1 and {N+1}, got {i}")

        if not deriv:
            if nodes[i-1] <= x <= nodes[i]:
                return (x-nodes[i-1])*(x-nodes[i])/2
            else:
                return 0

        else:
            if nodes[i-1] <= x <= nodes[i]:
                return x-(nodes[i-1]+nodes[i])/2
            else:
                return 0

    except TypeError as err:
        logger.error(f"TypeError: {err}")
    except ValueError as err:
        logger.error(f"ValueError: {err}")
    except Exception as err:
        logger.error(f"Error: {err}")
