import math, sys, warnings, random, time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

TOL = math.pow(10, -4)
P_ERROR = 0.5 * math.pow(10, -4)
BOUND_CORRECTION = math.pow(10, -7)
ITER_MAX = 10000
VERBOSE = False
MIN_NODES = 20

def find_area(method, f, a: float, b: float, n: int, 
              piecewise: list = None, zero=False) -> float:
    """
    Calculates the area under the curve 'f' in a given range [a,b] using the 
    specified method for approximating integration. It first started with the given nodes n, 
    but increases them by a factor of two for each iteration until an area with a given tolerance
    for accuracy

    Args:
        f (float -> float): function to integrate over
        a (float): start of range
        b (float): end of range
        n (int): number of nodes to approximate integration with
        piecewise (list[float -> float]): optional list of continuous functions
          for integration in case of a piecewise function
        zero (boolean): flag for indeterminate for for f(0)

    Returns:
        float: the approximate area under the curve under the given range

    """
    a1 = method(f, a, b, n, piecewise=piecewise, zero=zero)
    a2 = None
    i = 0
    error = None
    global VERBOSE

    while True:
        n = 2 * n
        if (method == simpson3_method or method == simpson8_method):  n += 1
        a2 = method(f, a, b, n, piecewise=piecewise, zero=zero)
        error = math.fabs((a1 - a2))
        i += 1
        if (error < P_ERROR): break 
        a1 = a2
        if (VERBOSE):
            print(f"Iternation {i}:\n  First Area: {a1}\n  Second Area: {a2}\n  Nodes: {int(n/2)} vs {n}\n  Error: {error}")

    a1 = math.fabs((a1 - 4*a2) / 3) # Richardson Extrapolation
    print(f"Iteration {i}:")
    print(f"Final Area: {a1}")
    print(f"Final Error: {error}, {error < P_ERROR}")
    print(f"Final Nodes: {n}")
    return a1


def midpoint_method(f, a: float, b: float, n: int, piecewise: list = None, zero = False) -> float:
    """
    The Midpoint Method of area approximation for a given function f
    over a given range [a,b] using n node. If the function is a piecewise 
    function over the given interval, provide ordered list of piecewise functions
    orderd by increasing x domain.

    Args:
        f (float -> float): function to integrate over
        a (float): start of range
        b (float): end of range
        n (int): number of nodes to approximate integration with
        piecewise (list[float -> float]): optional list of continuous functions
          for integration in case of a piecewise function

    Returns:
        float: the approximate area under the curve under the given range
    """
    if (n <= 1): raise ValueError("n must be two nodes or greater")
    if (b <= a): raise ValueError("b must be greater than a")

    global VERBOSE
    if (piecewise): 
        piecewise = piecewise.copy()
    interval = math.fabs(b - a) / (n-1)
    nodes = np.linspace(a, b, n)
    midpoints = np.array([])
    for i in range(0, len(nodes) - 1):
        midpoint = (nodes[i] + nodes[i + 1]) / 2
        midpoints = np.append(midpoints, midpoint)

    # print(f"a = {a}, b = {b}")
    # print(f"nodes ({len(nodes)}): {nodes}")
    # print(f"midpoints ({len(midpoints)}): {midpoints}" )
    # print(f"distance: {interval}")

    sum = 0
    for i in range(0, len(midpoints)):
        midpoint = midpoints[i]
        if (zero and midpoint == 0): continue
        midpoint = midpoint.astype(complex)
        if (VERBOSE): print(f"    INPUT: {midpoint}")
        value = f(midpoint)
        if (np.imag(value) > 0): 
            if (VERBOSE): print(f"     IMGAINARY ({i}): ({midpoint}, {value}")
            f = piecewise.pop(0)
            value = f(midpoint)
        sum += value
        if (VERBOSE): print(f"    OUTPUT: {value}, complex: {np.imag(value)}")
    return np.real(interval * sum)


def trapezoid_method(f, a: float, b: float, n: int, piecewise: list = None, zero = False) -> float:
    """
    The Trazpezoid Method of area approximation for a given function f
    over a given range [a,b] using n node. If the function is a piecewise 
    function over the given interval, provide ordered list of piecewise functions
    orderd by increasing x domain.

    Args:
        f (float -> float): function to integrate over
        a (float): start of range
        b (float): end of range
        n (int): number of nodes to approximate integration with
        piecewise (list[float -> float]): optional list of continuous functions
          for integration in case of a piecewise function

    Returns:
        float: the approximate area under the curve under the given range
    """
    if (n <= 1): raise ValueError("n must be two nodes or greater")
    if (b <= a): raise ValueError("b must be greater than a")

    if (piecewise): 
        piecewise = piecewise.copy()
    interval = math.fabs(b - a) / (n-1)
    nodes = np.linspace(a, b, n)
    global VERBOSE

    # print(f"nodes ({len(nodes)}): {nodes}")
    # print(f"distance: {interval}")
    sum = 0
    for i in range(0, len(nodes)):
        node = nodes[i]
        if (zero and node == 0): continue
        node = node.astype(complex)
        if (VERBOSE): print(f"    INPUT: {node}")
        value = f(node)
        if np.imag(value) > 0:
            if (VERBOSE): print(f"     IMGAINARY ({i}): ({node}, {value}")
            f = piecewise.pop(0)
            value = f(node)
        
        if (i == 0 or i == len(nodes)-1): sum += value
        else: sum += 2 * value
        if (VERBOSE): print(f"    OUTPUT: {value}, complex: {np.imag(value)}")

    sum = (interval / 2) * sum
    return np.real(sum)


def simpson3_method(f, a: float, b: float, n: int, piecewise: list = None, zero = None) -> float: 
    if (n <= 2): raise ValueError("n must be 3 nodes or greater")
    if (n % 2 != 1): raise ValueError("n must be odd, even intervals")
    if (b <= a): raise ValueError("b must be greater than a")

    if (piecewise): 
        piecewise = piecewise.copy()
    interval = math.fabs(b - a) / (n-1)
    nodes = np.linspace(a, b, n)
    global VERBOSE

    sum = 0
    for i in range(0, len(nodes)):
        node = nodes[i]
        if (zero and node == 0): continue

        node = node.astype(complex)
        value = f(node)
        if (VERBOSE): print(f"    INPUT: {node}")
        # if np.imag(value) > 0:
        #     if (VERBOSE): print(f"     IMGAINARY ({i}): ({node}, {value}")
        #     f = piecewise.pop(0)
        #     value = f(node)
        
        if (i == 0 or i == len(nodes)-1): sum += value
        elif (i % 2 == 1): sum += 4 * value
        else: sum += 2 * value

        if (VERBOSE): print(f"    OUTPUT: {value}, complex: {np.imag(value)}")

    sum = (interval / 3) * sum
    return np.real(sum)

def simpson8_method(f, a: float, b: float, n: int, piecewise: list = None, zero = None) -> float: 
    if (n <= 2): raise ValueError("n must be 3 nodes or greater")
    if (n % 2 != 1): raise ValueError("n must be odd, even intervals")
    if (b <= a): raise ValueError("b must be greater than a")

    if (piecewise): 
        piecewise = piecewise.copy()
    interval = math.fabs(b - a) / (n-1)
    nodes = np.linspace(a, b, n)
    global VERBOSE

    sum = 0
    for i in range(0, len(nodes)):
        node = nodes[i]
        if (zero and node == 0): continue

        node = node.astype(complex)
        value = f(node)
        if (VERBOSE): print(f"    INPUT: {node}")
        # if np.imag(value) > 0:
        #     if (VERBOSE): print(f"     IMGAINARY ({i}): ({node}, {value}")
        #     f = piecewise.pop(0)
        #     value = f(node)
        
        if (i == 0 or i == len(nodes)-1): sum += value
        elif (i % 4 == 2 or i % 4 == 3): sum += 3 * value
        else: sum += 2 * value

        if (VERBOSE): print(f"    OUTPUT: {value}, complex: {np.imag(value)}")

    sum = (3 * interval / 8) * sum
    return np.real(sum)



def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2*h)

def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

def forward_difference(f, x, h):
    return (f(x) - f(x - h)) / h


x= sp.symbols('x')
func = sp.exp(6*x)**4
f = sp.lambdify(x, func, modules='numpy')

a = 0
b = 2
REAL_VALUE = math.exp(2) - 1
area = find_area(trapezoid_method, f, a, b, 13)

print(f"Real Error: {REAL_VALUE - area}, {math.fabs(REAL_VALUE - area) < P_ERROR}")


# x, y = sp.symbols('x y')
# func = sp.sin(6*x)**4
# f = sp.lambdify(x, func, modules='numpy')
# # print(midpoint_method(f, 0, 1, 6))
# # print(trapezoid_method(f, 0, 1, 6))
# print(simpson3_method(f, 0, 1, 7))