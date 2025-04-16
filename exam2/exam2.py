# AMS 316: Exam 2
# Name: Joe Martinez
# SBUID: 11241692
# NetID: jjmmartinez

# Source Code found at:
# https://github.com/joeyboi145/ams326_scripts/blob/main/exam2.py


import math, sys, warnings, random, time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

TOL = math.pow(10, -4)
P_ERROR = 4 * math.pow(10, -6)
BOUND_CORRECTION = math.pow(10, -7)
ITER_MAX = 10000
VERBOSE = True
MIN_NODES = 20
VERBOSE = False

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
        elif (i % 3 == 0): sum += 3 * value
        else: sum += 2 * value

        if (VERBOSE): print(f"    OUTPUT: {value}, complex: {np.imag(value)}")

    sum = ((3 * interval) / 8) * sum
    return np.real(sum)

# def monte_carlo_method()


def show_graphs(a, b, solutions):
        x, y = sp.symbols('x y')
        # k1 = sp.lambdify(x, kidney_solutions[0], modules='numpy') 
        X_SPACE = np.linspace(a, b, 10000)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pass


        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

def main():
    global VERBOSE
    graphs = False

    if len(sys.argv) >= 2:
        if '-g' in sys.argv: graphs = True
        if '-V' in sys.argv: VERBOSE = True


    # x, y = sp.symbols('x y')
    # func = sp.exp(x**5)
    # f = sp.lambdify(x, func, modules='numpy')

    def f(x):
        return math.exp(x**5);

    midpoint_area = midpoint_method(f, -1, 1, 100)
    print("\nMidpoint Area:", midpoint_area)

    simpson3_area = simpson3_method(f, -1, 1, 101)
    print("\nSimpson 1/3:", simpson3_area)

    simpson8_area = simpson8_method(f, -1, 1, 101)
    print("\nSimpson 1/3:", simpson8_area)



    # if (graphs):
    #     # show_graphs(0, 1, [])
    #     pass


if __name__ == "__main__":
    main()