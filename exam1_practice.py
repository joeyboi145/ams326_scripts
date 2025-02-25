# AMS 316: Exam 1
# Name: Joe Martinez
# SBUID: 11241692
# NetID: jjmmartinez

import math, numpy as np, matplotlib.pyplot as plt

PI = math.pi

def uniform_nodes(n: int, a:float, b:float) -> list:
    range_size = math.fabs(b-a)
    interval = range_size / (n-1)
    x_points = []
    for i in range(0, n):
        x_points.append(i * interval)
    return x_points


def chebyshev_nodes(n: int, a: float, b: float) -> list:
    """
    Returns a list of x value representing n number of Chebyshev nodes in the closed interval [a,b]

    Args:
        n (int): number of nodes
        a (float): start of range
        b (float): end of range
    
    Returns:
        (list): list of x values
    """
    if a >= b:
        raise ValueError

    x_nodes = []
    for i in range(1, n + 1):
        value = (b+a)/2
        value += ((b-a)/2) * math.cos(((2*i-1)*PI)/(2*n))
        x_nodes.append(value)
    return x_nodes


def interpolate_polynomial(x: list, y: list) -> np.ndarray:
    """
    Returns a list of numbers that represent the coefficients of an
    interpolation polynomial that maps the list of inputed x and 
    y value.

    Args:
        x (list): A list of x values 
        y (list): A list of y values 

    Returns:
        numpy.ndarray: A list of coefficients for the interpolating polynomial
    """
    print("INTERPOLATE POLYNOMIAL:")
    equations = []
    num_points = len(x)
    for i in range(0, num_points):
        polynomial = []
        point = x[i]
        for j in range(num_points - 1, -1, -1):
            polynomial.append(math.pow(point, j))
        equations.append(polynomial)

    A = np.array(equations)
    coeff = np.linalg.solve(A, y)
    print("    X Points:" + str(x))
    print("    Y Points:" + str(y))
    print("Coefficient Matrix:\n" + str(A))
    print("Results:\n" + str(coeff))
    return coeff


def main():
    a = -1
    b = 1
    x_points = chebyshev_nodes(5, a, b);
    y_points = []
    for x in x_points:
        y_points.append(math.exp(x))
    coeff = interpolate_polynomial(x_points, y_points)


    x_points_normal = uniform_nodes(5, a, b)
    y_points_normal = []
    for x in x_points_normal:
        y_points_normal.append(math.exp(x))
    normal_coeff = interpolate_polynomial(x_points_normal, y_points_normal)

    x_space = np.linspace(a, b, 400)
    y1 = np.exp(x_space)
    plt.plot(x_space, y1, label='f(x)')

    y2 = np.polyval(coeff, x_space)
    plt.plot(x_space, y2, label='Chebyshev P(x)')

    y3 = np.polyval(normal_coeff, x_space)
    plt.plot(x_space, y3, label='P(x)')

    plt.plot(x_points, y_points,'o', label='Points', markersize=8)
    plt.legend()
    plt.show()





if __name__ == "__main__":
    main()