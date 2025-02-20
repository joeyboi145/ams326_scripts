# AMS 316: HW Set 1
# Name: Joe Martinez
# SBUID: 11241692
# NetID: jjmmartinez

import math, sys, numpy as np
import matplotlib.pyplot as plt

P_ERROR = 0.5 * (math.pow(10, -4))
ROOT = 0.641583
ITER_MAX = 10000

def test_f():
    print("\nTESTING Problem 1:")
    v = [1,2,3]
    for n in v: print(f"f({n}) = " + str(f(n)))
    print(f"P_ERROR: {P_ERROR}")
    print(f"ROOT: {ROOT}\n")


def test_p():
    print("\nTESTING PROBLEM 2.1:")
    # EXAMPLE: https://smashmath.github.io/math/polyinterp/
    x_points = np.array([1, 2, 3])
    y_points = np.array([9, 7, -1])
    c = interpolate_polynomial(x_points, y_points)
    for n in x_points:
        s = np.polyval(c, n)
        print(f"f({n}) = {s}")


def test_q():
    # From Notes Exmaple
    print("\nTESTING PROBLEM 2.2")
    x = np.array([-1, 0, 1, 2])
    y = np.array([1, 0, 0, -2])
    x_hat = quadratic_fit(x, y, True)


# f(x) = e^(-x^3) - x^4 - sin(x)
# Number of FLOP is 4 + 19 + 46 = 69
# 46 for e^x (16th degree term), 19 for sin(x) (13th degree term), 
# and 4 from multiplication for small x 
# Assume truncation at the 7th Taylor Series term
def f(x: float) -> float:
    double_x = x * x
    exponent = -1 * double_x * x 
    term1 = math.exp(exponent)
    term2 = double_x * double_x
    term3= math.sin(x)
    return term1 - term2 - term3

# f'(x) = (-3x^2)*e^(-x^3) - 4x^3 - cos(x)
# Number of FLOP is 7 + 18 + 46 = 71
# 46 for e^x (16th degree term), 18 for cos(x) (12th degree term), 
# and 7 from multiplcation for small x
# Assume truncation at the 7th Taylor Series
def df(x: float) -> float:
    double_x = x * x
    exponent = -1 * double_x * x 
    term1 = (-3 * double_x) * math.exp(exponent)
    term2 = 4 * (double_x * x)
    term3 = math.cos(x)
    return term1 - term2 - term3


def bisection_method(a: float, b: float, verbose = False) -> None:
    print("\nBISECTION METHOD:")

    initial_condition = f(a) * f(b)
    print(f"    Initial Condition f({a})*f({b}) = {initial_condition} < 0: "
        + ("TRUE" if initial_condition < 0 else "FALSE"))
    if not initial_condition < 0:
        print(f"ERROR: f(a)*f(b) must be below  0")
        return

    i = 0
    x = (a + b) / 2
    error = math.fabs(x - ROOT)
    while (error >= P_ERROR):
        x = (a + b) / 2
        if (verbose): 
            print(f"{i} & {a} & {b} & {x} & {str(f(a))[:8]} & {str(f(x))[:8]} \\\\")
            # print(f"    Iteration {i}: x = {x}")
            # print(f"      a = {a}")
            # print(f"      b = {b}")
            # print(f"      f(x) = {f(x)}")
        if f(a) * f(x) < 0: b = x
        else: a = x
        i += 1
        error = math.fabs(x - ROOT)

    # if (verbose): print(f"{i} & {a} & {b} & {x} & {str(f(a))[:8]} & {str(f(x))[:8]} \\\\")
    print(f"Number of iterations: {i}")
    print(f"Number of estimated FLOP needed: ~{142 * i + 4 + 140}")
    print(f"Final Result: {x}")
    print(f"Final Error: {error}\n")


def newton_method(x: float, verbose = False) -> None:
    print("\nNEWTON'S METHOD:")

    i = 0
    error = math.fabs(x - ROOT)
    # TOTAL FLOP: 2 * (i + 1) + 142 * i  = 144i + 2
    while (error >= P_ERROR):
        if (verbose): 
            # print(f"    Iteration {i}: x = {x}")
            print(f"{i} & {x} \\\\")

        # FLOP: 2 + 69 + 71 = 142
        x = x - (f(x) / df(x))
        i += 1
        error = math.fabs(x - ROOT)
        if (i > ITER_MAX):
            print("ERROR: Reached maximum iterations. Converging too slow or not at all")
            print(f"Result: {x}")
            print(f"Error: {error}\n")
            return

    print(f"Number of iterations: {i}")
    print(f"Number of estimated FLOP: ~{144 * i + 2}")
    print(f"Final Result: {x}")
    print(f"Final Error: {error}\n")


def secant_method(x1: float, x2: float, verbose = False) -> None:
    print("\nSECANT METHOD:")

    i = 0
    error = math.fabs(x2 - ROOT)
    # TOTAL FLOP: 2 * (i + 1) + ((69 * 2) + 7) * i
    # = 2i + 1 + 145i = 147i + 2
    while (error >= P_ERROR):
        if (verbose): 
            # print(f"{i} & {x1} & {x2} \\\\")
            print(f"    Iteration {i}: x = {x2}")
        fx_store = f(x2)    # Calculate f(x2) once to save FLOP
        denom = (fx_store - f(x1)) / (x2 - x1)
        x3 = x2 - (fx_store / denom)
        x1 = x2
        x2 = x3
        i += 1 
        error = math.fabs(x2 - ROOT)
        if (i > ITER_MAX):
            print("ERROR: Reached maximum iterations. Converging too slow or not at all")
            print(f"Result: {x2}")
            print(f"Error: {error}\n")
            return

    if (verbose): print(f"{i} & {x1} & {x2} \\\\")
    print(f"Number of iterations: {i}")
    print(f"Number of estimated FLOP: ~{147 * i + 2}")
    print(f"Final Result: {x2}")
    print(f"Final Error: {error}\n")


def monte_carlo_method() -> None:
    # https://en.wikipedia.org/wiki/Monte_Carlo_method#:~:text=Draw%20a%20square%2C%20then%20inscribe,placed%20in%20the%20whole%20square.
    pass


def interpolate_polynomial(x, y):
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


def quadratic_fit(x, y, verbose = False):
    print("QUADRATIC FIT:")
    equations = []
    for n in x:
        array = []
        for p in range(2, -1, -1):
            array.append(math.pow(n, p))
        equations.append(array)
    A = np.array(equations)
    A_t = A.transpose()
    M = A_t @ A
    b = A_t @ y

    # A_t * Ax = A_t * y
    # Let M = A_t * Ax
    # Let b = A_t * y
    # Therefore: Mx = b

    x_hat = np.linalg.solve(M, b)
    print(f"y:" + str(y))
    print(f"Coefficient Matrix:\n" + str(A))
    if verbose:
        print(f"\nA_t:\n" + str(A_t))
        print(f"\nA_t * A :\n" + str(A_t @ A))
        print(f"\nA_t * y:\n" + str(A_t @ y))
    print(f"\nResults: " + str(x_hat))
    return x_hat


if __name__ == "__main__":


    verbose = False
    graphs = False

    if len(sys.argv) >= 2:
        flag = sys.argv[1]
        if flag == '-v' or flag == '--verbose':
            verbose = True
        elif flag == '-h' or flag == '--help':
            print("Usage: \
                  \n  -v  --verbose\tProvides additional information and the value of each iteration for each method \
                  \n  -h  --help\tUsage menu \
                  \n  -g  --graph\tdisplay graphs \
                  \n  -t  --test\tPerform rudimentary tests ")
            sys.exit(0)
        elif flag == '-g' or flag == "--graph":  graphs = True
        elif flag == '-t' or flag == '--test': 
            test_f()
            test_p()
            test_q()
            sys.exit(0)


    # PROBLEM 1: ROOT APPROXIMATION
    bisection_method(-1, 1, verbose)
    newton_method(0, verbose)
    secant_method(-1, 1, verbose)
    monte_carlo_method()


    x_points = [1, 2, 3, 4, 5]
    y_points = [412, 407, 397, 398, 417]
    x = np.array(x_points)
    y = np.array(y_points)

    # PROBLEM 2.1: POLYNOMIAL INTERPOLATE
    print("\nPOLYNOMIAL INTERPOLATION P(t) OF TELSA STONKS:")
    coeff = interpolate_polynomial(x, y)
    # TEST RESULTS
    test_x = 6
    result = np.polyval(coeff, test_x)
    print(f"\nPolynomial Interpolation with t = {test_x}")
    print(f"P({test_x}) = {result}\n")


    # PROBLEM 2.2: QUADRATIC FIT
    print("\nQUADRATIC FIT Q(t) OF TELSA STONKS:")
    x_hat = quadratic_fit(x, y, verbose)
    # TEST RESULTS
    test_x = 6
    result = np.polyval(x_hat, test_x)
    print(f"\nQuadratic Fit t = {test_x}")
    print(f"Q({test_x}) = {result}\n")
    

    if graphs:
        xp = np.linspace(0,6,400)
        yp1 = np.polyval(coeff, xp)
        yp2 = np.polyval(x_hat, xp)

        plt.plot(x, y, 'o', label='Original data', markersize=10)
        plt.plot(xp, yp1, label='Interpolated Line')
        plt.plot(xp, yp2, label='Quadratic Fitted Line')
        plt.legend()
        plt.show()
        

    



    
