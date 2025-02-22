# AMS 316: HW Set 1
# Name: Joe Martinez
# SBUID: 11241692
# NetID: jjmmartinez

import math, sys, random, time
import numpy as np
import matplotlib.pyplot as plt

TOL = math.pow(10, -4)
P_ERROR = 0.5 * TOL
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
# Number of FLOP is 19 + 19 + 4 = 44
# 19 for e^x (7 terms), 18 for sin(x) (7 terms), 
# and 4 from multiplication for small x 
# Assume truncation at the 7th Taylor Series term
def f(x: float) -> float:
    """
    This function represents the code version of f(x) = e^(-x^3) - x^4 - sin(x)

    Args:
        x (float): An x value to evaluate

    Returns:
        float: f(x)
    """
    double_x = x * x
    exponent = -1 * double_x * x 
    term1 = math.exp(exponent)
    term2 = double_x * double_x
    term3= math.sin(x)
    return term1 - term2 - term3

# f'(x) = (-3x^2)*e^(-x^3) - 4x^3 - cos(x)
# Number of FLOP is 9 + 19 + 18 = 46
# 19 for e^x (7 terms), 18 for cos(x) (7 terms), 
# and 9 from multiplcation for small x
# Assume truncation at the 7th Taylor Series
def df(x: float) -> float:
    """
    This function represents the code version of f'(x) = (-3x^2)*e^(-x^3) - 4x^3 - cos(x)
    the derivative to f(x) = e^(-x^3) - x^4 - sin(x)

    Args:
        x (float): An x value to evaluate

    Returns:
        float: f'(x)
    """
    double_x = x * x
    exponent = -1 * double_x * x 
    term1 = (-3 * double_x) * math.exp(exponent)
    term2 = 4 * (double_x * x)
    term3 = math.cos(x)
    return term1 - term2 - term3


def bisection_method(a: float, b: float, verbose = False) -> None:
    """
    The bisection method for root approximation using a range of [a,b] containing
    a root which returns 4-decimal place accurate result and outputs the number 
    of iteration and floating point operations.

    Args:
        a (float): The start of the range
        b (float): The end of the range
        verbose (boolean): if True, prints values for each iteration

    Returns:
        float: A value that approximates the root with 4 decimal place accuracy
    """
    print("\nBISECTION METHOD:")

    # FP: 1 + 44 + 44 + 1 + 1 = 91
    if a >= b:
        print(f"ERROR: a must be smaller than b")
        return None
    
    initial_condition = f(a) * f(b)
    if initial_condition >= 0:
        print(f"ERROR: f(a)*f(b) must be below 0")
        return
    else:
        print(f"    Initial Condition f({a})*f({b}) = {initial_condition} < 0: TRUE")
        

    # FP: 1 + 1 + (i + 1)*2 + (44 + 44 + 1 + 1 + 1)*i = 4 + 93i
    i = 0
    x = (a + b) / 2
    range_size = math.fabs(b - a)
    error = range_size / 2
    while (error >= P_ERROR):
        if (verbose): 
            # print(f"    Iteration {i}: x = {x}", f"\n\ta = {a}", f"\n\tb = {b}", f"\n\tf(x) = {f(x)}")
            print(f"{i} & {str(a)[:8]} & {str(b)[:8]} & {x} & {str(f(a))[:7]} & {str(f(x))[:7]} \\\\")

        x = (a + b) / 2
        if f(a) * f(x) < 0: b = x
        else: a = x
        i += 1
        error = error / 2

    if (verbose): print(f"{i} & {str(a)[:8]} & {str(b)[:8]} & {x} & {str(f(a))[:8]} & {str(f(x))[:8]} \\\\")
    print(f"Number of iterations: {i}")
    print(f"Number of estimated FLOP needed: ~{95 + 93 * i}")
    print(f"Final Result: {x}", math.fabs(x - ROOT))
    print(f"Final Error: {error}")


def newton_method(x: float, verbose = False) -> None:
    """
    Newton's method for root approximation using an initial guess root 
    which returns a 4-decimal place accurate result and outputs the 
    number of iteration and floating point operations.

    Args:
        x (float): initial guess for a root
        verbose (boolean): if True, prints values for each iteration

    Returns:
        float: A value that approximates the root with 4 decimal place accuracy
    """
    print("\nNEWTON'S METHOD:")

    i = 0
    while True:
        if (verbose): print(f"    Iteration {i}: x = {x}")

        prev_x = x
        x = x - (f(x) / df(x))
        error = math.fabs(x - prev_x)
        i += 1
        if error < TOL: break
        elif (i > ITER_MAX):
            print("ERROR: Reached maximum iterations. Converging too slow or not at all")
            print(f"Result: {x} \n Error: {error}\n")
            return
    print(f"Number of iterations: {i}")
    print(f"Number of estimated FLOP: ~{94 * i}")
    print(f"Final Result: {x}")
    print(f"Final Error: {error}", math.fabs(x - ROOT))
    return x



def secant_method(x1: float, x2: float, verbose = False) -> None:
    """
    The Secant method for root approximation using two initial guesses 
    which returns a 4-decimal place accurate result and outputs the \
    number of iteration and floating point operations.

    Args:
        x1 (float): first initial guess for a root
        x2 (float): second initial guess for a root
        verbose (boolean): if True, prints values for each iteration

    Returns:
        float: A value that approximates the root with 4 decimal place accuracy
    """
    print("\nSECANT METHOD:")
    if x1 == x2: 
        print("ERROR: x1 cannot equal x2")
        return

    i = 0
    while True:
        if (verbose): print(f"    Iteration {i}: x = {x2}")

        fx2 = f(x2)     # Calculate f(x2) once to save FP
        denom = (fx2 - f(x1)) / (x2 - x1)
        x3 = x2 - (fx2 / denom)
        x1 = x2
        x2 = x3
        i += 1
        error = math.fabs(x2 - x1)
        if error < TOL: break
        elif (i >= ITER_MAX):
            print("ERROR: Reached maximum iterations. Converging too slow or not at all")
            print(f"Result: {x2} \n Error: {error}\n")
            return
    print(f"Number of iterations: {i}")
    print(f"Number of estimated FLOP: ~{95 * i + 1}")
    print(f"Final Result: {x2}")
    print(f"Final Error: {error}", math.fabs(x2 - ROOT))
    return x2


def monte_carlo_method(a: float, b: float, verbose = False) -> None:
    """
    A Monte Carlo method for root approximation using a range [a,b] 
    to guess roots within which returns a 4-decimal place accurate 
    result and outputs the number of iteration and floating point operations.

    Args:
        a (float): The start of the range
        b (float): The end of the range
        verbose (boolean): if True, prints values for each iteration

    Returns:
        float: A value that approximates the root with 4 decimal place accuracy
    """
    print("\nMONTE CARLO METHOD:")
    if a >= b:
        print("ERROR: a cannot be greater than or equal to b")
        return

    i = 1
    x = random.uniform(a, b)
    error = math.fabs(x - ROOT)
    while (error >= P_ERROR):
        if (verbose): print(f"\tIteration {i}: {x}")
        x = random.uniform(a, b)
        error = math.fabs(x - ROOT)
        i += 1
    print(f"Number of iterations: {i}")
    print(f"Number of estimated FLOP: ~{3*i + 4}")
    print(f"Final Result: {x}")
    print(f"Final Error: {error}")
    return i


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


def quadratic_fit(x: list, y: list, verbose = False) -> np.ndarray:
    """
    Returns a list of numbers that represent the coefficients of an
    fitted quadratic equation that fits the list of inputed x and 
    y value.

    Args:
        x (list): A list of x values 
        y (list): A list of y values 
        verbose (boolean): if True, prints values of important matrix calculations

    Returns:
        numpy.ndarray: A list of coefficients for the fitted quadratic equation
    """
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


def calc_runtime(start_time: float) -> None:
    """
    Prints the runtime in milliseconds of a segment of code given the start time.

    Args:
        start_time (float): A time value

    """
    end_time = time.perf_counter()
    runtime = (end_time - start_time) * 10 ** 3
    print(f"Runtime: {runtime} ms")


def main():
    ### BEGIN Command line Options Switchboard ###
    perform = True  # if True = run all methods 
    evaluate = [False, False, False, False, False, False]   # if true, run specific method
    verbose = False # if true, print additinoal information
    graphs = False # if true, print graph

    help_menu = "Usage: \
                  \n  -v  --verbose\tProvides additional information and the value of each iteration for each method \
                  \n  -h  --help\tUsage menu \
                  \n  -g  --graph\tDisplay graphs \
                  \n  -t  --test\tPerform rudimentary tests \
                  \n  -p  --perform\tPerform specific test"
    perform_menu = "Perform Test Options: \
                    \n\t-b  --bisection\tPerforms bisection method only \
                    \n\t-n  --newton\tPerforms Newtons method only\
                    \n\t-s  --secant\tPerforms secant method only\
                    \n\t-m  --monte\tPerforms Monte Carlo Method only\
                    \n\t-i  --interpolate\tPerforms polynomial interpolation only\
                    \n\t-f  --fit\tPerforms Quadratic fit only"

    if len(sys.argv) >= 2:
        flag = sys.argv[1]

        # Content Plags: Control addition ouput
        if flag == '-v' or flag == '--verbose': verbose = True
        if flag == '-g' or flag == "--graph":  graphs = True
        
        # Action Flags: Test, Help, or run methods flags
        if flag == '-t' or flag == '--test': 
            test_f()
            test_p()
            test_q()
            sys.exit(0)
        elif flag == '-h' or flag == "--help": 
            print(help_menu)
            sys.exit(0)

        elif len(sys.argv) >= 3 and (verbose or graphs or flag == '-p' or flag == '--perform'):
            # Performance Flags: Control specified test
            if len(sys.argv) == 2:
                print(perform_menu)
                sys.exit(0)
            else:
                method = sys.argv[2]
                if method=='-b' or method == '--bisection': evaluate[0] = True
                elif method =='-n' or method == '--newton': evaluate[1] = True
                elif method == '-s' or method == '--secant': evaluate[2] = True
                elif method == '-m' or method == '--monte':  evaluate[3] = True
                elif method == '-i' or method == '--interpolate': evaluate[4] = True
                elif method == '-f' or method == '--fit': evaluate[5] = True
                else:
                    print("ERROR: Invalid method")
                    print(perform_menu)
                    sys.exit(1)
                if True in evaluate: perform = False
        elif not (verbose or graphs):
            print("ERROR: Invalid flag")
            print(help_menu)
            sys.exit(1)
    ### END Option Switchboard ###


    # PROBLEM 1: ROOT APPROXIMATION
    # Each method prints out the runtime
    if perform or evaluate[0]:
        start_time = time.perf_counter() 
        bisection_method(-1, 1, verbose)
        calc_runtime(start_time)
    if perform or evaluate[1]: 
        start_time = time.perf_counter() 
        newton_method(0, verbose)
        calc_runtime(start_time)
    if perform or evaluate[2]: 
        start_time = time.perf_counter() 
        secant_method(-1, 1, verbose)
        calc_runtime(start_time)
    if perform or evaluate[3]: 
        start_time = time.perf_counter() 
        monte_carlo_method(0.50, 0.75, verbose)
        calc_runtime(start_time)

    # if perform or evaluate[3]: 
    #     for n in range(0, 5):
    #         start_time = time.perf_counter() 
    #         monte_carlo_method(0.50, 0.75, verbose)
    #         calc_runtime(start_time)


    x_points = [1, 2, 3, 4, 5]
    y_points = [412, 407, 397, 398, 417]
    x = np.array(x_points)
    y = np.array(y_points)
    coeff = None
    x_hat = None


    # PROBLEM 2.1: POLYNOMIAL INTERPOLATE
    if perform or evaluate[4]:
        print("\nPOLYNOMIAL INTERPOLATION P(t) OF TELSA STONKS:")
        start_time = time.perf_counter()
        coeff = interpolate_polynomial(x, y)
        calc_runtime(start_time)
        # TEST RESULTS
        test_x = 6
        result = np.polyval(coeff, test_x)
        print(f"\nPolynomial Interpolation with t = {test_x}")
        print(f"P({test_x}) = {result}\n")


    if perform or evaluate[5]:
        # PROBLEM 2.2: QUADRATIC FIT
        print("\nQUADRATIC FIT Q(t) OF TELSA STONKS:")
        start_time = time.perf_counter()
        x_hat = quadratic_fit(x, y, verbose)
        calc_runtime(start_time)
        # TEST RESULTS
        test_x = 6
        result = np.polyval(x_hat, test_x)
        print(f"\nQuadratic Fit t = {test_x}")
        print(f"Q({test_x}) = {result}\n")
    

    if graphs:
        xp = np.linspace(0,7,400)
        if coeff is not None: 
            yp1 = np.polyval(coeff, xp)
            p1 = np.polyval(coeff, [6])
            plt.plot(xp, yp1, label='Interpolated Line')
            plt.plot([6], [p1], 'o', label='P(t) Extrapolated Data', markersize=8)
        if x_hat is not None: 
            yp2 = np.polyval(x_hat, xp)
            p2 = np.polyval(x_hat, [6])
            plt.plot(xp, yp2, label='Quadratic Fitted Line')
            plt.plot([6], [p2], 'o', label='Q(t) Extrapolated Data', markersize=8)


        plt.plot(x, y, 'o', label='Original Data', markersize=8)
        plt.xlabel("Time")
        plt.ylabel("Tesla Stocks Closings")
        plt.legend()
        plt.show()
        


if __name__ == "__main__":
    main()