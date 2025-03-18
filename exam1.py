# AMS 316: Exam 1
# Name: Joe Martinez
# SBUID: 11241692
# NetID: jjmmartinez

import math, matplotlib.pyplot as plt, numpy as np

termperature = 64.89
TOL = math.pow(10, -4)
ITER_MAX = 1000000


def calculation_t_values():
    t_values = []
    for i in range(0, 12):
        t_values.append(16+31*i)
    return t_values


# Jan = 0, Feb = 1, ..., Nov = 10, Dec = 11 
# 
def calculate_t_day(month, day):
    return month * 31 + day


def cubic_fit(x: list, y: list, verbose = False) -> np.ndarray:
    """
    Returns a list of numbers that represent the coefficients of an
    fitted cubic equation that fits the list of inputed x and 
    y value.

    Args:
        x (list): A list of x values 
        y (list): A list of y values 
        verbose (boolean): if True, prints values of important matrix calculations

    Returns:
        numpy.ndarray: A list of coefficients for the fitted quadratic equation
    """
    print("CUBIC FIT:")
    equations = []
    for n in x:
        array = []
        for p in range(3, -1, -1):
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
    print(f"x: " + str(x))
    print(f"y: " + str(y))
    # print(f"Coefficient Matrix:\n" + str(A))

        
    if verbose:
        print(f"\nA_t:\n" + str(A_t))
        print(f"\nA_t * A :\n" + str(A_t @ A))
        print(f"\nA_t * y:\n" + str(A_t @ y))
    print(f"\nCoefficient Results: " + str(x_hat))
    return x_hat

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
    print(f"Final Result: {x2}")
    print(f"Final Error: {error}")
    return x2


def f(x):
    double_x = x * x
    term1 = -3.99050448 * math.pow(10, -6) * double_x * x
    term2 = 9.57687541 * math.pow(10, -4) * double_x
    term3 = 1.87139469 * math.pow(10, -1) * x
    term4 = 2.59417808 * 10
    return term1 + term2 + term3 + term4 - termperature


def main():
    a = 1
    b = calculate_t_day(11, 31)

    t_values = calculation_t_values()
    # y_min = [26, 26, 32, 42, 51, 61, 67, 66, 60, 49, 39, 31]
    y_average = [33, 34, 40, 51, 60, 69, 75, 74, 67, 56, 47, 38]
    # t_max = [40, 41, 48, 60, 69, 77, 83, 82, 75, 64, 54, 45]

    coeff = cubic_fit(t_values, y_average, True)
    x_space = np.linspace(a, b, 1000)
    y_space = np.polyval(coeff, x_space)




    print("\nEVALUTING P(t) for June 4th and December 25th")
    # CALCULATE P(t = June 4th)
    june_4 = calculate_t_day(5, 4)
    print(f"JUNE 4TH: t = {june_4}")
    temp_june_4 =np.polyval(coeff, june_4)
    print(f"\tP(t) = {temp_june_4}")


    # CALCULATE P(t = Dece 25th)
    dec_25 = calculate_t_day(11, 25)
    print(f"DEC 25TH: t = {dec_25}")
    temp_dec_25 = np.polyval(coeff, dec_25)
    print(f"\tP(t) = {temp_dec_25}")


    print("\n SOLVING P(t) = 64.89")
    t1 = secant_method(295, 284, True)
    y1 = np.polyval(coeff, t1)
    print(f"f(t1) = {y1}")
    t2 = secant_method(164, 165, True)
    y2 = np.polyval(coeff, t2)
    print(f"f(t2) = {y2}")



    plt.plot(x_space, y_space, label='P_3(t)')
    plt.plot(t_values, y_average, 'o', label='Average Data Points', markersize=8)
    plt.plot([june_4], [temp_june_4], 'o', label='P_3(t = June 4th)', markersize=8)
    plt.plot([dec_25], [temp_dec_25], 'o', label='P_3(t = Dec 25th)', markersize=8)
    plt.plot([t1, t2], [y1, y2], 'o', label='P_3(t)=64.89', markersize=8)
    plt.xlabel("Day of the year")
    plt.ylabel("Average Temperature of Day")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
