# AMS 316: HW Set 2
# Name: Joe Martinez
# SBUID: 11241692
# NetID: jjmmartinez

# Source Code found at:
# https://github.com/joeyboi145/ams326_scripts/blob/main/hw1.py

import math, sys, warnings
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


## DIFFERENT SECTIONS
#   SECTION 1:  x < (0.25 - sqrt(0.125)) UNTIL y = NAN
#       1.1 -   ADD area under k4 (kidney_top_left)
#       1.2 -   SUBTRACT area under k3 (kidney_bottom_left)

#   SECTION 2:  (0.25 - sqrt(0.125)) < x < 0.5 AND y > 0.25
#       2.1 -   ADD area under k4 (kidney_top_left) and k2 (kidney_top_right)
#       2.2 -   SUBTRACT area under c1 (circle_upper)
#
#   SECTION 3:  (0.25 - sqrt(0.125)) < x < 0 AND y < 0.25
#       3.1 -   ADD area under c2 (circle_lower)
#       3.2 -   SUBTRACT area under k3 (kidney_bottom_left) 

def main():
    graphs = False
    verbose = False

    if len(sys.argv) >= 2:
        flag = sys.argv[1]
        if flag == '-g' or flag == "--graph":  graphs = True
        if flag == '-v' or flag == "--verbose":  verbose = True

        help_menu = "Usage: \
                \n  -v  --verbose\tProvides additional information and the value of each iteration for each method \
                \n  -h  --help\tUsage menu \
                \n  -g  --graph\tDisplay graphs "
        
        if flag == '-h' or flag == "--help": 
            print(help_menu)
            sys.exit(0)

    # Define symbols
    x, y = sp.symbols('x y')

    # Define the implicit equation
    kidney = sp.Eq((x**2 + y**2)**2, x**3 + y**3)
    kidney_solutions = sp.solve(kidney, y)
    print(f"KIDNEY SOLUTIONS NUMBER: {len(kidney_solutions)}")
    k1 = sp.lambdify(x, kidney_solutions[0], modules='numpy', printer=None)   # Not used for calculations
    k2 = sp.lambdify(x, kidney_solutions[1], modules='numpy')  
    k3 = sp.lambdify(x, kidney_solutions[2], modules='numpy')  
    k4 = sp.lambdify(x, kidney_solutions[3], modules='numpy')
    if verbose:
        for solution in kidney_solutions:
            print("\n")
            print(solution)
        

    circle = sp.Eq((x-0.25)**2 + (y-0.25)**2 , 0.125)
    circle_solutions = sp.solve(circle, y)
    print(f"CIRCLE SOLUTIONS NUMBER: {len(circle_solutions)}")
    c1 = sp.lambdify(x, circle_solutions[0], modules='numpy')  
    c2 = sp.lambdify(x, circle_solutions[1], modules='numpy')  
    if verbose:
        for solution in circle_solutions:
            print("\n")
            print(solution)


    # THIS IS HOW TO TELL WHEN TO SWITCH
    # x1 = 0.34
    # x2 = 0.36
    # print(f"result: {k4(x1)}")
    # print(f"result: {k4(x2)}")


    if graphs:
        X_SPACE = np.linspace(-0.3, 1, 10000)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Kidney Equation 
            YK_lower_right = k1(X_SPACE)    # NOT USED FOR CALCULATIONS
            YK_top_right = k2(X_SPACE)
            YK_lower_left = k3(X_SPACE)
            YK_top_left = k4(X_SPACE)
            plt.plot(X_SPACE, YK_lower_right, label="Kidney", color="r")
            plt.plot(X_SPACE, YK_top_right, color="g")
            plt.plot(X_SPACE, YK_lower_left, color="c")
            plt.plot(X_SPACE, YK_top_left, color="y")

            # Circle Equation
            YC_upper = c1(X_SPACE)
            YC_lower = c2(X_SPACE)
            plt.plot(X_SPACE, YC_upper, label='Circle', color='b')
            plt.plot(X_SPACE, YC_lower, color='b')

        plt.plot(X_SPACE, X_SPACE, label='x=y', color='black', linewidth=0.5)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

if __name__ == "__main__":
    main()