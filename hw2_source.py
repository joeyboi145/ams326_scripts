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


TOL = math.pow(10, -4)
P_ERROR = (1/2) * (1/6) * 0.5 * (TOL + 2.51 * math.pow(10, -8))
BOUND_CORRECTION = math.pow(10, -7)
ITER_MAX = 10000
BOUND_ERROR = []
VERBOSE = False


def secant_method(f, x1: float, x2: float) -> float:
    """
    The Secant method for root approximation using two initial guesses 
    which returns a 4-decimal place accurate result and outputs the \
    number of iteration and floating point operations.

    Args:
        f (float -> float): function
        x1 (float): first initial guess for a root
        x2 (float): second initial guess for a root

    Returns:
        float: A value that approximates the root with 4 decimal place accuracy
    """
    print("SECANT METHOD:")
    if x1 == x2: 
        print("ERROR: x1 cannot equal x2")
        return

    global VERBOSE
    i = 0
    while True:
        if (VERBOSE): print(f"    Iteration {i}: x = {x2}")

        fx2 = f(x2)     # Calculate f(x2) once to save FP
        denom = (fx2 - f(x1)) / (x2 - x1)
        x3 = x2 - (fx2 / denom)
        x1 = x2
        x2 = x3
        i += 1
        error = math.fabs(x2.real - x1.real)
        if error < math.pow(10, -4): break
        elif (i >= ITER_MAX):
            print("ERROR: Reached maximum iterations. Converging too slow or not at all")
            print(f"Result: {x2} \n Error: {error}\n")
            return
    # print(f"Number of iterations: {i}")
    # print(f"Number of estimated FLOP: ~{95 * i + 1}")
    print(f"Final Result: {x2}")
    print(f"Final Error: {error}")
    return x2


def create_nodes(a: float, b: float, n: int) -> np.ndarray:
    """
    Creates a list of n nodes from a given range [a,b]

    Args:
        a (float): start of the range, first node
        b (float): end of the range, last node
        n (int): number of nodes

    Returns:
        list: A list of nodes
    """
    if (n <= 1): raise ValueError("n must be two nodes or greater")
    if (b <= a): raise ValueError("b must be greater than a")

    interval = math.fabs(b - a) / (n-1)
    nodes = np.array([a])
    current_val = a
    for i in range(0, n - 2):
        current_val = current_val + interval
        nodes = np.append(nodes, current_val)
    nodes = np.append(nodes, b)
    return nodes


def midpoint_method(f, a: float, b: float, n: int, piecewise: list = None, zero = False) -> float:
    """
     The Midpoint Method of integration for a given function f
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
        piecewise.reverse()
    interval = math.fabs(b - a) / (n-1)
    nodes = create_nodes(a, b, n)
    midpoints = np.array([])
    global VERBOSE
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
            f = piecewise.pop()
            value = f(midpoint)
        sum += f(midpoint)
        if (VERBOSE): print(f"    OUTPUT: {value}, complex: {np.imag(value)}")
    return np.real(interval * sum)


def trapezoid_method(f, a: float, b: float, n: int, piecewise: list = None, zero = False) -> float:
    """
    The Trazpezoid Method of integration for a given function f
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
        piecewise.reverse()
    interval = math.fabs(b - a) / (n-1)
    nodes = create_nodes(a, b, n)
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
            f = piecewise.pop()
            value = f(node)
        
        if (i == 0 or i == len(nodes)): sum += f(node)
        else: sum += 2*f(node)
        if (VERBOSE): print(f"    OUTPUT: {value}, complex: {np.imag(value)}")

    sum = (interval / 2) * sum
    return np.real(sum)
    

def find_area(method, f, a: float, b: float, n: int, 
              piecewise: list = None, zero=False) -> float:
    """
    Calculates the area under the curve 'f' in a given range [a,b] using the 
    Midpoint Method. It first started with the given nodes n, but increases
    them by a factor of two for each iteration until an area with a given accuracy
    tolerance

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
    a1 = method(f, a, b, n, piecewise=piecewise, zero=zero)
    a2 = None
    i = 0
    error = None
    global VERBOSE

    while True:
        n = 2 * n
        a2 = method(f, a, b, n, piecewise=piecewise, zero=zero)
        error = math.fabs((a1 - a2) / 3)
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


def test_suite():
    '''
    a = 0
    b = 1
    nodes = create_nodes(a, b, 5);
    print(nodes)
    '''

    # '''
    a = 0
    b = 2

    # Example Function e^(x+2)
    def f(x):
        return math.exp(x+2)
    
    real_value = math.exp(4) - math.exp(2)
    n = 5
    a1 = trapezoid_method(f, a, b, n)
    a2 = None
    i = 0
    error = None

    while True:
        n = 2 * n
        a2 = trapezoid_method(f, a, b, 2*n)
        error = math.fabs((a1 - a2) / 3)
        i += 1
        if (error < P_ERROR): break
        a1 = a2
        print(f"Iternation {i}:\n  First Area: {a1}\n  Second Area: {a2}\n  Nodes: {int(n/2)} vs {n}\n  Error: {error}")

    a1 = math.fabs((a1 - 4*a2) / 3) # Richardson Extrapolation
    print(f"Iteration {i}:")
    print(f"Final Area: {a1}")
    print(f"Final Error: {error}, {error < P_ERROR}")
    print(f"Real Error: {real_value - a1}, {math.fabs(real_value - a1) < P_ERROR}")
    print(f"Final Nodes: {n}")

    # print(midpoint_method(f, a, b, n))
    # '''


def kidney_midpoint_integration(kidney_solutions, circle_solutions, sections):
    print("\nMIDPOINT METHOD...")
    x, y = sp.symbols('x y')
    k1 = sp.lambdify(x, kidney_solutions[0], modules='numpy') 
    k2 = sp.lambdify(x, kidney_solutions[1], modules='numpy')  
    k3 = sp.lambdify(x, kidney_solutions[2], modules='numpy')  
    k4 = sp.lambdify(x, kidney_solutions[3], modules='numpy')
    c1 = sp.lambdify(x, circle_solutions[0], modules='numpy')  
    c2 = sp.lambdify(x, circle_solutions[1], modules='numpy')  
    all_sections = not (True in sections)
    section_1_area = section_2_area = section_3_area = 0

    # FINDING THE MINUMUM X
    # 1. We take the derivative of k1 (kidney_lower_right)
    # 2. Evaluate where it equals to 0 (optimization problem)
    # 3. Solve this k1(x) to find the minimum y
    # 4. Reflect this over x=y, therefore the minimum y is the minimum x
            
    print("\nFINDING MINIMUM X VALUE:")
    kidney_LR_derivative = sp.diff(kidney_solutions[0], x)
    dk1 = sp.lambdify(x, kidney_LR_derivative, modules='numpy')
    sol_x = secant_method(dk1, 0.62, 0.625)
    min_y = k1(sol_x)
    min_x = np.real(min_y)
    print(f"EVALUATE dk1(min_x): {dk1(sol_x)}")
    print(f"MIN x: {min_x}")
    # Due to the reflective property about x=y of the kidney function
    # this minimum y value is the minimum x value for the left side of the kueney


## AREA OF DIFFERENT SECTIONS
#   SECTION 1:  min_x < x < (0.25 - sqrt(0.125))
#       1.1 -   ADD area under k4 (kidney_top_left)
#       1.2 -   SUBTRACT area under k3 (kidney_bottom_left)
#   - No need for bound correction since calculations are done usings midpoints
    
    if (all_sections or sections[0]):
        max_x = 0.25 - math.sqrt(0.125)

        print("\nAREA OF SECTION 1.1 TOP KIDNEY CURVE:")
        whole_area = find_area(midpoint_method, k4, min_x, max_x, 20)
        print("\nAREA OF SECTION 1.2 BOTTOM KIDNEY CURVE:")
        subtract_area = find_area(midpoint_method, k3, min_x, max_x, 20)
        section_1_area = whole_area - subtract_area

    ## SOME RESULTS
    # 2 -> 0.10693920036921889
    # 3 -> 0.10692998866820072
    # 5 -> 0.10693350045373426
    # 1000 -> 0.10693350045373426


#   SECTION 2:  (0.25 - sqrt(0.125)) < x < 0.5 AND y > 0.25
#       2.1 -   ADD area under k4 (kidney_top_left) and k2 (kidney_top_right)
#       2.2 -   SUBTRACT area under c2 (circle_upper)
#   - No need for bound correction since calculations are done usings midpoints
    if (all_sections or sections[1]):
        x1 = 0.25 - math.sqrt(0.125)
        x2 = 0.5

        print("\nAREA OF SECTION 2.1 TOP KIDNEY CURVE:")
        whole_area = find_area(midpoint_method, k4, x1, x2, 20, piecewise=[k2])
        print("\nAREA OF SECTION 2.2 UPPER CIRCLE CURVE:")
        subtract_area = find_area(midpoint_method, c2, x1, x2, 20)
        section_2_area = whole_area - subtract_area

    ## SOME RESULTS:
    # 20 -> 0.1854595627693757


#   SECTION 3:  (0.25 - sqrt(0.125)) < x < 0 AND y < 0.25
#       3.1 -   ADD area under c1 (circle_lower)
#       3.2 -   SUBTRACT area under k3 (kidney_bottom_left) 
#   - No need for bound correction since calculations are done usings midpoints
    if (all_sections or sections[2]):
        x1 = 0.25 - math.sqrt(0.125)
        x2 = 0

        print("\nAREA OF SECTION 3.1 LOWER CIRCLE CURVE:")
        whole_area = find_area(midpoint_method, c1, x1, x2, 20)        
        print("\nAREA OF SECTION 3.2 BOTTOM KIDNEY CURVE:")
        subtract_area = find_area(midpoint_method, k3, x1, x2, 20)
        section_3_area = whole_area - subtract_area

    # SOME RESULTS:
    # 20 -> 0.002130206649589882
        

    print("")
    if all_sections or sections[0]: print(F"SECTION 1 RESULT: {section_1_area}")
    if all_sections or sections[1]: print(F"SECTION 2 RESULT: {section_2_area}")
    if all_sections or sections[2]: print(F"SECTION 3 RESULT: {section_3_area}")

    if (all_sections):
        total_area = 2 * (section_1_area + section_2_area + section_3_area) 
        # print(f"\n\nFINAL AREA (MIDPOINT): {total_area}\n\n")
        return total_area

def kidney_trapezoid_integration(kidney_solutions, circle_solutions, sections):
    print("\nTRAPEZOID METHOD...")
    x, y = sp.symbols('x y')
    k1 = sp.lambdify(x, kidney_solutions[0], modules='numpy') 
    k2 = sp.lambdify(x, kidney_solutions[1], modules='numpy')  
    k3 = sp.lambdify(x, kidney_solutions[2], modules='numpy')  
    k4 = sp.lambdify(x, kidney_solutions[3], modules='numpy')
    c1 = sp.lambdify(x, circle_solutions[0], modules='numpy')  
    c2 = sp.lambdify(x, circle_solutions[1], modules='numpy')  
    all_sections = not (True in sections)
    section_1_area = section_2_area = section_3_area = 0
    section_2_error = section_3_error = 0
    global BOUND_CORRECTION, BOUND_ERROR

    # FINDING THE MINUMUM X
    # 1. We take the derivative of k1 (kidney_lower_right)
    # 2. Evaluate where it equals to 0 (optimization problem)
    # 3. Solve this k1(x) to find the minimum y
    # 4. Reflect this over x=y, therefore the minimum y is the minimum x
            
    print("\nFINDING MINIMUM X VALUE:")
    kidney_LR_derivative = sp.diff(kidney_solutions[0], x)
    dk1 = sp.lambdify(x, kidney_LR_derivative, modules='numpy')
    sol_x = secant_method(dk1, 0.62, 0.625)
    min_y = k1(sol_x)
    min_x = np.real(min_y)
    print(f"EVALUATE dk1(min_x): {dk1(sol_x)}")
    print(f"MIN x: {min_y}")
    # Due to the reflective property about x=y of the kidney function
    # this minimum y value is the minimum x value for the left side of the kueney



## AREA OF DIFFERENT SECTIONS
#   SECTION 1:  min_x < x < (0.25 - sqrt(0.125))
#       1.1 -   ADD area under k4 (kidney_top_left)
#       1.2 -   SUBTRACT area under k3 (kidney_bottom_left)
#   - No bound correction needed (yet) since solution is approximate
    
    if (all_sections or sections[0]):
        max_x = 0.25 - math.sqrt(0.125)
        # BOUND_CORRECTION = math.pow(10, -7)
        
        print("\nAREA OF SECTION 1.1 TOP KIDNEY CURVE:")
        whole_area = find_area(trapezoid_method, k4, min_x, max_x, 20)
        print("\nAREA OF SECTION 1.2 BOTTOM KIDNEY CURVE:")
        subtract_area = find_area(trapezoid_method, k3, min_x, max_x, 20)
        section_1_area = whole_area - subtract_area

    ## SOME RESULTS
    # 2 -> 0.10693920036921889
    # 3 -> 0.10692998866820072
    # 5 -> 0.10693350045373426
    # 1000 -> 0.10693350045373426


#   SECTION 2:  (0.25 - sqrt(0.125)) < x < 0.5 AND y > 0.25
#       2.1 -   ADD area under k4 (kidney_top_left) and k2 (kidney_top_right)
#       2.2 -   SUBTRACT area under c1 (circle_upper)
#   - Bound correction needed for the upper circle curve
#       Therefore, the value will be a greater by a maximum of the calculated error 
    if (all_sections or sections[1]):
        x1 = 0.25 - math.sqrt(0.125)
        x2 = 0.5
        print("\nAREA OF SECTION 2.1 TOP KIDNEY CURVE:")
        whole_area = find_area(trapezoid_method, k4, x1, x2, 20, piecewise=[k2])
        print("\nAREA OF SECTION 2.2 UPPER CIRCLE CURVE:")
        subtract_area = find_area(trapezoid_method, c2, x1 + BOUND_CORRECTION, x2, 20)
        section_2_area = whole_area - subtract_area
        section_2_error = c2(x1 + BOUND_CORRECTION) * BOUND_CORRECTION
        BOUND_ERROR.append(section_2_error)

    ## SOME RESULTS:
    # 2 -> 0.48724335298516297
    # 3 -> 0.48723966188522994
    # 5 -> 0.48723625806601256
    # 1000 -> 0.4872498387064131


#   SECTION 3:  (0.25 - sqrt(0.125)) < x < 0 AND y < 0.25
#       3.1 -   ADD area under c1 (circle_lower)
#       3.2 -   SUBTRACT area under k3 (kidney_bottom_left) 
#   - Bound correction needed for the lower circle curve, 
#       Thereofore value will be short by a max of the calculated error
#   - Due to function definition, x = 0 will cause an indeterminate form
#       but the value will be 0 using either equation
    if (all_sections or sections[2]):  
        x1 = 0.25 - math.sqrt(0.125)
        x2 = 0
        print("\nAREA OF SECTION 3.1 LOWER CIRCLE CURVE:")
        whole_area = find_area(trapezoid_method, c1, x1 + BOUND_CORRECTION, x2, 20, zero=True)
        print("\nAREA OF SECTION 3.2 BOTTOM KIDNEY CURVE:")
        subtract_area = find_area(trapezoid_method, k3, x1, x2, 20, zero=True)
        section_3_area = whole_area - subtract_area
        section_3_error = c1(x1 + BOUND_CORRECTION) * BOUND_CORRECTION
        BOUND_ERROR.append(section_3_error)
       

    # SOME RESULTS:
    # 2 -> 0.037814791438147415
    # 3 -> 0.03780787562017857
    # 5 -> 0.03781339320013076
    # 1000 -> 0.03780748860295236

    print("")
    if all_sections or sections[0]: print(F"SECTION 1 RESULT: {section_1_area}")
    if all_sections or sections[1]: 
        print(f"SECTION 2 RESULT: {section_2_area}")
        print(f"SECTION 2 BOUND ERROR: {section_2_error}")
    if all_sections or sections[2]: 
        print(f"SECTION 3 RESULT: {section_3_area}")
        print(f"SECTION 3 BOUND ERROR: {section_3_error}")

    if (all_sections):
        total_area = 2 * (section_1_area + section_2_area + section_3_area) 
        # print(f"\n\nFINAL AREA (MIDPOINT): {total_area}\n\n")
        return total_area

def show_graphs(kidney_solutions, circle_solutions, sections):
        x, y = sp.symbols('x y')
        k1 = sp.lambdify(x, kidney_solutions[0], modules='numpy') 
        k2 = sp.lambdify(x, kidney_solutions[1], modules='numpy')  
        k3 = sp.lambdify(x, kidney_solutions[2], modules='numpy')  
        k4 = sp.lambdify(x, kidney_solutions[3], modules='numpy')
        c1 = sp.lambdify(x, circle_solutions[0], modules='numpy')  
        c2 = sp.lambdify(x, circle_solutions[1], modules='numpy')  
        X_SPACE = np.linspace(-0.3, 1, 10000)
        all_sections = not (True in sections) or not (False in sections)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # Kidney Equation 

            if (all_sections):
                YK_lower_right = k1(X_SPACE)    
                plt.plot(X_SPACE, YK_lower_right, label="Kidney", color="r")

            if (all_sections or sections[1]):
                YK_top_right = k2(X_SPACE)
                plt.plot(X_SPACE, YK_top_right, color="g")

            if (all_sections or sections[0] or sections[2]):
                YK_lower_left = k3(X_SPACE)
                plt.plot(X_SPACE, YK_lower_left, color="c")

            if (all_sections or sections[1] or sections[0]):
                YK_top_left = k4(X_SPACE)
                plt.plot(X_SPACE, YK_top_left, color="y")

            # Circle Equation
            if (all_sections or sections[2] or sections[0]):
                YC_upper = c1(X_SPACE)
                plt.plot(X_SPACE, YC_upper, label='Circle', color='b')
            if (all_sections or sections[1] or sections[0]):
                YC_lower = c2(X_SPACE)
                plt.plot(X_SPACE, YC_lower, color='b')

        plt.plot(X_SPACE, X_SPACE, label='x=y', color='black', linewidth=0.5)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


def main():
    global VERBOSE, BOUND_ERROR
    graphs = False
    perform = True
    evalute = [False, False, False]
    sections = [False, False, False]

    ### BEGIN SWITCHBOARD
    if len(sys.argv) >= 2:
        flag = sys.argv[1]
        help_menu = "Usage: \
                \n  -v  --verbose\tProvides additional information about functions and approximation method iterations \
                \n  -h  --help\tUse and help menu \
                \n  -g  --graph\tDisplay graphs \
                \n   | -s \t- To specify specific sections of the graph to load \
                \n  -t  --test\tRun test suite \
                \n  -p  --perform\tRun specific methods or functions\
                \n   | -m  --midpoint\t- Run Midpoint Method \
                \n   | -t  --trapezoid\t- Run Trapezoid Method \
                \n   |  |   -s\t To specify specific sections of the method to run ('13' or '2')" 
        
        if '-g' in sys.argv or '--graph' in sys.argv: graphs = True
        if '-v' in sys.argv or '--verbose' in sys.argv: VERBOSE = True
        
        if flag == '-h' or flag == "--help": 
            print(help_menu)
            sys.exit(0)

        if flag == '-t' or flag == "--test":
            test_suite()
            sys.exit(0)

        if flag == '-p' or flag == '--perform':
            if (len(sys.argv) == 2):
                print("ERROR: Please specify the method or problem to run.")
                sys.exit(1)
            
            method = sys.argv[2]
            if method == '-t' or method == '--trapezoid':
                evalute[1] = True
            elif method == '-m' or method == '--midpoint':
                evalute[0] = True
            if True in evalute: perform = False

        if len(sys.argv) >= 4:
            section_flags = sys.argv[3]
            for c in section_flags:
                sections[int(c) - 1] = True
    ### END SWTICHBOARD

    print("\n\nFINDING EXPLICIT FORMULAS:")
    # Define symbols
    x, y = sp.symbols('x y')
    # Define the implicit equation
    kidney = sp.Eq((x**2 + y**2)**2, x**3 + y**3)
    kidney_solutions = sp.solve(kidney, y)
    print(f"Kidney solution number: {len(kidney_solutions)}")
    if VERBOSE:
        for solution in kidney_solutions:
            print("\n")
            print(solution)
        

    circle = sp.Eq((x-0.25)**2 + (y-0.25)**2 , 0.125)
    circle_solutions = sp.solve(circle, y)
    print(f"Cirlce solution number: {len(circle_solutions)}") 
    if VERBOSE:
        for solution in circle_solutions:
            print("\n")
            print(solution)
            

    if (perform or evalute[0]):
        midpoint_area = kidney_midpoint_integration(kidney_solutions, circle_solutions, sections)

    if (perform or evalute[1]):
        trapezoid_area = kidney_trapezoid_integration(kidney_solutions, circle_solutions, sections)

    print("\n")
    if ((perform or evalute[0]) and not (True in sections)):
        print(f"Total Midpoint Area: {midpoint_area}")

    if ((perform or evalute[1]) and not (True in sections)):
        print(f"Total Trapezoid Area: {trapezoid_area}")
        print(f"  Additional Error from Bound Correction: [+{BOUND_ERROR[1]}, -{BOUND_ERROR[0]}]")

        # Since both errors subtract from each other and the greater error among the two 
        # is having a calculate area > 2.50...e-8 from the real value...
        # We will consider this as the maximum error that can occur from bound reduction
        print(f"  [{trapezoid_area + BOUND_ERROR[1]}, {trapezoid_area - BOUND_ERROR[0]}]")

    print("\n")

    if graphs:
        show_graphs(kidney_solutions, circle_solutions, sections)

if __name__ == "__main__":
    main()