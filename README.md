
# Repository for AMS 326, Spring 2025 Scripts
Student: Joe Martinez

email: joe.martinez@stonybrook.edu

netID: jjmmartinez

SBU ID: 112416928


# Homework 2: Numerical Integration and Linear Algebra
This homework set explores finding the approximate area of a kidney function with a circle carved out using the Midpoint/Rectangle method
and the Trapezoid method. The program computes the approximate area by dividing a symmetric part of the kidney into 3 parts: The left part (*section 1*) which is the area between the top and bottom curves of the kidney before the circle, the top right part (*section 2*), which is the area between the top curve of the kidney and the top curve of the cirle, and the bottom right part (*section 3*), which is the area between the bottom curve of the circle and the bottom curve of the kidney.

### Python script *hw2_source.py* libraries/requirements:
- Numpy
- Matplotlib

## Command Line Arguments: 
-   -v  --verbose       Provides additional information and the value of each iteration for each method
-   -h  --help          Usage menu
-   -g  --graph         Display graphs
-  ----   -s [123]+      Used to graph specific parts of the kidney and circle functions
-   -t  --test          Perform rudimentary tests 
-   -p  --perform       Perform specific test

## Performance Options (-p) : Choose which method to run:
-   -m  --midpoint      Performs Midpoint method for area approximation
-   -t  --trapezoid     Performs Trapezoid method for area approximation
-  ----   -s [123]+      Performs area calculation for specific parts of the kidney and circle functions

# Homework 1: Root Approximation and Extrapolation
This homework set explores finding roots using the bisection method, Newton's method, the Secant method, and a Monte Carlo method. Additionally, given a set of 5 data points, we interpolated the data with a polynomial equation and  fitted a quadratic equation to the data.

### Python script *hw1_source.py* libraries/requirements:
- Numpy
- Matplotlib

## Command Line Arguments: 
-   -v  --verbose       Provides additional information and the value of each iteration for each method
-   -h  --help          Usage menu
-   -g  --graph         Display graphs 
-   -t  --test          Perform rudimentary tests 
-   -p  --perform       Perform specific test


## Performance Options (-p) : Choose which method to run:
-   -b  --bisection         Performs bisection method only 
-   -n  --newton            Performs Newtons method only
-   -s  --secant            Performs secant method only
-   -m  --monte             Performs Monte Carlo Method only
-   -i  --interpolate       Performs polynomial interpolation only
-   -f  --fit               Performs Quadratic fit only