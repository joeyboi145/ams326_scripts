
# Repository for AMS 326, Spring 2025 Scripts
Student: Joe Martinez

email: joe.martinez@stonybrook.edu

netID: jjmmartinez

SBU ID: 112416928


# Homework 2: Numerical Integration and Linear Algebra
This homework set explores finding the approximate area of a kidney function with a circle carved out using the Midpoint/Rectangle method
and the Trapezoid method. The program computes the approximate area by dividing a symmetric part of the kidney into 3 parts, and then adding the areas to together. Additionally, this homework asks us to solve a matrix equation where the matrix is randomly generated uniformly distributed matrix and the output vector is a vectors of ones.

### Python script *hw2_source.py* libraries/requirements:
- Numpy
- Matplotlib
- Sympy

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
-   -a  --algebra       Performs solving randomly generated matrix equation

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