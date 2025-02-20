
# Repository for AMS 326, Spring 2025 Scripts
Student: Joe Martinez
email: joe.martinez@stonybrook.edu
netID: jjmmartinez
SBU ID: 112416928

## Homework 1: Root Approximation and Extrapolation
This homework set explores finding roots using the bisection method, Newton's method, the Secant method, and a Monte Carlo method. Additionally, given a set of 5 data points, we interpolated the data with a polynomial equation and  fitted a quadratic equation to the data.

### Program script *hw1.py* requirements:
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