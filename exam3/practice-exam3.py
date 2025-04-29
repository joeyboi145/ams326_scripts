# AMS 326: PRACTICE Exam 3
# Name: Joe Martinez
# SBUID: 11241692
# NetID: jjmmartinez

import sys, random, math
import matplotlib.pyplot as plt

''' 
NOTE

Monte Carlo Algorithms:
1. Understand simple sampling MC algorithm
2. Study Recursive stratified sampling MC Algorithm
3. Study MISER MC Algorithm
4. Study VEGAS MC Algorithm

Integration Methods:
1. Forward Euler Method
2. Backward Euler Method
3. Midpoint Method
4. Heun's Method
5. Runge-Kutta RK3 Method
6. Runge-Kutta RK4 Method (OR RKI General Method) 
7. Yoshida Algorithm 
'''


def forward_euler_method(f, x0, y0, h, x_end, y_end):
    '''
        FORWARD Euler method, positive h
    '''
    x = x0
    y = y0
    x_trajectory = []
    y_trajectory = []
    while (x <= x_end or y <= y_end):
        y += h * f(x, y)
        x += h
        if (x <= x_end and y <= y_end):
            x_trajectory.append(x)
            y_trajectory.append(y)
            print(f"Progress: {(int)(((x0 - x)/100)*100)}%\r", end="")

    return [x_trajectory, y_trajectory]


def backward_euler_method(f, x0, y0, h, x_end, y_end):
    '''
        BACKWARDS EULER METHOD, negative h
    '''
    x = x0
    y = y0
    x_trajectory = []
    y_trajectory = []
    while (x >= x_end and y >= y_end):
        y += h * f(x + h, y + h)
        x += h
        if (x >= x_end and y >= y_end):
            x_trajectory.append(x)
            y_trajectory.append(y)
            print(f"Progress: {(int)(((x0 - x)/100)*100)}%\r", end="")

    return [x_trajectory, y_trajectory]


def midpoint_method(f, x0, y0, h, x_end, y_end):
    '''
        Midpoint Method for a future value, positive h
    '''
    x = x0
    y = y0
    x_trajectory = []
    y_trajectory = []
    while (x >= x_end and y >= y_end):
        y += h*f(x + (h/2), y + (h/2)*f(x,y))
        x += h
        if (x >= x_end and y >= y_end):
            x_trajectory.append(x)
            y_trajectory.append(y)
            print(f"Progress: {(int)(((x0 - x)/100)*100)}%\r", end="")

    return [x_trajectory, y_trajectory]


def heun_method(f, x0, y0, h, x_end, y_end):
    '''
        Heun method for a future value, positive h
    '''
    x = x0
    y = y0
    x_trajectory = []
    y_trajectory = []
    while (x >= x_end and y >= y_end):
        y += (h/2)*(f(x,y) + f(x + h, y + h * f(x, y)))
        x += h
        if (x >= x_end and y >= y_end):
            x_trajectory.append(x)
            y_trajectory.append(y)
            print(f"Progress: {(int)(((x0 - x)/100)*100)}%\r", end="")

    return [x_trajectory, y_trajectory]


def heun_euler_method(f, x0, y0, h, x_end, y_end):
    '''
        Heun Improved Euler Method for a future value, positive h
    '''
    x = x0
    y = y0
    x_trajectory = []
    y_trajectory = []
    while (x >= x_end and y >= y_end):
        y += (h/4) * (f(x,y) + 3 * f(x + (2/3)*h, y + (2/3)*h*f(x,y)))
        x += h
        if (x >= x_end and y >= y_end):
            x_trajectory.append(x)
            y_trajectory.append(y)
            print(f"Progress: {(int)(((x0 - x)/100)*100)}%\r", end="")

    return [x_trajectory, y_trajectory]


def runge_kutta3(f, x0, y0, h, x_end, y_end):
    '''
        Runge-Kutta (RK3) Method for a future value, positive h
    '''
    x = x0
    y = y0
    x_trajectory = []
    y_trajectory = []
    while (x >= x_end and y >= y_end):
        k1 = f(x, y)
        k2 = f(x + (h/2), y + ((h/2) * k1)) 
        k3 = f(x + h, y - (h*k1) + (2*h*k2)) 

        y += (h/6) * (k1 + (4*k2) + k3)
        x += h

        if (x >= x_end and y >= y_end):
            x_trajectory.append(x)
            y_trajectory.append(y)
            print(f"Progress: {(int)(((x0 - x)/100)*100)}%\r", end="")

    return [x_trajectory, y_trajectory]


def runge_kutta4(f, x0, y0, h, x_end, y_end):
    '''
        Runge-Kutta (RK4) Method for a future value, positive h
    '''
    x = x0
    y = y0
    x_trajectory = []
    y_trajectory = []

    while (x >= x_end and y >= y_end):
        k1 = f(x, y)
        k2 = f(x + (h/2), y + ((h/2) * k1)) 
        k3 = f(x + (h/2), y + ((h/2) * k2)) 
        k4 = f(x + h, y + h*k3)

        y += (h/6) * (k1 + (2*k2) + (2*k3) + k4)
        x += h

        if (x >= x_end and y >= y_end):
            x_trajectory.append(x)
            y_trajectory.append(y)
            print(f"Progress: {(int)(((x0 - x)/100)*100)}%\r", end="")

    return [x_trajectory, y_trajectory]


w1 = 1 / (2 - 2**(1/3))
w0 = 1 - 2 * w1
def yoshida_step(x, p, h):
    def gradV(x): return x  # since V(x) = 1/2 * x^2 => dV/dx = x

    for w in [w1, w0, w1]:
        p -= 0.5 * w * h * gradV(x)
        x += w * h * p
        p -= 0.5 * w * h * gradV(x)
    return x, p



k = 44 / 88
def plane_DE(x: float, y: float):
    '''
    Our differential equation for the trajectory of the plane.

    Returns: 
    float: the value of the equation with a given x and y values
    '''
    return (y / x) - (k * math.sqrt(1 + (y / x) ** 2))


def main():
    # results = backward_euler_method(plane_DE, 100, 0, -0.0001, 0, 0)
    # plt.plot(results[0], results[1], color='k', label='Backwards Euler')
    results = forward_euler_method(lambda x, y: (x + y + x*y), 0, 1, 0.0001, 1.4, 12)
    print(results[0][0], results[1][0])
    plt.plot(results[0], results[1], color='b', label='Forward Euler')

    # results = runge_kutta3(plane_DE, 100, 0, -0.0001, 0, 0)
    # plt.plot(results[0], results[1], color='r', label='RK3')
    # results = runge_kutta4(plane_DE, 100, 0, -0.0001, 0, 0)
    # plt.plot(results[0], results[1], color='b', label='RK4')
    # results = heun_euler_method(plane_DE, 100, 0, -0.0001, 0, 0)
    # plt.plot(results[0], results[1], color='y', label='Heun Euler')
    # results = heun_method(plane_DE, 100, 0, -0.0001, 0, 0)
    # plt.plot(results[0], results[1], color='c', label='Heun')
    # results = midpoint_method(plane_DE, 100, 0, -0.0001, 0, 0)
    # plt.plot(results[0], results[1], color='k', label='Midpoint')

    plt.legend()
    plt.show()


if __name__ == '__main__':


    main()