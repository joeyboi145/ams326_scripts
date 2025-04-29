# AMS 326: Exam 3
# Name: Joe Martinez
# SBUID: 11241692
# NetID: jjmmartinez

import random, math, shapely
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString


def rose_curve(k: float = 2, a: float = 1, num_points: int = 1000) -> Polygon:
    """
    This function creates a rose curve of the form r = a * cos(k * theta)

    Parameters:
    k (float): 'k' number
    a (float): 'a' number
    num_points (int): number of points to create curve with

    Returns:
    Polygon: A rose curve that fits the specifications 
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = a * np.sin(k * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = list(zip(x, y))
    return LineString(points)


def simulate_needles(l, num_needles):
    print(f"\nSimulating Needles:\n  length {l}\n  number: {num_needles}")
    rose_poly = rose_curve()
    
    # Needles
    crossing = 0
    count = 0
    percentUnit = (int)(num_needles / 100)
    while (count < num_needles):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        theta = random.uniform(0, 2 * math.pi)
        deltaX = (l/2) * math.cos(theta)
        deltaY = (l/2) * math.sin(theta)
        needle = LineString([(x-deltaX, y-deltaY), (x+deltaX, y+deltaY)])

        crossPoint = shapely.intersection(rose_poly, needle)
        if not crossPoint.is_empty:
            crossing += 1
        count += 1

        if (count % percentUnit == 0):
            print(f"Progress: {(int)((count / num_needles)*100)}%\r", end="")

    print("Progess: 100%")
    print('total crossing: ', crossing)
    print('probability: ', (crossing / num_needles))
    return (crossing / num_needles)

def main():
    l_length = [1/10, 1/5, 1/4, 1/3, 1/2, 1/1]
    num_needles = 444444


    for length in l_length:
        simulate_needles(length, num_needles)

if __name__ == '__main__':
    main()