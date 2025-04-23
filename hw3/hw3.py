# AMS 316: HW Set 3
# Name: Joe Martinez
# SBUID: 11241692
# NetID: jjmmartinez

import sys, random, math
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.affinity import rotate, translate
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

W = 1   # distance between lines
NUM_NEEDLES = 1000 # number of lines on the board
NUM_DISKS = 4444444 # number of disks
VERBOSE = False
GRAPHS = False


def buffon_needles(disks: int, disk_d: float, needles: int, line_width: float) -> list[int]:
    '''
    This function will report a list of integers that indicate the number of randomly throwing disks on a 
    plane of lines, seperated by some distance w, that touch i lines, where i is the index of the array.
    Therefore, the first index of this array is the number of disks that touch 0 lines, then 1 lines, etc...

    Parameters:
    disks (int): The number of disks to simulation.
    disk_r (int or float): The radius of the disk.
    needles (int): The number of needles.
    line_width (int or float): The width between consecutive lines.

    Returns:
    list: The number of disks that touch i lines, where i is the index of the array.
    '''
    print("\nBuffon's Needles:")
    print(f"   Disk Diameter: {disk_d}")
    if VERBOSE:
        print(f"   Disks: {disks}")
        print(f"   Needles: {needles}")
        print(f"   Line Width: {line_width}")
    percent_unit = (int)(disks / 100) # for output percentage

    # Calculate maximum number of lines 
    disk_r = disk_d / 2
    min_lines = (int)((disk_d) / line_width)
    max_lines = min_lines + 1
    # Create array to bin disks into by the number of lines they touch
    disksTouchLines = [0] * 5 #(max_lines + 1)
    last_line = (needles - 1) * line_width

    # For each disk
    for i in range(0, disks):
        lines = 0   
        step = 0    # ith set of closest lines
        d1 = 0
        d2 = 0
        # While the last iteration distances where within the disk
        while (d1 < disk_r or d2 < disk_r):
            x = random.uniform(0, last_line)    # random circle center
            prev_line =  math.floor(x - step)   # ith line to the left
            next_line = math.ceil(x + step)     # ith line to the right
            d1 = x - prev_line    # distance to the ith line to the left (less than x)
            d2 = next_line - x    # distance to the ith line to the right (greater than x)

            # If disk is on top of a line
            if d1 == 0 or d2 == 0:  lines += 1
            else:
                # The disk touches the left line
                if (d1 < disk_r and prev_line >= 0):            lines += 1
                # The disk touches the right line
                if (d2 < disk_r and next_line <= last_line):    lines += 1
            # Look at the next closest set of lines
            step += 1

        if (i % percent_unit) == 0: 
            print(f"Progress: {(int)((i/disks)*100)}%\r", end="")
        disksTouchLines[lines] += 1

    # If a line touched n+1 lines, it also touched n lines (except n = 0)
    for i in range(max_lines, 1, -1):
        disksTouchLines[i-1] += disksTouchLines[i]
    print("Progress: 100%")
    return disksTouchLines

def getLinePercents(lines, map):
    percents = []
    for key in map:
        percents.append(map[key][lines])
    return percents

def simulation_buffon_needles():
    d_values = [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10, 10/10, 15/10, 20/10, 30/10]
    reults_map = {}
    for d in d_values:
        results = buffon_needles(NUM_DISKS, d, NUM_NEEDLES, W)
        percents = list(map(lambda x: x/NUM_DISKS, results))
        reults_map[d] = percents
        if VERBOSE:
            print(results)
            print(percents)

    print("\nResults Map: d = {")
    for key in reults_map:
        print(f"  {key}: {reults_map[key]}")
    print("}")

    if GRAPHS:
        for l in range(0, 5):
            plt.plot(d_values, getLinePercents(l, reults_map))
            plt.title(f"The probability of a Disk Landing on {l} Line{'s' if (l != 1) else ''}\nvs. the Diameter of the disk")
            plt.ylabel('Probability')
            plt.xlabel('Disk Diameter')
            plt.show()



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
    
    # Close the loop manually
    points = list(zip(x, y))
    points.append(points[0])
    
    return Polygon(points).buffer(0)


def create_cutter(width=1, height=(1/np.sqrt(2)), angle=0, x0=0, y0=0):
    rect = Polygon([(-width/2, -height/2), (width/2, -height/2),
                    (width/2, height/2), (-width/2, height/2)])
    rect = rotate(rect, angle, origin=(0, 0), use_radians=True)
    rect = translate(rect, xoff=x0, yoff=y0)
    return rect


def negative_intersection_area(params, rose):
    x, y, angle = params
    cutter = create_cutter(angle=angle, x0=x, y0=y)
    intersection = rose.intersection(cutter)
    return -intersection.area  # we want to maximize area, so minimize negative


def plot_shape(ax, shape, title, label, color='red'):
    if isinstance(shape, MultiPolygon):
        labeled = False
        for poly in shape.geoms:
            x, y = poly.exterior.xy
            if not labeled:
                ax.plot(x, y, color=color, label=label)
                labeled = True
            else:
                ax.plot(x, y, color=color)
    else:
        x, y = shape.exterior.xy
        ax.plot(x, y, color=color)
    
    ax.set_aspect('equal')
    ax.set_title(title)


def simulate_rose_area():
    print("\nSimulating Optimal Rose Curve Area Cut:")
    # Create rose graph polygon
    rose = rose_curve().buffer(0)

    # Initial guess and bounds: x, y in [-0.25, 0.25], angle in [0, 2π]
    init_x = random.uniform(-0.25, 0.25)
    init_y = random.uniform(-0.25, 0.25)
    init_theta = random.uniform(0, 2 * np.pi)
    x0 = [init_x, init_y, init_theta]
    bounds = [(-0.25, 0.25), (-0.25, 0.25), (0,  2 * np.pi)]

    # Run optimizer
    result = minimize(
        negative_intersection_area,
        x0,
        args=(rose,),
        method='L-BFGS-B',
        bounds=bounds
    )

    # Extract best cutter
    best_x, best_y, best_angle = result.x
    best_cutter = create_cutter(angle=best_angle, x0=best_x, y0=best_y)
    max_area = -result.fun
    print(f"Optimized Area Cut: {max_area}")
    print(f"Rectangle placed a ({best_x},{best_y}) at {best_angle} radians.\n")

    # Plotting
    if GRAPHS:
        fig, ax = plt.subplots()
        plot_shape(ax, rose, f'Optimized Area Cut = {max_area:.4f}', 'Rose Curve')

        cutter_x, cutter_y = best_cutter.exterior.xy
        ax.plot(cutter_x, cutter_y, color='blue', label='Best Cutter')

        ax.set_aspect('equal')
        plt.legend()
        plt.show()

  

def main():
    global VERBOSE, GRAPHS
    tests = [False, False]
    ### BEGIN SWITCHBOARD ###
    if len(sys.argv) > 1:
        if '-v' in sys.argv: VERBOSE = True
        if '-g' in sys.argv: GRAPHS = True

        flag = sys.argv[1]
        if flag == '-p':
            if len(sys.argv) < 2:
                print("ERROR: performance flag must specify test to run")
                sys.exit(1)
            else:
                test = sys.argv[2]
                if test == '-b':    tests[0] = True
                elif test == '-r':  tests[1] = True
    ### END SWITCHBOARD ###
    perform = ((True not in tests) or (False not in tests))

    # Simulate Buffon's needles Data
    if (perform or tests[0]):
        simulation_buffon_needles()

    # Simulate Rose Curve needles
    if (perform or tests[1]):
        simulate_rose_area()


if __name__ == "__main__":

    main()