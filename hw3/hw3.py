# AMS 326: HW Set 3
# Name: Joe Martinez
# SBUID: 11241692
# NetID: jjmmartinez

import sys, random, math
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.affinity import rotate, translate
from scipy.optimize import minimize

W = 1   # distance between lines
NUM_NEEDLES = 1000 # number of lines on the board
NUM_DISKS = 4444444 # number of disks
VERBOSE = False
GRAPHS = False
k = 44 / 88


def buffon_disk(disks: int, disk_d: float, needles: int, line_width: float) -> list[int]:
    '''
    This function will report a list of integers that indicate the number of randomly throwing disks on a plane of lines, seperated by some distance w, that touch i lines, where i is the index of the array. Therefore, the first index of this array is the number of disks that touch 0 lines, then 1 lines, etc...

    Parameters:
    disks (int): The number of disks to simulation.
    disk_r (int or float): The radius of the disk.
    needles (int): The number of needles.
    line_width (int or float): The width between consecutive lines.

    Returns:
    list: The number of disks that touch i lines, where i is the index of the array.
    '''
    print("Buffon's Needles:")
    print(f"   Disk Diameter: {disk_d}")
    if VERBOSE:
        print(f"   Disks: {disks}")
        print(f"   Needles: {needles}")
        print(f"   Line Width: {line_width}")
    percent_unit = (int)(disks / 100) # for output percentage

    # Calculate maximum number of lines 
    disk_r = disk_d / 2
    max_lines = (int)((disk_d) / line_width) + 1
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
    '''
    Returns a list of probabilities that indicate the probability  of disks of varying diameter to land on a specified number of line, which is given by 'lines'
    
    This function requires a mapping of disk diameter (key) to a list of probabilities (value) for a disk of that specific key diameter landing on i lines, where i is the ith index of the list.

    Parameters:
    lines (int): number of lines to base probabilities on. If for example 0, the returned list will be the probabilites of varying disk diameters landing on 0 lines
    map (Map): Mapping to disk diameter to probabilities of landing on lines

    Returns:
    list: probabilities of varying disk size landing on 'lines' number of lines
    '''
    percents = []
    for key in map:
        percents.append(map[key][lines])
    return percents

def simulation_buffon_disk():
    '''
    Simulates the Buffon Needles problem with 13 different diameter disks and outputs PDF graphs for the chances of landing on l lines based on the disk diameter.
    '''
    d_values = [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10, 10/10, 15/10, 20/10, 30/10]
    reults_map = {}
    count = 1
    print("SIMULATING MULTIPLE BUFFON NEEDLES PROBLEM")
    for d in d_values:
        print(f"\nSimulation {count}:", end='\n')
        results = buffon_disk(NUM_DISKS, d, NUM_NEEDLES, W)
        percents = list(map(lambda x: x/NUM_DISKS, results))
        reults_map[d] = percents
        count += 1
        if VERBOSE:
            print(results)
            print(percents)

    print("\nResults Map: d = {")
    for key in reults_map:
        print(f"  {key}: {reults_map[key]}")
    print("}")

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
    '''
    Creates a Rectangle of size 1 and 1/sqrt(2) by default to cut the area out of our Rose Curve

    Parameters:
    width (float): width of the rectangle, 1 by default.
    heigh (float): height of the rectangle, 1/sqrt(2) by default
    angle (float): angle to place rectangle at, 0 by default
    x0 (float): Initial x position of the rectangle: 0 by default
    y0 (float): Initial y position of the rectangle: 0 by default

    Returns:
    Polygon: The 1 by 1/sqrt(2) rectangle as a Polygon
    '''
    rect = Polygon([(-width/2, -height/2), (width/2, -height/2),
                    (width/2, height/2), (-width/2, height/2)])
    rect = rotate(rect, angle, origin=(0, 0), use_radians=True)
    rect = translate(rect, xoff=x0, yoff=y0)
    return rect


def negative_intersection_area(params, rose):
    '''
    Returns the negative area of rose curve using specified parameters to place out 1 by 1/sqrt(2) cutter.

    Parameters:
    params (tuple): The parameters where and how to place the rectangle cutter.
    rose (Polygon): The rose curve as a Polygon . 

    Returns:
    int: The negative area of the rose curve where the cutter was placed
    '''
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


def simulate_rose_area(rose):
    '''
    Simulates the optimal 1 by 1/sqrt(2) retangle cut out of a rose curve to maximum the area being cut. Outputs a graph showing the optimal cut.
    '''
    print("\nSIMULATING OPTIMAL ROSE CURVE AREA CUT:")
    # Create rose graph polygon
    # rose = rose_curve().buffer(0)

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
    print(f"\n Max area: {max_area}")
    print(f"Rectangle placed a ({best_x},{best_y}) at {best_angle} radians.\n")

    # Plotting
    fig, ax = plt.subplots()
    plot_shape(ax, rose, f'Optimized Area Cut for Rose Curve and a\nRectangle Cutter. Max Area: {max_area:.4f}', 'Rose Curve')

    cutter_x, cutter_y = best_cutter.exterior.xy
    ax.plot(cutter_x, cutter_y, color='blue', label='Best Cutter')

    ax.set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def f(x: float, y: float):
    '''
    Our differential equation for the trajectory of the plane.

    Returns: 
    float: the value of the equation with a given x and y values
    '''
    return (y / x) - (k * math.sqrt(1 + (y / x) ** 2))

def simulate_plane(start_x, start_y, h):
    '''
    Simulates the airplane trajectory using a Runge-Kutta Method of numerically approximate the solution to a differential equaiton. A graph with the trajectory is displayed.
    '''
    print("SIMULATING AIRPLANE TRAJECTORY")
    x = start_x
    y = start_y
    x_trajectory = []
    y_trajectory = []
    while (x >= 0 and y >= 0):
        # Runge-Kutta Method
        k1 = f(x, y)
        k2 = f(x + (h/2), y + ((h/2) * k1)) 
        k3 = f(x + h, y - (h*k1) + (2*h*k2)) 

        y += (h/6) * (k1 + (4*k2) + k3)
        x += h

        if (x >= 0 and y >= 0):
            x_trajectory.append(x)
            y_trajectory.append(y)
            if VERBOSE: print(f"({x}, {y})")
            else: print(f"Progress: {(int)((start_x - x))}%\r", end="")


    print("Progress: 100%")

    plt.plot(x_trajectory, y_trajectory, label='Trajectory')
    plt.title('Airplane Trajectory using a Runge-Kutta Method')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    trajectory = []
    for i in range(0, len(x_trajectory)):
        trajectory.append([x_trajectory[i], y_trajectory[i]])
    return trajectory




def main():
    global VERBOSE, GRAPHS
    tests = [False, False, False]
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
                elif test == '-p':  tests[2] = True
    ### END SWITCHBOARD ###
    perform = ((True not in tests) or (False not in tests))

    # Simulate Buffon's needles Data
    if (perform or tests[0]):
        simulation_buffon_disk()

    # Simulate Rose Curve needles
    if (perform or tests[1]):
        rose_poly = rose_curve().buffer(0)
        simulate_rose_area(rose_poly)

    if (perform or tests[2]):
        start_x = 100
        start_y = 0
        h = -1*math.pow(10, -4)
        simulate_plane(start_x, start_y, h)


if __name__ == "__main__":
    main()