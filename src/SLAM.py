import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from lidarScan import lidarScan
from gridMap import gridMap, pose, position, orientation

def lsqnl_matching(scan, lsq_map: gridMap, x0: pose, max_range):
    # Remove the no-return scans from scan
    scan.removeFarPoints(max_range)

    x_0 = np.array([x0.position.x, x0.position.y, x0.orientation.yaw])

    # Perform the least squares optimization
    x = least_squares(lsq_fun, x_0, max_nfev=500, args=(scan, lsq_map), method='lm')
    x = x.x
    x = pose(position(x[0], x[1], 0.0), orientation(0.0, 0.0, x[2]))
    return x

def lsq_fun(relPose, lsq_scan: lidarScan, lsq_map: gridMap):
    # Extract grid parameters
    limit_x = lsq_map.frame.w*lsq_map.frame.r
    limit_y = lsq_map.frame.h*lsq_map.frame.r
    origin_x = lsq_map.frame.ox*lsq_map.frame.r
    origin_y = lsq_map.frame.oy*lsq_map.frame.r
    cell_length = lsq_map.frame.r

    # Create the grid
    x = np.linspace(origin_x, origin_x + limit_x - cell_length, lsq_map.data.shape[0])
    y = np.linspace(origin_y, origin_y + limit_y - cell_length, lsq_map.data.shape[1])

    # Transform the scan
    relPose = pose(position(relPose[0], relPose[1], 0.0), orientation(0.0, 0.0, relPose[2]))
    transCart = lsq_scan.computeRelativeCartesian(relPose)

    # Compute the cost function using RegularGridInterpolator
    interp = RegularGridInterpolator((x, y), lsq_map.data[:, :, 0], bounds_error=False, method='linear', fill_value=0)
    cost = 1 - interp(transCart)
    return cost

def plotCostFunction(scan, lsq_map, x_t, max_range, res=2):
    # Remove the no-return scans from scan
    lsq_scan = scan.removeNoReturn(max_range)

    # Extract grid parameters
    limit_x = lsq_map.width
    limit_y = lsq_map.height
    origin_x = lsq_map.origin[0]
    origin_y = lsq_map.origin[1]
    cell_length = 1 / lsq_map.resolution

    # Create the grid
    x = np.linspace(origin_x, origin_x + limit_x - cell_length, lsq_map.data.shape[0])
    y = np.linspace(origin_y, origin_y + limit_y - cell_length, lsq_map.data.shape[1])

    # Transform the scan
    transCart = lsq_scan.computeRelativeCartesian(x_t)

    # Compute the cost function using RegularGridInterpolator
    interp = RegularGridInterpolator((x, y), lsq_map.data, bounds_error=False, method='linear', fill_value=0)
    cost = 1 - interp(transCart)

    # Meshgrid for plotting
    Xq, Yq = np.meshgrid(np.arange(origin_x + cell_length / res, origin_x + limit_x - cell_length / res, cell_length / res),
                         np.arange(origin_y + cell_length / res, origin_y + limit_y - cell_length / res, cell_length / res))
    
    # Evaluate the cost function on the grid
    Vq = 1 - interp((Xq, Yq))

    # Plot the cost function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xq, Yq, Vq, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.
    ax.plot(transCart[:, 0], transCart[:, 1], cost, 'r.')
    plt.show()