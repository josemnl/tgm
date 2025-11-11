import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from lidarScan import lidarScan
from gridMap import gridMap

def lsqnl_matching(scan, lsq_map: gridMap, x0, max_range):
    # Remove the no-return scans from scan
    scan.removeFarPoints(max_range)

    # Perform the least squares optimization
    x = least_squares(lsq_fun, x0, max_nfev=500, args=(scan, lsq_map), method='lm')
    x = x.x
    return x

def lsq_fun(relPose, lsq_scan, lsq_map: gridMap):
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
    transCart = lsq_scan.computeRelativeCartesian(relPose)

    # Compute the cost function using RegularGridInterpolator
    interp = RegularGridInterpolator((x, y), lsq_map.data, bounds_error=False, method='linear', fill_value=0)
    cost = 1 - interp(transCart)
    return cost

