import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from tgm.lidarScan import lidarScan

def lsqnl_matching(scan, lsq_map, x0, max_range):
    # Remove the no-return scans from scan
    lsq_scan = scan.removeNoReturn(max_range)

    x = least_squares(lsq_fun, x0, max_nfev=500, args=(lsq_scan, lsq_map), method='lm')
    return x

def lsq_fun(relPose, lsq_scan, lsq_map):
    limit_x = lsq_map.width
    limit_y = lsq_map.height
    origin_x = lsq_map.origin[0]
    origin_y = lsq_map.origin[1]
    cell_length = 1 / lsq_map.resolution

    #x = np.linspace(cell_length / 2, limit_x - cell_length / 2, lsq_map.data.shape[0])
    #y = np.linspace(cell_length / 2, limit_y - cell_length / 2, lsq_map.data.shape[1])
    
    #x = np.linspace(0, limit_x, lsq_map.data.shape[0])
    #y = np.linspace(0, limit_y, lsq_map.data.shape[1])

    x = np.linspace(origin_x, origin_x + limit_x - cell_length, lsq_map.data.shape[0])
    y = np.linspace(origin_y, origin_y + limit_y - cell_length, lsq_map.data.shape[1])

    transCart = lsq_scan.computeRelativeCartesian(relPose)

    # Option 1: Use RegularGridInterpolator
    interp = RegularGridInterpolator((x, y), lsq_map.data, bounds_error=False, method='linear', fill_value=0)
    cost = 1 - interp(transCart)

    # Option 2: Use RectBivariateSpline
    #interp = RectBivariateSpline(x, y, lsq_map.data, kx=1, ky=1)
    #cost = 1 - interp.ev(transCart[:, 0],transCart[:, 1])

    # Plot the cost:
    '''
    X, Y = np.meshgrid(x, y)
    Xq, Yq = np.meshgrid(np.arange(cell_length / 2, limit_x - cell_length / 2, cell_length / 2),
                          np.arange(cell_length / 2, limit_y - cell_length / 2, cell_length / 2))
    Vq = 1 - interp.ev(Xq, Yq)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xq, Yq, Vq, edgecolor='none')
    ax.plot(transCart[:, 0], transCart[:, 1], cost, 'r.')
    plt.show()
    '''

    return cost