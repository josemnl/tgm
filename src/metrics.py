from lidarScan import lidarScan, lidarScan3D
from gridMap import gridMap
import numpy as np
from utilities import read3DLabledLidarBIN, loadConfigAsDict
from sensorModel import sensorModel

def computeMetrics(z_t, x_t, gM, label=1):
    assert isinstance(z_t, lidarScan)
    assert isinstance(x_t, np.ndarray)
    assert isinstance(gM, gridMap)
    
    # Keep only the snow points
    z_t_snow = z_t.filterInByLabel(label)

    # Compute point cloud in global frame
    global_pointCloud = z_t_snow.computeRelativeCartesian(x_t)

    # For each snow point, check the probability of being occupied
    n_occ_cells = 0
    for point in global_pointCloud:
        # Get the occupancy of the cell where the point is
        occ = gM.occupancy(point[0], point[1])
        if occ > 0.1:
            n_occ_cells += 1

    return n_occ_cells, z_t_snow.ranges.size

def classificationMetrics(gM1, gM2, verbose=False):
    assert isinstance(gM1, gridMap)
    assert isinstance(gM2, gridMap)
    assert gM1.frame.w == gM2.frame.w
    assert gM1.frame.h == gM2.frame.h
    assert gM1.frame.r == gM2.frame.r
    assert gM1.frame.ox == gM2.frame.ox
    assert gM1.frame.oy == gM2.frame.oy
    assert gM1.frame.r == gM2.frame.r
    assert gM1.isBool
    assert gM2.isBool

    # Compute the intersection
    intersection = np.logical_and(gM1.data, gM2.data)
    intersection_sum = np.sum(intersection)

    # Compute the union
    union = np.logical_or(gM1.data, gM2.data)
    union_sum = np.sum(union)

    # Compute IoU
    IoU = intersection_sum / union_sum

    # Compute precision and recall (gM1 is the ground truth)
    precision = intersection_sum / np.sum(gM2.data)
    recall = intersection_sum / np.sum(gM1.data)

    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    if verbose:
        print('')
        print('IoU metrics:')
        print('Intersection: ' + str(intersection_sum))
        print('Union: ' + str(union_sum))
        print('Precision: ' + str(precision))
        print('Recall: ' + str(recall))

    return intersection_sum, union_sum, IoU, precision, recall, f1

