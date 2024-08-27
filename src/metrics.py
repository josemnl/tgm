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
    z_t.filterInByLabel(label)

    #z_t.angles = z_t.angles + x_t[2]

    # Compute point cloud in global frame
    global_pointCloud = z_t.computeRelativeCartesian(x_t)

    # For each snow point, check the probability of being occupied
    n_occ_cells = 0
    for point in global_pointCloud:
        # Get the occupancy of the cell where the point is
        occ = gM.occupancy(point[0], point[1])
        if occ > 0.7:
            n_occ_cells += 1

    return n_occ_cells, z_t.ranges.size

if __name__ == "__main__":
    # Config file
    configPath = './config/'
    logID = 'SnowyKitti-00'

    # Load parameters as dictionary
    conf = loadConfigAsDict(configPath, logID)

    sM = sensorModel(conf.origin, conf.smWidth, conf.smHeight, conf.resolution, conf.sensorRange, conf.invModel, conf.occPrior)

    pathLabels = './SnowyKITTI/dataset/sequences/00/snow_labels/'
    z_t_3D = read3DLabledLidarBIN(conf.lidarPath, pathLabels, conf.initialTimeStep)

    # Create a fake 3D lidar scan with one point
    #points3D = np.array([[10, 2, 0]])
    #labels = np.array([1])
    #z_t_3D = lidarScan3D(points3D, labels)

    z_t_3D.removeSky(conf.skyThreshold)
    z_t_ground_3D, z_t_objects_3D = z_t_3D.splitByHeight(conf.groundThreshold)
    #z_t_objects_3D.ROR(5, 0.2)
    #z_t_objects_3D.SOR(5, 3)
    #z_t_objects_3D.DROR(5, 0.01)
    #z_t_objects_3D.DSOR(5, 2, 0.1)
    z_t_ground = z_t_ground_3D.convertTo2D()
    z_t_ground.removeFarPoints(conf.maxDistance)
    #z_t_ground.voxelGridFilter(voxelGridSize) # No filtering for ground points since it's more expensive than dealing with them on the sensor model
    z_t = z_t_objects_3D.convertTo2D()
    z_t.plot()
    z_t.removeClosePoints(conf.minDistance)
    z_t.removeFarPoints(conf.maxDistance)
    #z_t.voxelGridFilter(conf.voxelGridSize)
    z_t.orderByAngle()

    x_t = np.array(conf.startPoseSLAM)

    
    

    sM.updateBasedOnPose(x_t)
    print(z_t.ranges, z_t.angles)
    if conf.freeUpGroundDetections:
        gm = sM.generateGridMap(z_t, x_t, z_t_ground)
    else:
        gm = sM.generateGridMap(z_t, x_t)

    
    print(z_t.ranges, z_t.angles)

    gm.plot()
    print(computeMetrics(z_t, x_t, gm))