import numpy as np
import matplotlib.pyplot as plt
import time

from utilities import read2DLidarCSV, read3DLidarCSV, read3DLidarBIN, read3DLabledLidarBIN, readPose, createVideo, loadConfigAsDict
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching
from metrics import computeMetrics

def run():
    # Config file
    configPath = './config/'
    defConfFile = 'config'
    logID = 'SnowyKitti-00'

    # Load parameters
    conf = loadConfigAsDict(configPath, defConfFile)
    specificConf = loadConfigAsDict(configPath, logID)
    conf.__dict__.update(specificConf.__dict__)

    # Paths
    videoPath = './results/' + logID + '/'

    # Create results folder if it does not exist
    import os
    if not os.path.exists(videoPath):
        os.makedirs(videoPath)

    # Create Sensor Model and TGM
    sM = sensorModel(conf.origin, conf.smWidth, conf.smHeight, conf.resolution, conf.sensorRange, conf.invModel, conf.occPrior)
    tgm = TGM(conf.origin, conf.width, conf.height, conf.resolution, conf.staticPrior, conf.dynamicPrior, conf.weatherPrior, conf.maxVelocity, conf.saturationLimits, conf.fftConv)

    # Empty arrays for the results
    x_t_SLAM_array = []
    n_occ_cells_array = []
    n_snow_occ_cells_original = []
    n_snow_occ_cells_baseline = []
    n_snow_occ_cells_our_method = []


    # Initial guess for the velocity
    v_t = [0, 0, 0]

    # Main loop
    fig= plt.figure()
    for i in range(conf.initialTimeStep, conf.initialTimeStep + conf.simHorizon):
        timeStart = time.time()

        # Import sensor data
        if conf.is3D:
            if conf.lidarFormat == 'CSV':
                z_t_3D = read3DLidarCSV(conf.lidarPath, i)
            elif conf.lidarFormat == 'BIN':
                if conf.isLabeled:
                    z_t_3D = read3DLabledLidarBIN(conf.lidarPath, conf.labelPath, i)
                else:
                    z_t_3D = read3DLidarBIN(conf.lidarPath, i)
            else:
                raise ValueError('Invalid lidar format')
            if conf.freeUpGroundDetections:
                z_t_3D.removeSky(conf.skyThreshold)
                z_t_ground_3D, z_t_objects_3D = z_t_3D.splitByHeight(conf.groundThreshold)
                z_t_before_filter = z_t_objects_3D.convertTo2D()
                z_t_before_filter.removeClosePoints(conf.minDistance)
                z_t_before_filter.removeFarPoints(conf.maxDistance)
                #z_t_objects_3D.ROR(5, 0.2)
                #z_t_objects_3D.SOR(5, 3)
                #z_t_objects_3D.DROR(5, 0.01)
                z_t_objects_3D.DSOR(3, 2, 0.08) # Use this one for SnowyKITTI
                # z_t_objects_3D.DSOR(5, 2, 0.01) # Use this one for WADS
                z_t_ground = z_t_ground_3D.convertTo2D()
                z_t_ground.removeFarPoints(conf.maxDistance)
                #z_t_ground.voxelGridFilter(voxelGridSize) # No filtering for ground points since it's more expensive than dealing with them on the sensor model
                z_t = z_t_objects_3D.convertTo2D()
                z_t.removeClosePoints(conf.minDistance)
                z_t.removeFarPoints(conf.maxDistance)
                #z_t.voxelGridFilter(conf.voxelGridSize)
                z_t.orderByAngle()
            else:
                z_t = z_t_3D.removeGround(conf.groundThreshold).removeSky(conf.skyThreshold).convertTo2D().removeClosePoints(conf.minDistance).removeFarPoints(conf.maxDistance).voxelGridFilter(conf.voxelGridSize).orderByAngle()
        else:
            z_t = read2DLidarCSV(conf.lidarPath, i)
        timeData = time.time()

        # Compute robot pose with SLAM or get it from log
        if not conf.isSLAM:
            x_t = readPose(conf.lidarPath, i)
        elif i <= conf.initialTimeStep + conf.numTimeStepsSLAM:
            try:
                x_t = readPose(conf.lidarPath, i)
            except:
                x_t = np.array(conf.startPoseSLAM)
        else:
            x_prev = x_t
            if conf.velTracking:
                initialGuess = x_t + v_t
            else:
                initialGuess = x_t
            slam_map = tgm.oneLayer('static', following=True, width=conf.smWidth, height=conf.smHeight)
            x_t = lsqnl_matching(z_t, slam_map, initialGuess, conf.sensorRange)
            v_t = x_t - x_prev
        timeSLAM = time.time()

        # Save SLAM results
        if conf.isSLAM:
            x_t_SLAM_array.append(x_t)

        # Compute instantaneous grid map with inverse sensor model
        sM.updateBasedOnPose(x_t)
        if conf.freeUpGroundDetections:
            gm = sM.generateGridMap(z_t, x_t, z_t_ground)
        else:
            gm = sM.generateGridMap(z_t, x_t)
        timeSensorModel = time.time()

        # Update TGM
        tgm.update(gm, x_t)
        timeTGM = time.time()

        # Plot maps
        fig.clear()
        tgm.plot(fig, saveImg=conf.saveVideo, saveSvg=conf.saveSvg, imgName= videoPath + 'frame_' + str(i-conf.initialTimeStep+1), section = conf.videoSection, width=conf.videoWidth, height=conf.videoHeight, origin=conf.videoOrigin, style=conf.style)
        timePlot = time.time()

        # Print progress
        print('Frame:   ' + str(i-conf.initialTimeStep+1) + ' / ' + str(conf.simHorizon))

        # Print times
        print('Data:    ' + str(timeData - timeStart))
        print('SLAM:    ' + str(timeSLAM - timeData))
        print('InvSenM: ' + str(timeSensorModel - timeSLAM))
        print('TGM:     ' + str(timeTGM - timeSensorModel))
        print('Plots:   ' + str(timePlot - timeTGM))
        print('Total:   ' + str(time.time() - timeStart))
        print('')

        # Snow metrics
        if conf.isLabeled:
            # Before the filter
            gm_before_filter = sM.generateGridMap(z_t_before_filter, x_t)
            n_occ_cells, n_snow_points = computeMetrics(z_t_before_filter, x_t, gm_before_filter, conf.snowLabel)
            n_snow_occ_cells_original.append(n_occ_cells)
            print('Occupied cells: ' + str(n_occ_cells) + ' / ' + str(n_snow_points))
            # Baseline: Instantaneous occupancy map
            n_occ_cells, n_snow_points = computeMetrics(z_t, x_t, gm, conf.snowLabel)
            n_snow_occ_cells_baseline.append(n_occ_cells)
            print('Occupied cells: ' + str(n_occ_cells) + ' / ' + str(n_snow_points))
            # Our method: Occupancy map from TGM
            tgm_gm = tgm.computeStaticDynamicGridMap()
            n_occ_cells, n_snow_points = computeMetrics(z_t, x_t, tgm_gm, conf.snowLabel)
            n_snow_occ_cells_our_method.append(n_occ_cells)
            print('Occupied cells: ' + str(n_occ_cells) + ' / ' + str(n_snow_points))

    # Save SLAM results
    if conf.isSLAM:
        np.savetxt(videoPath + 'x_t_SLAM.csv', x_t_SLAM_array, delimiter=',')

    # Save snow metrics
    if conf.isLabeled:
        np.savetxt(videoPath + 'n_snow_occ_cells_original.csv', n_snow_occ_cells_original, delimiter=',')
        np.savetxt(videoPath + 'n_snow_occ_cells_baseline.csv', n_snow_occ_cells_baseline, delimiter=',')
        np.savetxt(videoPath + 'n_snow_occ_cells_our_method.csv', n_snow_occ_cells_our_method, delimiter=',')

    # Save video
    if conf.saveVideo:
        createVideo(logID, videoPath, removeFrames = conf.removeFrames)

    # Save last frame
    tgm.plot(fig, saveImg=True, imgName= videoPath + logID)

    # Save static grid map
    tgm.plot(fig, saveImg=True, imgName= videoPath + logID + '_static', style='static')

    # Save dynamic grid map
    tgm.plot(fig, saveImg=True, imgName= videoPath + logID + '_dynamic', style='dynamic')

    # Save weather grid map
    tgm.plot(fig, saveImg=True, imgName= videoPath + logID + '_weather', style='weather')

if __name__ == '__main__':
    run()