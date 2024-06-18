import numpy as np
import matplotlib.pyplot as plt
import time

from utilities import read2DLidarCSV, read3DLidarCSV, read3DLidarBIN, readPose, createVideo, loadConfig, loadConfigAsDict
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching, plotCostFunction

def run():
    # Config file
    configPath = './config/'
    configFile = 'WADS-11'

    # Load parameters as dictionary
    conf = loadConfigAsDict(configPath, configFile)

    # Paths
    #logPath = './logs/' + conf.logID + '/'
    #logPath = './SnowyKITTI/dataset/sequences/' + conf.logID + '/' + 'snow_velodyne/'
    logPath = './WADS/' + conf.logID + '/' + 'velodyne/'
    videoPath = './results/' + configFile + '/'

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

    # Initial guess for the velocity
    v_t = [0, 0, 0]

    # Main loop
    fig= plt.figure()
    for i in range(conf.initialTimeStep, conf.initialTimeStep + conf.simHorizon):
        timeStart = time.time()

        # Import sensor data
        if conf.is3D:
            z_t_3D = read3DLidarBIN(logPath, i)
            if conf.freeUpGroundDetections:
                z_t_3D.removeSky(conf.skyThreshold)
                z_t_ground_3D, z_t_objects_3D = z_t_3D.splitByHeight(conf.groundThreshold)
                #z_t_objects_3D.radiousOutlierRemoval(3, 1)
                #z_t_objects_3D.statisticalOutlierRemoval(3, 3)
                z_t_ground = z_t_ground_3D.convertTo2D()
                z_t_ground.removeFarPoints(conf.maxDistance)
                #z_t_ground.voxelGridFilter(voxelGridSize) # No filtering for ground points since it's more expensive than dealing with them on the sensor model
                z_t = z_t_objects_3D.convertTo2D()
                z_t.removeClosePoints(conf.minDistance)
                z_t.removeFarPoints(conf.maxDistance)
                z_t.voxelGridFilter(conf.voxelGridSize)
                z_t.orderByAngle()
            else:
                z_t = z_t_3D.removeGround(conf.groundThreshold).removeSky(conf.skyThreshold).convertTo2D().removeClosePoints(conf.minDistance).removeFarPoints(conf.maxDistance).voxelGridFilter(conf.voxelGridSize).orderByAngle()
        else:
            z_t = read2DLidarCSV(logPath, i)
        timeData = time.time()

        # Compute robot pose with SLAM or get it from log
        if not conf.isSLAM:
            x_t = readPose(logPath, i)
        elif i <= conf.initialTimeStep + conf.numTimeStepsSLAM:
            try:
                x_t = readPose(logPath, i)
            except:
                x_t = np.array(conf.startPoseSLAM)
        else:
            x_prev = x_t
            if conf.velTracking:
                initialGuess = x_t + v_t
            else:
                initialGuess = x_t
            time1 = time.time()
            slam_map = tgm.computeStaticGridMap(following=True, width=conf.smWidth, height=conf.smHeight)
            print('Time to compute static grid map: ' + str(time.time() - time1))
            print("")
            x_t = lsqnl_matching(z_t, slam_map, initialGuess, conf.sensorRange)
            v_t = x_t - x_prev
        timeSLAM = time.time()

        # Plot cost function
        #if i == 303:
        #    plotCostFunction(z_t, tgm.computeStaticGridMap(), x_t, sensorRange, res=10)

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

    # Save SLAM results
    if conf.isSLAM:
        np.savetxt(videoPath + 'x_t_SLAM.csv', x_t_SLAM_array, delimiter=',')

    # Save video
    if conf.saveVideo:
        createVideo(conf.logID, videoPath, removeFrames = conf.removeFrames)

    # Save last frame
    tgm.plot(fig, saveImg=True, imgName= videoPath + conf.logID)

    # Save static grid map
    tgm.plot(fig, saveImg=True, imgName= videoPath + conf.logID + '_static', style='static')

    # Save dynamic grid map
    tgm.plot(fig, saveImg=True, imgName= videoPath + conf.logID + '_dynamic', style='dynamic')

    # Save weather grid map
    tgm.plot(fig, saveImg=True, imgName= videoPath + conf.logID + '_weather', style='weather')

if __name__ == '__main__':
    run()