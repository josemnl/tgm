import numpy as np
import matplotlib.pyplot as plt
import time

from utilities import readLidarData, readLidarData3D, readPoseData, createVideo, loadConfig
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching, plotCostFunction

def run():
    # Config file
    configPath = './config/'
    configFile = 'Exp1-2024-03-01-15-10-32-TGM'

    # Load parameters
    (
        logID, is3D, initialTimeStep, simHorizon, isSLAM, velTracking, numTimeStepsSLAM, 
        startPoseSLAM, saveVideo, removeFrames, saveSvg, videoSection, videoWidth, videoHeight,
        videoOrigin, style, origin, width, height, resolution, staticPrior, 
        dynamicPrior, weatherPrior, maxVelocity, saturationLimits, fftConv, 
        groundThreshold, skyThreshold, minDistance, maxDistance, voxelGridSize, 
        angRes, smWidth, smHeight, sensorRange, invModel, occPrior, freeUpGroundDetections
    ) = loadConfig(configPath, configFile)

    # Paths
    logPath = './logs/' + logID + '/'
    videoPath = './results/' + configFile + '/'

    # Create results folder if it does not exist
    import os
    if not os.path.exists(videoPath):
        os.makedirs(videoPath)

    # Create Sensor Model and TGM
    sM = sensorModel(origin, smWidth, smHeight, resolution, sensorRange, invModel, occPrior)
    tgm = TGM(origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits, fftConv)

    # Empty arrays for the results
    x_t_SLAM_array = []
    n_occ_cells_array = []

    # Initial guess for the velocity
    v_t = [0, 0, 0]

    # Main loop
    fig= plt.figure()
    for i in range(initialTimeStep, initialTimeStep + simHorizon):
        timeStart = time.time()

        # Import sensor data
        if is3D:
            z_t_3D = readLidarData3D(logPath, i)
            # Option with new sensor model
            #z_t = z_t_3D.removeGround(groundThreshold).removeSky(skyThreshold).removeClosePoints(minDistance).convertTo2D_new(angRes, maxDistance).removeFarPoints(maxDistance).voxelGridFilter(voxelGridSize).orderByAngle()
            # Option with old sensor model
            if freeUpGroundDetections:
                z_t_ground_3D, z_t_objects_3D = z_t_3D.removeSky(skyThreshold).splitByHeight(groundThreshold)
                z_t_ground = z_t_ground_3D.convertTo2D().removeFarPoints(maxDistance).voxelGridFilter(voxelGridSize)
                z_t = z_t_objects_3D.convertTo2D().removeClosePoints(minDistance).removeFarPoints(maxDistance).voxelGridFilter(voxelGridSize).orderByAngle()
            else:
                z_t = z_t_3D.removeGround(groundThreshold).removeSky(skyThreshold).convertTo2D().removeClosePoints(minDistance).removeFarPoints(maxDistance).voxelGridFilter(voxelGridSize).orderByAngle()
        else:
            z_t = readLidarData(logPath, i)
        timeData = time.time()

        # Compute robot pose with SLAM or get it from log
        if not isSLAM:
            x_t = readPoseData(logPath, i)
        elif i <= initialTimeStep + numTimeStepsSLAM:
            try:
                x_t = readPoseData(logPath, i)
            except:
                x_t = np.array(startPoseSLAM)
        else:
            x_prev = x_t
            if velTracking:
                initialGuess = x_t + v_t
            else:
                initialGuess = x_t
            time1 = time.time()
            slam_map = tgm.computeStaticGridMap(following=True, width=smWidth, height=smHeight)
            print('Time to compute static grid map: ' + str(time.time() - time1))
            print("")
            x_t = lsqnl_matching(z_t, slam_map, initialGuess, sensorRange)
            v_t = x_t - x_prev
        timeSLAM = time.time()

        # Plot cost function
        #if i == 303:
        #    plotCostFunction(z_t, tgm.computeStaticGridMap(), x_t, sensorRange, res=10)

        # Save SLAM results
        if isSLAM:
            x_t_SLAM_array.append(x_t)

        # Compute instantaneous grid map with inverse sensor model
        sM.updateBasedOnPose(x_t)
        if freeUpGroundDetections:
            gm = sM.generateGridMap(z_t, x_t, z_t_ground)
        else:
            gm = sM.generateGridMap(z_t, x_t)
        timeSensorModel = time.time()

        # Update TGM
        tgm.update(gm, x_t)
        timeTGM = time.time()

        # Plot maps
        fig.clear()
        tgm.plot(fig, saveImg=saveVideo, saveSvg=saveSvg, imgName= videoPath + 'frame_' + str(i-initialTimeStep+1), section = videoSection, width=videoWidth, height=videoHeight, origin=videoOrigin, style=style)
        timePlot = time.time()

        # Print progress
        print('Frame:   ' + str(i-initialTimeStep+1) + ' / ' + str(simHorizon))

        # Print times
        print('Data:    ' + str(timeData - timeStart))
        print('SLAM:    ' + str(timeSLAM - timeData))
        print('InvSenM: ' + str(timeSensorModel - timeSLAM))
        print('TGM:     ' + str(timeTGM - timeSensorModel))
        print('Plots:   ' + str(timePlot - timeTGM))
        print('Total:   ' + str(time.time() - timeStart))
        print('')

    # Save SLAM results
    if isSLAM:
        np.savetxt(videoPath + 'x_t_SLAM.csv', x_t_SLAM_array, delimiter=',')

    # Save video
    if saveVideo:
        createVideo(logID, videoPath, removeFrames = removeFrames)

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