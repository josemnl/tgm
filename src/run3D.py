import numpy as np
import matplotlib.pyplot as plt
import time
import os

from utilities import read2DLidarCSV, read3DLidarCSV, read3DLidarBIN, read3DLabledLidarBIN, readPose, createVideo, loadConfigAsDict, listFilesExt
from sensorModel import sensorModel3D
from TGM import TGM
from SLAM import lsqnl_matching3D
from gridMap import gridMap, frame, origin, size, position, pose, orientation

def run3D(logID, conf):
    # Print logID
    print('Running ' + logID)
    # Paths
    videoPath = './results/' + logID + '/'

    # Create results folder if it does not exist
    if not os.path.exists(videoPath):
        os.makedirs(videoPath)

    # Read initial pose
    x_0 = readPose(conf.lidarPath + str(conf.initialTimeStep).zfill(6) + ".csv")

    # Create Sensor Model and TGM
    sMsize = size(conf.smWidth, conf.smHeight, conf.smDepth)
    smOrigin = origin(conf.origin[0], conf.origin[1], conf.origin[2])
    # Create the frame around the initial position
    smFrame = frame.frameAroundPosition(x_0.position, sMsize, conf.resolution)
    sM = sensorModel3D(smFrame, conf.invModel, conf.occPrior)
    tgmOrigin = origin(conf.origin[0], conf.origin[1], conf.origin[2])
    tgmSize = size(conf.width, conf.height, conf.depth)
    tgmFrame = frame(tgmOrigin, tgmSize, conf.resolution)
    tgm = TGM(tgmFrame, (conf.staticPrior, conf.dynamicPrior, conf.weatherPrior), conf.maxVelocity, conf.saturationLimits, conf.fftConv, conf.isGPU)

    print('TGM initial frame is: origin (' + str(tgm.frame.origin.x) + ', ' + str(tgm.frame.origin.y) + ', ' + str(tgm.frame.origin.z) + ') with size (' + str(tgm.frame.size.w) + ', ' + str(tgm.frame.size.h) + ', ' + str(tgm.frame.size.d) + ') and resolution ' + str(tgm.frame.r))
    # Initial guess for the velocity (pose type to support pose arithmetic)
    v_t = pose(position(0.0, 0.0, 0.0), orientation(0.0, 0.0, 0.0))
    
    # Define transformations
    link_world_base = pose(position(0.0, 0.0, 0.0), orientation(np.pi, 0.0, 0.0))
    link_base_sensor = pose(position(0.22, 0.0, -0.15), orientation(0.0, -np.pi/6, 0.0))

    # Main loop
    fig= plt.figure()
    for i in range(conf.initialTimeStep, conf.initialTimeStep + conf.simHorizon):

        # IMPORT SENSOR DATA
        if conf.lidarFormat == 'CSV':
            z_t_3D = read3DLidarCSV(conf.lidarPath + "z_" + str(i) + ".csv")
        elif conf.lidarFormat == 'BIN':
            z_t_3D = read3DLidarBIN(conf.lidarPath + str(i).zfill(6) + ".bin")
        else:
            raise ValueError('Invalid lidar format')

        # Convert point cloud to adjust for coordinate frame:
        # In the original data, the sensor faces towards +X, +Y is right, +Z is downward
        # We want the sensor to face towards +X, +Y to be left, +Z to be upward
        #z_t_3D.points3D[:, 0] = -z_t_3D.points3D[:, 0]
        #z_t_3D.points3D[:, 1] = -z_t_3D.points3D[:, 1]
        #z_t_3D.points3D[:, 2] = -z_t_3D.points3D[:, 2]

        #z_t_3D.plot()

        # FILTER POINT CLOUD
        z_t_3D.removeClosePoints(conf.minDistance)
        z_t_3D.removeFarPoints(conf.maxDistance)
        print(conf.skyThreshold)
        if conf.skyThreshold is not None:
            z_t_3D.removeSky(conf.skyThreshold)
        if conf.groundThreshold is not None:
            z_t_3D.removeGround(conf.groundThreshold)
        if conf.isVoxelGridFilter:
            z_t_3D.voxelGridFilter(conf.voxelGridSize)

        # Compute robot pose with SLAM or get it from log
        if not conf.isSLAM:
            x_t = readPose(conf.lidarPath + str(i).zfill(6) + ".csv")
            # Apply transformation from world to base link and from base link to sensor
            x_t = link_world_base @ x_t @ link_base_sensor
        elif i <= conf.initialTimeStep + conf.numTimeStepsSLAM:
            try:
                x_t = readPose(conf.lidarPath + "x_" + str(i) + ".csv")
            except:
                x_t = pose(position(conf.startPoseSLAM[0], conf.startPoseSLAM[1], conf.startPoseSLAM[2]),
                           orientation(0.0, 0.0, 0.0))
        else:
            x_prev = x_t
            if conf.velTracking:
                initialGuess = x_t + v_t
            else:
                initialGuess = x_t
            slamSize = size(conf.smWidth, conf.smHeight, conf.smDepth)
            slamFrame = frame.frameAroundPosition(x_t.position, slamSize, tgm.frame.r)
            slam_map = tgm.oneLayer('static', slamFrame).toCPU()
            x_t = lsqnl_matching3D(z_t_3D, slam_map, initialGuess, conf.sensorRange)
            v_t = x_t - x_prev
        
        print('Current pose at time step ' + str(i) + ': (' + str(x_t.position.x) + ', ' + str(x_t.position.y) + ', ' + str(x_t.position.z) + '), with orientation (' + str(x_t.orientation.roll) + ', ' + str(x_t.orientation.pitch) + ', ' + str(x_t.orientation.yaw) + ')')

        # Compute instantaneous grid map with inverse sensor model
        sM.updateBasedOnPose(x_t)
        gm = sM.generateGridMap(z_t_3D, x_t)

        # If gm is partially outside the TGM, resize the TGM
        if not tgm.contains(gm.frame):
            newSize = size(tgm.frame.size.w, tgm.frame.size.h, tgm.frame.size.d)
            newFrame = frame.frameAroundPosition(x_t.position, newSize, tgm.frame.r)
            print('Resizing TGM to contain the new grid map')
            print('Old TGM frame: origin (' + str(tgm.frame.origin.x) + ', ' + str(tgm.frame.origin.y) + ', ' + str(tgm.frame.origin.z) + ') with size (' + str(tgm.frame.size.w) + ', ' + str(tgm.frame.size.h) + ', ' + str(tgm.frame.size.d) + ') and resolution ' + str(tgm.frame.r))
            print('New TGM frame: origin (' + str(newFrame.origin.x) + ', ' + str(newFrame.origin.y) + ', ' + str(newFrame.origin.z) + ') with size (' + str(newFrame.size.w) + ', ' + str(newFrame.size.h) + ', ' + str(newFrame.size.d) + ') and resolution ' + str(newFrame.r))
            tgm.reshape(newFrame)

        # Update TGM
        tgm.update(gm, x_t)

        # Plot maps
        fig.clear()
        # Compute the frame for the plot
        if conf.videoSection == 'Full':
            plotFrame = tgm.frame
        elif conf.videoSection == 'Following':
            print('Following frame around position (' + str(x_t.position.x) + ', ' + str(x_t.position.y) + ', ' + str(x_t.position.z) + ')')
            plotSize = size(int(conf.videoWidth / tgm.frame.r), int(conf.videoHeight / tgm.frame.r), int(conf.videoDepth / tgm.frame.r))
            plotFrame = frame.frameAroundPosition(x_t.position, plotSize, tgm.frame.r)
        elif conf.videoSection == 'Constant':
            plotOrigin = origin(int(conf.videoOrigin[0] / tgm.frame.r), int(conf.videoOrigin[1] / tgm.frame.r), int(conf.videoOrigin[2] / tgm.frame.r))
            plotSize = size(int(conf.videoWidth / tgm.frame.r), int(conf.videoHeight / tgm.frame.r), int(conf.videoDepth / tgm.frame.r))
            plotFrame = frame(plotOrigin, plotSize, tgm.frame.r)
        print('Plotting frame at origin (' + str(plotFrame.origin.x) + ', ' + str(plotFrame.origin.y) + ', ' + str(plotFrame.origin.z) + ') with size (' + str(plotFrame.size.w) + ', ' + str(plotFrame.size.h) + ', ' + str(plotFrame.size.d) + ') and resolution ' + str(plotFrame.r))
        tgm.plot3D(fig, plotFrame, isPause=False, value_min = 0.7, value_max = 1.0)

        # Pause indefinitely for every i multiple of 50
        if (i - conf.initialTimeStep) % 100 == 0 and i != conf.initialTimeStep:
            plt.pause(0.1)
            input("Press Enter to continue...")

        # Print progress
        print('Frame:   ' + str(i-conf.initialTimeStep+1) + ' / ' + str(conf.simHorizon))

if __name__ == '__main__':
    # Config file
    configPath = './config/'
    defConfFile = 'config'
    logID = 'underwater3D'

    # Load parameters
    conf = loadConfigAsDict(configPath, defConfFile)
    specificConf = loadConfigAsDict(configPath, logID)
    conf.__dict__.update(specificConf.__dict__)

    run3D(logID, conf)