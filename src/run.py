import numpy as np
import matplotlib.pyplot as plt
import os

from utilities import read2DLidarCSV, read3DLidarCSV, read3DLidarBIN, read3DLabledLidarBIN, readPose, createVideo, loadConfigAsDict
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching
from spatial import frame, origin, size, position, pose, orientation

def run(logID, conf):
    # Print logID
    print('Running ' + logID)
    # Paths
    resultsPath = './results/' + logID + '/'

    # Create results folder if it does not exist
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    # Create Sensor Model and TGM
    sMsize = size(conf.smWidth, conf.smHeight, 1)
    smOrigin = origin(conf.origin[0], conf.origin[1], 0)
    smFrame = frame(smOrigin, sMsize, conf.resolution)
    sM = sensorModel(smFrame, conf.sensorRange, conf.invModel, conf.occPrior)
    tgmOrigin = origin(conf.origin[0], conf.origin[1], 0)
    tgmSize = size(conf.width, conf.height, 1)
    tgmFrame = frame(tgmOrigin, tgmSize, conf.resolution)
    tgm = TGM(tgmFrame, (conf.staticPrior, conf.dynamicPrior, conf.weatherPrior), conf.maxVelocity, conf.saturationLimits, conf.fftConv, conf.isGPU)

    # Initial guess for the velocity (pose type to support pose arithmetic)
    v_t = pose(position(0.0, 0.0, 0.0), orientation(0.0, 0.0, 0.0))

    # Initialize figure for plotting
    fig= plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Main loop
    for i in range(conf.initialTimeStep, conf.initialTimeStep + conf.simHorizon):

        # Import sensor data
        if conf.is3D:
            if conf.lidarFormat == 'CSV':
                z_t_3D = read3DLidarCSV(conf.lidarPath + "z_" + str(i) + ".csv")
            elif conf.lidarFormat == 'BIN':
                if conf.isLabeled:
                    z_t_3D = read3DLabledLidarBIN(conf.lidarPath + str(i).zfill(6) + ".bin", conf.labelPath + str(i).zfill(6) + ".label")
                else:
                    z_t_3D = read3DLidarBIN(conf.lidarPath + str(i).zfill(6) + ".bin")
            else:
                raise ValueError('Invalid lidar format')
        else:
            z_t = read2DLidarCSV(conf.lidarPath + "z_" + str(i) + ".csv")

        # Filter the point cloud
        if conf.is3D:
            # Remove close, far and sky points
            z_t_3D.removeClosePoints(conf.minDistance)
            z_t_3D.removeFarPoints(conf.maxDistance)
            z_t_3D.removeSky(conf.skyThreshold)

            # Split ground and objects
            if conf.freeUpGroundDetections:
                z_t_ground_3D, z_t_objects_3D = z_t_3D.splitByHeight(conf.groundThreshold)
                z_t_ground = z_t_ground_3D.convertTo2D()
            else:
                z_t_3D.removeGround(conf.groundThreshold)
                z_t_objects_3D = z_t_3D
                z_t_ground = None
            
            # Convert to 2D
            z_t = z_t_objects_3D.convertTo2D()

            # Voxel grid filter
            if conf.isVoxelGridFilter:
                z_t.voxelGridFilter(conf.voxelGridSize)
                z_t_ground.voxelGridFilter(conf.voxelGridSize)

            # Order by angle
            z_t.orderByAngle()

        # Compute robot pose with SLAM or get it from log
        if not conf.isSLAM:
            x_t = readPose(conf.lidarPath + "../snow_pose/" + "x_" + str(i).zfill(6) + ".csv")
        elif i <= conf.initialTimeStep + conf.numTimeStepsSLAM:
            try:
                x_t = readPose(conf.lidarPath + "x_" + str(i) + ".csv")
            except:
                x_t = np.array(conf.startPoseSLAM, dtype=float)
                x_t = pose(position(x_t[0], x_t[1], 0.0), orientation(0.0, 0.0, x_t[2]))
        else:
            x_prev = x_t
            if conf.velTracking:
                initialGuess = x_t + v_t
            else:
                initialGuess = x_t
            slamSize = size(conf.smWidth, conf.smHeight, 1)
            slamFrame = frame.frameAroundPosition(x_t.position, slamSize, tgm.frame.r)
            slam_map = tgm.oneLayer('static', slamFrame).toCPU()
            x_t = lsqnl_matching(z_t, slam_map, initialGuess, conf.sensorRange)
            v_t = x_t - x_prev

        # Compute instantaneous grid map with inverse sensor model
        sM.updateBasedOnPose(x_t)
        if conf.freeUpGroundDetections:
            gm = sM.generateGridMap(z_t, x_t, z_t_ground)
        else:
            gm = sM.generateGridMap(z_t, x_t)

        # If gm is partially outside the TGM, resize the TGM
        if not tgm.contains(gm.frame):
            newSize = size(tgm.frame.size.w, tgm.frame.size.h, tgm.frame.size.d)
            newFrame = frame.frameAroundPosition(x_t.position, newSize, tgm.frame.r)
            tgm.reshape(newFrame)

        # Update TGM
        tgm.update(gm, x_t)

        # Compute the frame for the plot
        if conf.videoSection == 'Full':
            plotFrame = tgm.frame
        elif conf.videoSection == 'Following':
            plotSize = size(int(conf.videoWidth / tgm.frame.r), int(conf.videoHeight / tgm.frame.r), int(1))
            plotFrame = frame.frameAroundPosition(x_t.position, plotSize, tgm.frame.r)
        elif conf.videoSection == 'Constant':
            plotOrigin = origin(int(conf.videoOrigin[0] / tgm.frame.r), int(conf.videoOrigin[1] / tgm.frame.r), 0)
            plotSize = size(int(conf.videoWidth / tgm.frame.r), int(conf.videoHeight / tgm.frame.r), 1)
            plotFrame = frame(plotOrigin, plotSize, tgm.frame.r)
        tgm.plot(fig, ax, plotFrame, saveMap=conf.saveMap, savePNG=conf.saveVideo, saveSvg=conf.saveSvg, imgName= resultsPath + 'frame_' + str(i-conf.initialTimeStep+1), style=conf.style)

        # Print progress
        print('Frame:   ' + str(i-conf.initialTimeStep+1) + ' / ' + str(conf.simHorizon))

    # Save video
    if conf.saveVideo:
        createVideo(logID, resultsPath, removeFrames = conf.removeFrames)

    # Save last frame
    fig.clear()
    tgm.plot(fig, ax, saveMap=True, imgName= resultsPath + logID)

    # Save static grid map
    fig.clear()
    tgm.plot(fig, ax, saveMap=True, imgName= resultsPath + logID + '_static', style='static')
    # Save dynamic grid map
    fig.clear()
    tgm.plot(fig, ax, saveMap=True, imgName= resultsPath + logID + '_dynamic', style='dynamic')

if __name__ == '__main__':
    # Config file
    configPath = './config/'
    defConfFile = 'config'
    logID = 'Exp2-TGM-GPU'

    # Load parameters
    conf = loadConfigAsDict(configPath, defConfFile)
    specificConf = loadConfigAsDict(configPath, logID)
    conf.__dict__.update(specificConf.__dict__)

    run(logID, conf)