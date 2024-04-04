import numpy as np
import matplotlib.pyplot as plt
import time

from utilities import readLidarData, readLidarData3D, readPoseData, createVideo
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching

def run():
    # Our method vs baseline
    isBaseline = True

    # Log parameters
    logID = '2024-03-15-11-25-54'
    is3D = True
    initialTimeStep = 600
    simHorizon = 10000

    # SLAM parameters
    isSLAM = True
    numTimeStepsSLAM = 1
    startPoseSLAM = [500, 500, 0]
    
    # Plotting parameters
    saveVideo = False
    followingVideo = True
    followingWidth = 100
    followingWeight = 100
    
    # Grid parameters
    origin = [0,0]
    width = 1000
    height = 1000
    resolution = 2
    
    # TGM parameters
    if isBaseline:
        staticPrior = 0.5
        dynamicPrior = 0
    else:
        staticPrior = 0.3
        dynamicPrior = 0.3
    weatherPrior = 0
    maxVelocity = 1
    saturationLimits = [0, 0.99, 0.01, 1]
    fftConv = True

    # Filter parameters
    groundThreshold = -1.5
    skyThreshold = 1
    minDistance = 2
    maxDistance = 10
    voxelGridSize = 1/resolution
    
    # Sensor Model parameters
    smWidth = 100
    smHeight = 100
    sensorRange = 50
    invModel = [0.1, 0.9]
    occPrior = staticPrior + dynamicPrior + weatherPrior

    # Paths
    logPath = './logs/' + logID + '/'
    videoPath = './videos/'

    # Create Sensor Model and TGM
    sM = sensorModel(origin, smWidth, smHeight, resolution, sensorRange, invModel, occPrior)
    tgm = TGM(origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits, fftConv)

    # Main loop
    fig= plt.figure()
    for i in range(initialTimeStep, initialTimeStep + simHorizon):
        timeStart = time.time()

        # Import sensor data
        if is3D:
            z_t_3D = readLidarData3D(logPath, i)
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
            x_t = lsqnl_matching(z_t, tgm.computeStaticGridMap(), x_t, sensorRange).x
        timeSLAM = time.time()

        # Compute instantaneous grid map with inverse sensor model
        sM.updateBasedOnPose(x_t)
        gm = sM.generateGridMap(z_t, x_t)
        timeSensorModel = time.time()

        # Update TGM
        tgm.update(gm, x_t)
        timeTGM = time.time()

        # Plot maps
        fig.clear()
        tgm.plotCombinedMap(fig, saveImg=saveVideo, imgName= videoPath + 'frame_' + str(i-initialTimeStep+1), following=followingVideo, width=followingWidth, height=followingWeight)
        timePlot = time.time()

        # Print times
        print('Data:    ' + str(timeData - timeStart))
        print('SLAM:    ' + str(timeSLAM - timeData))
        print('InvSenM: ' + str(timeSensorModel - timeSLAM))
        print('TGM:     ' + str(timeTGM - timeSensorModel))
        print('Plots:   ' + str(timePlot - timeTGM))
        print('Total:   ' + str(time.time() - timeStart))
        print('')

    # Save video
    if saveVideo:
        createVideo(logID, videoPath)

    # Save last frame
    tgm.plotCombinedMap(fig, saveImg=True, imgName= videoPath + logID)

if __name__ == '__main__':
    run()