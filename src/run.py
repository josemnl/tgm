import numpy as np
import matplotlib.pyplot as plt
import time
import yaml

from utilities import readLidarData, readLidarData3D, readPoseData, createVideo
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching

def run():
    # Config file
    configFile = './config/baseline.yaml'
    # Import parameters from config file
    parameters = yaml.safe_load(open(configFile))

    # Load parameters
    # Log parameters
    logID = parameters['logID']
    is3D = parameters['is3D']
    initialTimeStep = parameters['initialTimeStep']
    simHorizon = parameters['simHorizon']
    # SLAM parameters
    isSLAM = parameters['isSLAM']
    numTimeStepsSLAM = parameters['numTimeStepsSLAM']
    startPoseSLAM = parameters['startPoseSLAM']
    # Plotting parameters
    saveVideo = parameters['saveVideo']
    followingVideo = parameters['followingVideo']
    followingWidth = parameters['followingWidth']
    followingWeight = parameters['followingWeight']
    # Grid parameters
    origin = parameters['origin']
    width = parameters['width']
    height = parameters['height']
    resolution = parameters['resolution']
    # TGM parameters
    staticPrior = parameters['staticPrior']
    dynamicPrior = parameters['dynamicPrior']
    weatherPrior = parameters['weatherPrior']
    maxVelocity = parameters['maxVelocity']
    saturationLimits = parameters['saturationLimits']
    fftConv = parameters['fftConv']
    # Filter parameters
    groundThreshold = parameters['groundThreshold']
    skyThreshold = parameters['skyThreshold']
    minDistance = parameters['minDistance']
    maxDistance = parameters['maxDistance']
    voxelGridSize = 1/resolution
    angRes = parameters['angRes']
    # Sensor Model parameters
    smWidth = parameters['smWidth']
    smHeight = parameters['smHeight']
    sensorRange = parameters['sensorRange']
    invModel = parameters['invModel']
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
            # Option with new sensor model
            #z_t = z_t_3D.removeGround(groundThreshold).removeSky(skyThreshold).removeClosePoints(minDistance).convertTo2D_new(angRes, maxDistance).removeFarPoints(maxDistance).voxelGridFilter(voxelGridSize).orderByAngle()
            # Option with old sensor model
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