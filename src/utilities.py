import numpy as np
import subprocess
import os
from lidarScan import lidarScan, lidarScan3D
import yaml

def readLidarData(path, i):
    with open(path + "z_" + str(i) + ".csv") as data:
        z_t = lidarScan(*np.array([line.split(",") for line in data]).astype(float).T)
    return z_t

def readLidarData3D(path, i):
    with open(path + "z_" + str(i) + ".csv") as data:
        z_t_3D = lidarScan3D(np.array([line.split(",") for line in data]).astype(float))
    return z_t_3D

def readPoseData(path, i):
    with open(path + "x_" + str(i) + ".csv") as data:
        x_t = np.array([line.split(",") for line in data]).astype(float)[0]
    return x_t

def createVideo(logID, videoPath, removeFrames = True):
    subprocess.call(['ffmpeg', '-framerate', '8', '-i', videoPath + 'frame_%d.png', '-r', '10', '-pix_fmt', 'yuv420p',videoPath + logID + '.mp4'])
    if removeFrames:
        for file in os.listdir(videoPath):
            if file.endswith('.png'):
                os.remove(videoPath + file)

def loadConfig(configPath, configFile):
    # Import parameters from config file
    parameters = yaml.safe_load(open(configPath + configFile + '.yaml'))

    # Log parameters
    logID = parameters['logID']
    is3D = parameters['is3D']
    initialTimeStep = parameters['initialTimeStep']
    simHorizon = parameters['simHorizon']

    # If simHorizon is not defined, check all the z_t files in the log folder and set simHorizon to the number of files found
    if simHorizon == 0:
        logPath = './logs/' + logID + '/'
        simHorizon = len([name for name in os.listdir(logPath) if os.path.isfile(os.path.join(logPath, name)) and 'z_' in name])
        print('simHorizon set to ' + str(simHorizon))
    
    # SLAM parameters
    isSLAM = parameters['isSLAM']
    numTimeStepsSLAM = parameters['numTimeStepsSLAM']
    startPoseSLAM = parameters['startPoseSLAM']

    # Plotting parameters
    saveVideo = parameters['saveVideo']
    removeFrames = parameters['removeFrames']
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

    return logID, is3D, initialTimeStep, simHorizon, isSLAM, numTimeStepsSLAM, startPoseSLAM, saveVideo, removeFrames, followingVideo, followingWidth, followingWeight, origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits, fftConv, groundThreshold, skyThreshold, minDistance, maxDistance, voxelGridSize, angRes, smWidth, smHeight, sensorRange, invModel, occPrior