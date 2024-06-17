import numpy as np
import subprocess
import os
from lidarScan import lidarScan, lidarScan3D
import yaml
import pandas as pd

def readLidarData(path, i):
    with open(path + "z_" + str(i) + ".csv") as data:
        z_t = lidarScan(*np.array([line.split(",") for line in data]).astype(float).T)
    return z_t

def readLidarData3D(path, i):
    data = pd.read_csv(path + "z_" + str(i) + ".csv", header=None)
    z_t_3D = lidarScan3D(data.values.astype(float))
    return z_t_3D

def readPoseData(path, i):
    with open(path + "x_" + str(i) + ".csv") as data:
        x_t = np.array([line.split(",") for line in data]).astype(float)[0]
    return x_t

def readLidarData3DBin(path, i):
    rawdata = np.fromfile(path + str(i).zfill(6) + ".bin", dtype=np.float32)
    # Convert raw data to float
    rawdata = rawdata.astype(float)
    data = np.reshape(rawdata, (-1, 4))
    z_t_3D = lidarScan3D(data[:,0:3])
    return z_t_3D

def readLidarData3DBinLabels(pathData, pathLabels, i):
    rawdata = np.fromfile(pathData + str(i).zfill(6) + ".bin", dtype=np.float32)
    # Convert raw data to float
    rawdata = rawdata.astype(float)
    data = np.reshape(rawdata, (-1, 4))
    rawlabels = np.fromfile(pathLabels + str(i).zfill(6) + ".label", dtype=np.uint32)
    labels = np.reshape(rawlabels, (-1, 1))
    z_t_3D = lidarScan3D(data[:,0:3], labels)
    return z_t_3D

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
    velTracking = parameters['velTracking']
    numTimeStepsSLAM = parameters['numTimeStepsSLAM']
    startPoseSLAM = parameters['startPoseSLAM']

    # Plotting parameters
    saveVideo = parameters['saveVideo']
    removeFrames = parameters['removeFrames']
    saveSvg = parameters['saveSvg']
    videoSection = parameters['videoSection']
    videoWidth = parameters['videoWidth']
    videoHeight = parameters['videoHeight']
    videoOrigin = parameters['videoOrigin']
    style = parameters['style']

    # Grid parameters
    origin = parameters['origin']
    width = int(parameters['width']/parameters['resolution'])
    height = int(parameters['height']/parameters['resolution'])
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
    voxelGridSize = resolution
    angRes = parameters['angRes']

    # Sensor Model parameters
    smWidth = int(parameters['smWidth']/parameters['resolution'])
    smHeight = int(parameters['smHeight']/parameters['resolution'])
    sensorRange = int(parameters['sensorRange']/parameters['resolution'])
    invModel = parameters['invModel']
    occPrior = staticPrior + dynamicPrior + weatherPrior
    freeUpGroundDetections = parameters['freeUpGroundDetections']

    return logID, is3D, initialTimeStep, simHorizon, isSLAM, velTracking, numTimeStepsSLAM, startPoseSLAM, saveVideo, removeFrames, saveSvg, videoSection, videoWidth, videoHeight, videoOrigin, style, origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits, fftConv, groundThreshold, skyThreshold, minDistance, maxDistance, voxelGridSize, angRes, smWidth, smHeight, sensorRange, invModel, occPrior, freeUpGroundDetections

if __name__ == "__main__":
    pathData = './snow_velodyne/'
    pathLabels = './snow_labels/'
    i = 100
    z_t_3D = readLidarData3DBinLabels(pathData, pathLabels, i)
    #z_t_3D.plot()
    z_t = z_t_3D.convertTo2D()
    z_t.plot()