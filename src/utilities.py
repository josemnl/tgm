import numpy as np
import subprocess
import os
from lidarScan import lidarScan, lidarScan3D
import yaml
import pandas as pd
from types import SimpleNamespace
from gridMap import pose, position, orientation

def readPose(file):
    with open(file) as data:
        # if there are only 3 values in the line, it is x, y, yaw
        # if there are 6 values in the line, it is x, y, z, roll, pitch, yaw

        if len(data.readline().split(",")) == 3:
            data.seek(0)
            x_t = np.array([line.split(",") for line in data]).astype(float)[0]
            x_t = pose(position(x_t[0], x_t[1], 0.0), orientation(0.0, 0.0, x_t[2]))
        else:
            data.seek(0)
            x_t = np.array([line.split(",") for line in data]).astype(float)[0]
            x_t = pose(position(x_t[0], x_t[1], x_t[2]), orientation(x_t[3], x_t[4], x_t[5]))
    return x_t

def read2DLidarCSV(file):
    with open(file) as data:
        z_t = lidarScan(*np.array([line.split(",") for line in data]).astype(float).T)
    return z_t

def read3DLidarCSV(file):
    data = pd.read_csv(file, header=None)
    z_t_3D = lidarScan3D(data.values.astype(float))
    return z_t_3D

def read3DLidarBIN(file):
    rawdata = np.fromfile(file, dtype=np.float32)
    # Convert raw data to float
    rawdata = rawdata.astype(float)
    data = np.reshape(rawdata, (-1, 4))
    z_t_3D = lidarScan3D(data[:,0:3])
    return z_t_3D

def read3DLabledLidarBIN(lidarFile, labelFile):
    rawdata = np.fromfile(lidarFile, dtype=np.float32)
    # Convert raw data to float
    rawdata = rawdata.astype(float)
    data = np.reshape(rawdata, (-1, 4))
    rawlabels = np.fromfile(labelFile, dtype=np.uint32)
    labels = np.reshape(rawlabels, -1)
    z_t_3D = lidarScan3D(data[:,0:3], labels)
    return z_t_3D

def listFilesExt(path, ext):
    return sorted([f for f in os.listdir(path) if f.endswith(ext)])

def createVideo(logID, videoPath, removeFrames = True):
    subprocess.call(['ffmpeg', '-framerate', '8', '-i', videoPath + 'frame_%d.png', '-r', '10', '-pix_fmt', 'yuv420p',videoPath + logID + '.mp4'])
    if removeFrames:
        for file in os.listdir(videoPath):
            if file.endswith('.png') and not file.endswith('_map.png'):
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

def loadConfigAsDict(configPath, configFile):
    # Import parameters from config file
    config = yaml.safe_load(open(configPath + configFile + '.yaml'))
    config = SimpleNamespace(**config)
    # Convert meters to cells
    config.width = int(config.width/config.resolution)
    config.height = int(config.height/config.resolution)
    config.smWidth = int(config.smWidth/config.resolution)
    config.smHeight = int(config.smHeight/config.resolution)
    config.sensorRange = int(config.sensorRange/config.resolution)
    # Compute occupancy prior
    config.occPrior = config.staticPrior + config.dynamicPrior + config.weatherPrior
    # Voxel grid size is the same as the resolution
    config.voxelGridSize = config.resolution
    return config

if __name__ == "__main__":
    pathData = './SnowyKITTI/dataset/sequences/00/snow_velodyne/'
    pathLabels = './SnowyKITTI/dataset/sequences/00/snow_labels/'
    i = 100
    z_t_3D = read3DLabledLidarBIN(pathData + str(i).zfill(6) + '.bin', pathLabels + str(i).zfill(6) + '.label')
    #z_t_3D.plot()
    z_t = z_t_3D.convertTo2D()
    z_t.plot()