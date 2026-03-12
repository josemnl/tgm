import numpy as np
import subprocess
import os
from lidarScan import lidarScan, lidarScan3D
import yaml
import pandas as pd
from types import SimpleNamespace
from spatial import position, orientation, pose

def readPose(file):
    """Read a single-line pose CSV file.

    Supported formats (comma-separated, first non-empty line only):
    - 3 values: x, y, yaw (z assumed 0)
    - 6 values: x, y, z, roll, pitch, yaw
    - 7 values: x, y, z, qx, qy, qz, qw (quaternion -> roll, pitch, yaw)
    """
    # Use utf-8-sig to gracefully handle potential BOM
    with open(file, 'r', encoding='utf-8-sig', newline='') as f:
        first_non_empty = None
        for line in f:
            line = line.strip()
            if line:
                first_non_empty = line
                break

        if first_non_empty is None:
            raise ValueError('Empty pose file: ' + file)

        parts = [p.strip() for p in first_non_empty.split(',') if p.strip() != '']
        n = len(parts)

        # Parse as floats if count is valid
        if n not in (3, 6, 7):
            raise ValueError(
                'Invalid pose format in file: ' + file + '. Expected 3, 6 or 7 values per line, got ' + str(n)
            )

        vals = list(map(float, parts))

        if n == 3:
            x, y, yaw = vals
            return pose(position(x, y, 0.0), orientation(0.0, 0.0, yaw))

        if n == 6:
            x, y, z, roll, pitch, yaw = vals
            return pose(position(x, y, z), orientation(roll, pitch, yaw))

        # n == 7: quaternion -> roll, pitch, yaw
        x, y, z, qx, qy, qz, qw = vals
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2)  # clamp at 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        ori_obj = orientation(roll, pitch, yaw)
        return pose(position(x, y, z), ori_obj)

def read2DLidarCSV(file):
    with open(file) as data:
        z_t = lidarScan(*np.array([line.split(",") for line in data]).astype(float).T)
    return z_t

def read3DLidarCSV(file):
    data = pd.read_csv(file, header=None)
    z_t_3D = lidarScan3D(data.values.astype(float))
    return z_t_3D

def read3DLidarBIN(file, n_fields=4):
    rawdata = np.fromfile(file, dtype=np.float32)
    # Convert raw data to float
    rawdata = rawdata.astype(float)
    data = np.reshape(rawdata, (-1, n_fields))
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
    # One-liner: shallow-normalize 'None' (string) to Python None across top-level keys
    config = {k: (None if isinstance(v, str) and v.strip().lower() == 'none' else v) for k, v in config.items()}
    # Wrap into a SimpleNamespace for attribute access
    config = SimpleNamespace(**config)
    # Convert meters to cells
    config.width = int(config.width/config.resolution)
    config.height = int(config.height/config.resolution)
    if hasattr(config, 'depth'):
        config.depth = int(config.depth/config.resolution)
    config.smWidth = int(config.smWidth/config.resolution)
    config.smHeight = int(config.smHeight/config.resolution)
    if hasattr(config, 'smDepth'):
        config.smDepth = int(config.smDepth/config.resolution)
    config.sensorRange = int(config.sensorRange/config.resolution)
    config.maxVelocity = int(config.maxVelocity/config.resolution)
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