import numpy as np
import subprocess
import os
from lidarScan import lidarScan, lidarScan3D
import yaml
import pandas as pd
from types import SimpleNamespace

def readPose(file):
    with open(file) as data:
        x_t = np.array([line.split(",") for line in data]).astype(float)[0]
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