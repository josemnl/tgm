import numpy as np
import subprocess
import os
from lidarScan import lidarScan, lidarScan3D

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

def createVideo(logID, videoPath):
    subprocess.call(['ffmpeg', '-framerate', '8', '-i', videoPath + 'frame_%d.png', '-r', '10', '-pix_fmt', 'yuv420p',videoPath + logID + '.mp4'])
    for file in os.listdir(videoPath):
        if file.endswith('.png'):
            os.remove(videoPath + file)