import numpy as np
import matplotlib.pyplot as plt
import time
import os

from gridMap import discreteDist
from utilities import read2DLidarCSV, read3DLidarCSV, read3DLidarBIN, read3DLabledLidarBIN, readPose, createVideo, loadConfigAsDict
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching
from metrics import classificationMetrics
from spatial import frame, origin, size, position, pose, orientation
from gridMap import gridMap, discreteDist

def run():
    # Bool cardinality
    isCardinality = True
    # Paths
    videoPath = './results/cardinality/'

    # Create results folder if it does not exist
    if not os.path.exists(videoPath):
        os.makedirs(videoPath)

    # Create TGM
    tgmOrigin = origin(0, 0, 0)
    tgmSize = size(20, 10, 1)
    tgmFrame = frame(tgmOrigin, tgmSize, 1.0)
    tgm = TGM(tgmFrame, [0.25, 0.25, 0.0], 1, [0, 1, 0, 1], False, True)

    # Initial cardinality distribution
    cadinalityDist = tgm.dynamicCardinality
    # Create array of zeros with length equal to number of cells + 1
    # initialProbabilities = np.zeros(tgm.frame.size.w * tgm.frame.size.h + 1)
    # Set probability of 4 cells to 1
    # initialProbabilities[4] = 1.0
    # cadinalityDist = discreteDist(initialProbabilities, np.arange(tgm.frame.size.w * tgm.frame.size.h + 1))

    # Main loop
    fig= plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Initial plots
    cadinalityDist.plot(fig, ax2)
    plotFrame = tgm.frame
    tgm.plot(fig, ax1, plotFrame, saveMap=False, savePNG=True, saveSvg=True, imgName= videoPath + 'frame_0', style='combined')

    # Pause to view initial plots
    plt.pause(2)

    for i in range(1, 106):

        # Load instantaneous grid map
        gm = gridMap.loadFromPNG('./logs/cardinality/' + str(i) + '.png', tgmOrigin, 1.0)

        # Compute dynamic cardinality before update
        if isCardinality:
            dynamicCardinality_before = tgm.dynamicCardinality
            print('Expected cardinality before update:', dynamicCardinality_before.expected_value())

        # Update TGM
        tgm.update(gm, None)

        # Compute dynamic cardinality after update
        if isCardinality:
            dynamicCardinality_after = tgm.dynamicCardinality
            print('Expected cardinality after update:', dynamicCardinality_after.expected_value())

        # Compute the likelihood of the current observation
        if isCardinality:
            # Safe likelihood computation to avoid division by zero/NaNs
            before = dynamicCardinality_before.probabilities
            after = dynamicCardinality_after.probabilities
            eps = 1e-12
            safe_before = np.where(before <= 0, eps, before)
            likelihood_obs = after / safe_before
            likelihood_obs = np.nan_to_num(likelihood_obs, nan=0.0, posinf=1.0, neginf=0.0)
            likelihood_obs = discreteDist(likelihood_obs, cadinalityDist.values).normalize()

        # Update the cardinality distribution
        if isCardinality:
            cadinalityDist = discreteDist(cadinalityDist.probabilities * likelihood_obs.probabilities, cadinalityDist.values).normalize()

        # Update the TGM to follow the new cardinality distribution
        if isCardinality:
            tgm.dynamicRebalance(cadinalityDist)

        # Plot cardinality distribution
        cadinalityDist.plot(fig, ax2)

        # Plot map
        plotFrame = tgm.frame
        tgm.plot(fig, ax1, plotFrame, saveMap=False, savePNG=True, saveSvg=False, imgName= videoPath + 'frame_' + str(i), style='combined')

        # Print progress
        print('Frame:   ' + str(i) + ' / ' + '106')

    # Save video
    createVideo('Cardinality', videoPath, removeFrames = False)

if __name__ == '__main__':

    run()