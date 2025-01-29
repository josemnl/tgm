import numpy as np
import matplotlib.pyplot as plt
import time
import os

from utilities import read2DLidarCSV, read3DLidarCSV, read3DLidarBIN, read3DLabledLidarBIN, readPose, createVideo, loadConfigAsDict, listFilesExt
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching
from metrics import classificationMetrics
from gridMap import gridMap, frame

def run(logID, conf):
    # Print logID
    print('Running ' + logID)
    # Paths
    videoPath = './results/' + logID + '/'

    # Create results folder if it does not exist
    if not os.path.exists(videoPath):
        os.makedirs(videoPath)

    # Create Sensor Model and TGM
    sM = sensorModel(conf.origin, conf.smWidth, conf.smHeight, conf.resolution, conf.sensorRange, conf.invModel, conf.occPrior)
    tgmFrame = frame(conf.origin[0], conf.origin[1], conf.width, conf.height, conf.resolution)
    tgm = TGM(tgmFrame, conf.staticPrior, conf.dynamicPrior, conf.weatherPrior, conf.maxVelocity, conf.saturationLimits, conf.fftConv, conf.isGPU)

    # Empty arrays for the results
    x_t_SLAM_array = []
    nWrongSnowGrids_original_array = []
    nWrongSnowGrids_baseline_array = []
    nWrongSnowGrids_tgm_array = []
    Intersection_array = []
    Union_array = []
    IoU_array = []
    precision_array = []
    recall_array = []
    f1_array = []

    # Initial guess for the velocity
    v_t = [0, 0, 0]

    # Dict to store runtimes. Each key is a type of time, and each value is an array with the time for each frame
    runtimes = {}
    runtimes['Data'] = []
    runtimes['SLAM'] = []
    runtimes['InvSenM'] = []
    runtimes['TGM'] = []
    runtimes['Plots'] = []
    runtimes['Total'] = []

    # Main loop
    fig= plt.figure()
    for i in range(conf.initialTimeStep, conf.initialTimeStep + conf.simHorizon):
        timeStart = time.time()

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
            z_t_before_filter = z_t_objects_3D.convertTo2D() # THIS IS TO BE REMOVED
            
            # Snow filtering
            if sum([conf.isROR, conf.isSOR, conf.isDROR, conf.isDSOR]) > 1:
                print('WARNING: More than one snow filter activated')
            if conf.isROR:
                z_t_objects_3D.ROR(conf.ROR_k, conf.ROR_r)
            if conf.isSOR:
                z_t_objects_3D.SOR(conf.SOR_k, conf.SOR_s)
            if conf.isDROR:
                z_t_objects_3D.DROR(conf.DROR_k, conf.DROR_rho)
            if conf.isDSOR:
                z_t_objects_3D.DSOR(conf.DSOR_k, conf.DSOR_s, conf.DSOR_rho)
            z_t = z_t_objects_3D.convertTo2D()

            # Voxel grid filter
            if conf.isVoxelGridFilter:
                z_t.voxelGridFilter(conf.voxelGridSize)

            # Order by angle
            z_t.orderByAngle()
        
        timeData = time.time()

        # Compute robot pose with SLAM or get it from log
        if not conf.isSLAM:
            x_t = readPose(conf.lidarPath + "x_" + str(i) + ".csv")
        elif i <= conf.initialTimeStep + conf.numTimeStepsSLAM:
            try:
                x_t = readPose(conf.lidarPath + "x_" + str(i) + ".csv")
            except:
                x_t = np.array(conf.startPoseSLAM)
        else:
            x_prev = x_t
            if conf.velTracking:
                initialGuess = x_t + v_t
            else:
                initialGuess = x_t
            slamFrame = frame.frameAroundPose(x_t[0], x_t[1], conf.smWidth, conf.smHeight, tgm.frame.r)
            slam_map = tgm.oneLayer('static', slamFrame).toCPU()
            x_t = lsqnl_matching(z_t, slam_map, initialGuess, conf.sensorRange)
            v_t = x_t - x_prev
        timeSLAM = time.time()

        # Save SLAM results
        if conf.isSLAM:
            x_t_SLAM_array.append(x_t)

        # Compute instantaneous grid map with inverse sensor model
        sM.updateBasedOnPose(x_t)
        if conf.freeUpGroundDetections:
            gm = sM.generateGridMap(z_t, x_t, z_t_ground)
        else:
            gm = sM.generateGridMap(z_t, x_t)
        timeSensorModel = time.time()

        # If gm is partially outside the TGM, resize the TGM
        if not tgm.contains(gm.frame):
            newFrame = frame.frameAroundPose(x_t[0], x_t[1], tgm.frame.w, tgm.frame.h, tgm.frame.r)
            tgm.reshape(newFrame)

        # Update TGM
        tgm.update(gm, x_t)
        timeTGM = time.time()

        # Plot maps
        fig.clear()
        # Compute the frame for the plot
        if conf.videoSection == 'Full':
            plotFrame = tgm.frame
        elif conf.videoSection == 'Following':
            plotFrame = frame.frameAroundPose(x_t[0], x_t[1], int(conf.videoWidth / tgm.frame.r), int(conf.videoHeight / tgm.frame.r), tgm.frame.r)
        elif conf.videoSection == 'Constant':
            plotFrame = frame(int(conf.videoOrigin[0] / tgm.frame.r), int(conf.videoOrigin[1] / tgm.frame.r), int(conf.videoWidth / tgm.frame.r), int(conf.videoHeight / tgm.frame.r), tgm.frame.r)
        tgm.plot(fig, plotFrame, saveMap=conf.saveMap, savePNG=conf.saveVideo, saveSvg=conf.saveSvg, imgName= videoPath + 'frame_' + str(i-conf.initialTimeStep+1), style=conf.style)
        timePlot = time.time()

        # Print progress
        print('Frame:   ' + str(i-conf.initialTimeStep+1) + ' / ' + str(conf.simHorizon))

        # Add times to dict
        runtimes['Data'].append(timeData - timeStart)
        runtimes['SLAM'].append(timeSLAM - timeData)
        runtimes['InvSenM'].append(timeSensorModel - timeSLAM)
        runtimes['TGM'].append(timeTGM - timeSensorModel)
        runtimes['Plots'].append(timePlot - timeTGM)
        runtimes['Total'].append(time.time() - timeStart)

        # Print times
        print('')
        print('Data:    ' + str(timeData - timeStart))
        print('SLAM:    ' + str(timeSLAM - timeData))
        print('InvSenM: ' + str(timeSensorModel - timeSLAM))
        print('TGM:     ' + str(timeTGM - timeSensorModel))
        print('Plots:   ' + str(timePlot - timeTGM))
        print('Total:   ' + str(time.time() - timeStart))

        # Snow metrics
        if conf.isLabeled:
            '''SNOW METRICS'''
            # Compute the number of wrong snow grids before filter
            z_t_snow_before_filter = z_t_before_filter.filterInByLabel(conf.snowLabel)
            snow_grid_map_before_filter = sM.generateGridMap(z_t_snow_before_filter, x_t).toBool(conf.occPrior)
            nWrongSnowGrids_original = np.sum(snow_grid_map_before_filter.data)

            # Compute the number of wrong snow grids after filter
            z_t_snow = z_t.filterInByLabel(conf.snowLabel)
            snow_gm_baseline = sM.generateGridMap(z_t_snow, x_t).toBool(conf.occPrior)
            nWrongSnowGrids_baseline = np.sum(snow_gm_baseline.data)

            # Compute IoU
            snow_gm_our_method = tgm.maxLayer('weather', snow_gm_baseline.frame).toCPU()
            intersection, union, IoU, precision, recall, f1 = classificationMetrics(snow_gm_baseline, snow_gm_our_method)

            # Compute the number of wrong snow grids with our method
            nWrongSnowGrids_tgm = nWrongSnowGrids_baseline - intersection

            print('')
            print('Snow metrics:')
            print('Total snow grids before filter: ' + str(nWrongSnowGrids_original))
            print('Total snow grids baseline: ' + str(nWrongSnowGrids_baseline))
            print('Total snow grids our method: ' + str(nWrongSnowGrids_tgm))
            print('IoU: {:.10f}'.format(IoU))
            print('Precision: {:.10f}'.format(precision))
            print('Recall: {:.10f}'.format(recall))
            print('F1: {:.10f}'.format(f1))

            # Append results to arrays
            nWrongSnowGrids_original_array.append(nWrongSnowGrids_original)
            nWrongSnowGrids_baseline_array.append(nWrongSnowGrids_baseline)
            nWrongSnowGrids_tgm_array.append(nWrongSnowGrids_tgm)
            Intersection_array.append(intersection)
            Union_array.append(union)
            IoU_array.append(IoU)
            precision_array.append(precision)
            recall_array.append(recall)
            f1_array.append(f1)

    # Save runtimes as csv
    runtimes['Data'] = np.array(runtimes['Data'])
    runtimes['SLAM'] = np.array(runtimes['SLAM'])
    runtimes['InvSenM'] = np.array(runtimes['InvSenM'])
    runtimes['TGM'] = np.array(runtimes['TGM'])
    runtimes['Plots'] = np.array(runtimes['Plots'])
    runtimes['Total'] = np.array(runtimes['Total'])
    np.savetxt(videoPath + 'runtimes.csv', np.array(list(runtimes.values())).T, delimiter=',', header=','.join(runtimes.keys()), comments='')

    # Save SLAM results
    if conf.isSLAM:
        np.savetxt(videoPath + 'x_t_SLAM.csv', x_t_SLAM_array, delimiter=',')

    # Save snow metrics
    if conf.isLabeled:
        np.savetxt(videoPath + 'nWrongSnowGrids_original.csv', nWrongSnowGrids_original_array, delimiter=',')
        np.savetxt(videoPath + 'nWrongSnowGrids_baseline.csv', nWrongSnowGrids_baseline_array, delimiter=',')
        np.savetxt(videoPath + 'nWrongSnowGrids_tgm.csv', nWrongSnowGrids_tgm_array, delimiter=',')
        np.savetxt(videoPath + 'Intersection.csv', Intersection_array, delimiter=',')
        np.savetxt(videoPath + 'Union.csv', Union_array, delimiter=',')
        np.savetxt(videoPath + 'IoU.csv', IoU_array, delimiter=',')
        np.savetxt(videoPath + 'precision.csv', precision_array, delimiter=',')
        np.savetxt(videoPath + 'recall.csv', recall_array, delimiter=',')
        np.savetxt(videoPath + 'f1.csv', f1_array, delimiter=',')

    # Save video
    if conf.saveVideo:
        createVideo(logID, videoPath, removeFrames = conf.removeFrames)

    # Save last frame
    fig.clear()
    tgm.plot(fig, saveMap=True, imgName= videoPath + logID)

    # Save static grid map
    fig.clear()
    tgm.plot(fig, saveMap=True, imgName= videoPath + logID + '_static', style='static')

    # Save dynamic grid map
    fig.clear()
    tgm.plot(fig, saveMap=True, imgName= videoPath + logID + '_dynamic', style='dynamic')

    # Save weather grid map
    fig.clear()
    tgm.plot(fig, saveMap=True, imgName= videoPath + logID + '_weather', style='weather')

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