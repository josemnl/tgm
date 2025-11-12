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

    # Arrays for the new snow metrics
    IoU_b_array = []
    precision_b_array = []
    recall_b_array = []
    f1_b_array = []
    IoU_t_array = []
    precision_t_array = []
    recall_t_array = []
    f1_t_array = []
    IoU_t_b_array = []
    precision_t_b_array = []
    recall_t_b_array = []
    f1_t_b_array = []

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
                z_t_ground.voxelGridFilter(conf.voxelGridSize)

            # Order by angle
            z_t.orderByAngle()
        
        timeData = time.time()

        # Compute robot pose with SLAM or get it from log
        if not conf.isSLAM:
            x_t = readPose(conf.posePath + "x_" + str(i).zfill(6) + ".csv")
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

        # Special plots for snow
        if i == 14 or i == 349 or i == 846:
            plotFrameSnow = frame.frameAroundPose(x_t[0], x_t[1], int(conf.videoWidth / tgm.frame.r), int(20 / tgm.frame.r), tgm.frame.r)
            gm_unfiltered = sM.generateGridMap(z_t_before_filter, x_t)
            fig.clear()
            # Create a new unfiltered tgm with the same frame
            tgm_unfiltered = TGM(tgm.frame, conf.staticPrior, conf.dynamicPrior, conf.weatherPrior, conf.maxVelocity, conf.saturationLimits, conf.fftConv, conf.isGPU)
            tgm_unfiltered.update(gm_unfiltered, x_t)
            tgm_unfiltered.plot(fig, plotFrameSnow, saveMap=conf.saveMap, savePNG=conf.saveVideo, saveSvg=conf.saveSvg, imgName= videoPath + 'frame_' + str(i-conf.initialTimeStep+1) + '_unfiltered', style=conf.style)
            
            gm_filtered = sM.generateGridMap(z_t, x_t)
            fig.clear()
            tgm_filtered = TGM(tgm.frame, conf.staticPrior, conf.dynamicPrior, conf.weatherPrior, conf.maxVelocity, conf.saturationLimits, conf.fftConv, conf.isGPU)
            tgm_filtered.update(gm_filtered, x_t)
            tgm_filtered.plot(fig, plotFrameSnow, saveMap=conf.saveMap, savePNG=conf.saveVideo, saveSvg=conf.saveSvg, imgName= videoPath + 'frame_' + str(i-conf.initialTimeStep+1) + '_filtered', style=conf.style)

            tgm.plot(fig, plotFrameSnow, saveMap=conf.saveMap, savePNG=conf.saveVideo, saveSvg=conf.saveSvg, imgName= videoPath + 'frame_' + str(i-conf.initialTimeStep+1) + '_tgm', style=conf.style)


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
            gm_baseline_snow = sM.generateGridMap(z_t_snow, x_t).toBool(conf.occPrior)
            nWrongSnowGrids_baseline = np.sum(gm_baseline_snow.data)

            # Compute IoU
            snow_gm_our_method = tgm.maxLayer('weather', gm_baseline_snow.frame).toCPU()
            intersection, union, IoU, precision, recall, f1 = classificationMetrics(gm_baseline_snow, snow_gm_our_method)

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

            # --------------------------------------------------------

            # NEW SNOW METRICS
            # Here we aim to compare the performance of the baselines vs baselines + TGM

            # Compute the original grid maps (full and snow)
            gm_original_full = sM.generateGridMap(z_t_before_filter, x_t).toBool(conf.occPrior)
            gm_original_snow = sM.generateGridMap(z_t_before_filter.filterInByLabel(conf.snowLabel), x_t).toBool(conf.occPrior)
            non_snow_gm_original = sM.generateGridMap(z_t_before_filter.filterOutByLabel(conf.snowLabel), x_t).toBool(conf.occPrior)
            gm_original_snow = gm_original_snow.diff(non_snow_gm_original) # This is to make sure that cells that contain both snow and non-snow are considered as non-snow

            # Compute the baseline grid maps (full and snow)
            gm_baseline_full = sM.generateGridMap(z_t, x_t).toBool(conf.occPrior)
            gm_baseline_snow = sM.generateGridMap(z_t.filterInByLabel(conf.snowLabel), x_t).toBool(conf.occPrior)
            non_snow_gm_baseline = sM.generateGridMap(z_t.filterOutByLabel(conf.snowLabel), x_t).toBool(conf.occPrior)
            gm_baseline_snow = gm_baseline_snow.diff(non_snow_gm_baseline) # This is to make sure that cells that contain both snow and non-snow are considered as non-snow

            # Compute the cells that had been removed by the baseline
            gm_removed_by_baseline = gm_original_full.diff(gm_baseline_full)

            # Compute the snow grid map with the baseline + TGM
            gm_removed_by_tgm = tgm.maxLayer('weather', gm_baseline_snow.frame).toCPU()

            # Compute the snow cells that had been removed by the baseline + TGM
            gm_removed_by_baseline_and_tgm = gm_removed_by_baseline.union(gm_removed_by_tgm)

            # Compute metrics baseline / original
            intersection_b, union_b, IoU_b, precision_b, recall_b, f1_b = classificationMetrics(gm_original_snow, gm_removed_by_baseline)

            # Compute metrics baseline + TGM / original
            intersection_t, union_t, IoU_t, precision_t, recall_t, f1_t = classificationMetrics(gm_original_snow, gm_removed_by_baseline_and_tgm)

            # Compute metrics baseline + TGM / baseline
            intersection_t_b, union_t_b, IoU_t_b, precision_t_b, recall_t_b, f1_t_b = classificationMetrics(gm_baseline_snow, gm_removed_by_tgm)

            # Append results to arrays
            IoU_b_array.append(IoU_b)
            precision_b_array.append(precision_b)
            recall_b_array.append(recall_b)
            f1_b_array.append(f1_b)
            IoU_t_array.append(IoU_t)
            precision_t_array.append(precision_t)
            recall_t_array.append(recall_t)
            f1_t_array.append(f1_t)
            IoU_t_b_array.append(IoU_t_b)
            precision_t_b_array.append(precision_t_b)
            recall_t_b_array.append(recall_t)
            f1_t_b_array.append(f1_t_b)

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

        # Save new snow metrics
        np.savetxt(videoPath + 'IoU_b.csv', IoU_b_array, delimiter=',')
        np.savetxt(videoPath + 'precision_b.csv', precision_b_array, delimiter=',')
        np.savetxt(videoPath + 'recall_b.csv', recall_b_array, delimiter=',')
        np.savetxt(videoPath + 'f1_b.csv', f1_b_array, delimiter=',')
        np.savetxt(videoPath + 'IoU_t.csv', IoU_t_array, delimiter=',')
        np.savetxt(videoPath + 'precision_t.csv', precision_t_array, delimiter=',')
        np.savetxt(videoPath + 'recall_t.csv', recall_t_array, delimiter=',')
        np.savetxt(videoPath + 'f1_t.csv', f1_t_array, delimiter=',')
        np.savetxt(videoPath + 'IoU_t_b.csv', IoU_t_b_array, delimiter=',')
        np.savetxt(videoPath + 'precision_t_b.csv', precision_t_b_array, delimiter=',')
        np.savetxt(videoPath + 'recall_t_b.csv', recall_t_b_array, delimiter=',')
        np.savetxt(videoPath + 'f1_t_b.csv', f1_t_b_array, delimiter=',')

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