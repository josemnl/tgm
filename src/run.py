import numpy as np
import matplotlib.pyplot as plt
import time

from utilities import read2DLidarCSV, read3DLidarCSV, read3DLidarBIN, read3DLabledLidarBIN, readPose, createVideo, loadConfigAsDict, listFilesExt
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching
from metrics import computeMetrics, IoU

def run():
    # Config file
    configPath = './config/'
    defConfFile = 'config'
    logID = 'Exp2-2024-03-15-11-25-54-Baseline'

    # Load parameters
    conf = loadConfigAsDict(configPath, defConfFile)
    specificConf = loadConfigAsDict(configPath, logID)
    conf.__dict__.update(specificConf.__dict__)

    # Paths
    videoPath = './results/' + logID + '/'

    # Create results folder if it does not exist
    import os
    if not os.path.exists(videoPath):
        os.makedirs(videoPath)

    # Create Sensor Model and TGM
    sM = sensorModel(conf.origin, conf.smWidth, conf.smHeight, conf.resolution, conf.sensorRange, conf.invModel, conf.occPrior)
    tgm = TGM(conf.origin, conf.width, conf.height, conf.resolution, conf.staticPrior, conf.dynamicPrior, conf.weatherPrior, conf.maxVelocity, conf.saturationLimits, conf.fftConv)

    # Empty arrays for the results
    x_t_SLAM_array = []
    n_occ_cells_array = []
    n_snow_occ_cells_original = []
    n_snow_occ_cells_baseline = []
    n_snow_occ_cells_our_method = []
    IoU_array = []

    # Initial guess for the velocity
    v_t = [0, 0, 0]

    # List all lidar files
    lidarFiles = listFilesExt(conf.lidarPath, conf.lidarFormat.lower())

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
                z_t_3D = read3DLidarCSV(conf.lidarPath, i)
            elif conf.lidarFormat == 'BIN':
                if conf.isLabeled:
                    z_t_3D = read3DLabledLidarBIN(conf.lidarPath, conf.labelPath, i)
                else:
                    z_t_3D = read3DLidarBIN(conf.lidarPath + lidarFiles[i])
            else:
                raise ValueError('Invalid lidar format')
        else:
            z_t = read2DLidarCSV(conf.lidarPath, i)

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
            z_t_before_filter = z_t_3D.convertTo2D() # THIS IS TO BE REMOVED
            
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
            x_t = readPose(conf.lidarPath, i)
        elif i <= conf.initialTimeStep + conf.numTimeStepsSLAM:
            try:
                x_t = readPose(conf.lidarPath, i)
            except:
                x_t = np.array(conf.startPoseSLAM)
        else:
            x_prev = x_t
            if conf.velTracking:
                initialGuess = x_t + v_t
            else:
                initialGuess = x_t
            slam_map = tgm.oneLayer('static', following=True, width=conf.smWidth, height=conf.smHeight)
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

        # Update TGM
        tgm.update(gm, x_t)
        timeTGM = time.time()

        # Plot maps
        fig.clear()
        tgm.plot(fig, saveMap=conf.saveMap, savePNG=conf.saveVideo, saveSvg=conf.saveSvg, imgName= videoPath + 'frame_' + str(i-conf.initialTimeStep+1), section = conf.videoSection, width=conf.videoWidth, height=conf.videoHeight, origin=conf.videoOrigin, style=conf.style)
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
        print('Data:    ' + str(timeData - timeStart))
        print('SLAM:    ' + str(timeSLAM - timeData))
        print('InvSenM: ' + str(timeSensorModel - timeSLAM))
        print('TGM:     ' + str(timeTGM - timeSensorModel))
        print('Plots:   ' + str(timePlot - timeTGM))
        print('Total:   ' + str(time.time() - timeStart))
        print('')

        # Snow metrics
        if conf.isLabeled:
            # Before the filter
            gm_before_filter = sM.generateGridMap(z_t_before_filter, x_t)
            n_occ_cells, n_snow_points = computeMetrics(z_t_before_filter, x_t, gm_before_filter, conf.snowLabel)
            n_snow_occ_cells_original.append(n_occ_cells)
            print('Occupied cells: ' + str(n_occ_cells) + ' / ' + str(n_snow_points))
            # Baseline: Instantaneous occupancy map
            n_occ_cells, n_snow_points = computeMetrics(z_t, x_t, gm, conf.snowLabel)
            n_snow_occ_cells_baseline.append(n_occ_cells)
            print('Occupied cells: ' + str(n_occ_cells) + ' / ' + str(n_snow_points))
            # Our method: Occupancy map from TGM
            tgm_gm = tgm.computeStaticDynamicGridMap()
            n_occ_cells, n_snow_points = computeMetrics(z_t, x_t, tgm_gm, conf.snowLabel)
            n_snow_occ_cells_our_method.append(n_occ_cells)
            print('Occupied cells: ' + str(n_occ_cells) + ' / ' + str(n_snow_points))

            # New metric: IoU
            # Compute grid map from Baseline
            #snow_gm_baseline = sM.generateGridMap(z_t_snow, x_t)
            # Compute grid map from Our method
            #snow_gm_our_method = tgm.oneLayer('weather', following=True, width=conf.smWidth, height=conf.smHeight)
            

            # Compute IoU
            #iou_baseline = IoU(snow_gm_groundTruth, snow_gm_baseline)
            #iou_our_method = IoU(snow_gm_groundTruth, snow_gm_our_method)

            #print('IoU Baseline: ' + str(iou_baseline))
            #print('IoU Our method: ' + str(iou_our_method))

            z_t_snow = z_t.filterInByLabel(conf.snowLabel)
            snow_gm_baseline = sM.generateGridMap(z_t_snow, x_t)
            snow_gm_our_method = tgm.oneLayer2('weather', origin_x=snow_gm_baseline.origin_x, origin_y=snow_gm_baseline.origin_y, width=snow_gm_baseline.width, height=snow_gm_baseline.height)
            IoU_result = IoU(snow_gm_baseline, snow_gm_our_method)
            print('IoU: {:.10f}'.format(IoU_result))
            IoU_array.append(IoU_result)

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
        np.savetxt(videoPath + 'n_snow_occ_cells_original.csv', n_snow_occ_cells_original, delimiter=',')
        np.savetxt(videoPath + 'n_snow_occ_cells_baseline.csv', n_snow_occ_cells_baseline, delimiter=',')
        np.savetxt(videoPath + 'n_snow_occ_cells_our_method.csv', n_snow_occ_cells_our_method, delimiter=',')
        np.savetxt(videoPath + 'IoU.csv', IoU_array, delimiter=',')

    # Save video
    if conf.saveVideo:
        createVideo(logID, videoPath, removeFrames = conf.removeFrames)

    # Save last frame
    tgm.plot(fig, saveMap=True, imgName= videoPath + logID)

    # Save static grid map
    tgm.plot(fig, saveMap=True, imgName= videoPath + logID + '_static', style='static')

    # Save dynamic grid map
    tgm.plot(fig, saveMap=True, imgName= videoPath + logID + '_dynamic', style='dynamic')

    # Save weather grid map
    tgm.plot(fig, saveMap=True, imgName= videoPath + logID + '_weather', style='weather')

if __name__ == '__main__':
    run()