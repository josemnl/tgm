import numpy as np
import matplotlib.pyplot as plt
import time

from utilities import readLidarNuScenes, createVideo, loadConfigAsDict, listFilesExt
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching
from metrics import computeMetrics, IoU

# NuScenes stuff
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

def run():
    # NuScenes: Load the first scene
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=True)
    scene = nusc.scene[0]
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    sample_data_token = sample['data']['LIDAR_TOP']

    # Config file
    configPath = './config/'
    defConfFile = 'config'
    logID = 'nuscenes'

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

    # Main loop
    fig= plt.figure()
    i = 0
    while sample_data_token != '':
        i += 1
        timeStart = time.time()

        # Import sensor data
        sample_data = nusc.get('sample_data', sample_data_token)
        lidar_path = nusc.get('sample_data', sample_data_token)['filename']
        z_t_3D = readLidarNuScenes('./nuscenes/' + lidar_path)
        #pc = LidarPointCloud.from_file('./nuscenes/' + lidar_path)

        # Transform from sensor to ego vehicle
        calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        translation = calibrated_sensor['translation']
        rotation = calibrated_sensor['rotation']
        #pc.rotate(Quaternion(rotation).rotation_matrix)
        #pc.translate(np.array(translation))
        z_t_3D.rotate(Quaternion(rotation).rotation_matrix)
        z_t_3D.translate(np.array(translation))

        # Transform from ego vehicle to global (Only the rotation)
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        #pc.rotate(Quaternion(ego_pose['rotation']).rotation_matrix)
        z_t_3D.rotate(Quaternion(ego_pose['rotation']).rotation_matrix)

        # Filter the point cloud
        if conf.is3D:
            # Remove close, far and sky points
            z_t_3D.removeClosePoints(conf.minDistance)
            z_t_3D.removeFarPoints(conf.maxDistance)
            z_t_3D.removeSky(conf.skyThreshold)
            
            # Split ground and objects
            if conf.groundFilter == 'RANSAC':
                z_t_ground_3D, z_t_objects_3D = z_t_3D.RANSAC(conf.ransacDistance, conf.ransacIterations)
            elif conf.groundFilter == 'Height':
                z_t_ground_3D, z_t_objects_3D = z_t_3D.splitByHeight(conf.groundThreshold)
            elif conf.groundFilter == 'RMF':
                z_t_ground_3D, z_t_objects_3D = z_t_3D.RMF_GroundSeg()
            else:
                raise ValueError('Invalid ground filter')
            
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

            # Transform detections to 2D
            z_t = z_t_objects_3D.convertTo2D()
            if conf.freeUpGroundDetections:
                z_t_ground = z_t_ground_3D.convertTo2D()
            else:
                z_t_ground = None

            # Voxel grid filter
            if conf.isVoxelGridFilter:
                # The fast voxel grid filter might break SLAM
                z_t.fastVoxelGridFilter(conf.voxelGridSize)
                if conf.freeUpGroundDetections and conf.rayTraceGround:
                    z_t_ground.fastVoxelGridFilter(conf.voxelGridSize)

            # Order by angle
            z_t.orderByAngle()
        
        timeData = time.time()

        # Get pose from log
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        angle = -Rotation.from_quat(ego_pose['rotation']).as_euler('zyx')[2] + np.pi
        x_t = np.array([ego_pose['translation'][0], ego_pose['translation'][1], angle])

        # Make the first pose the origin without changing the angle
        if i == 1:
            x_t_diff = np.array([x_t[0] - conf.startPoseNuScenes[0], x_t[1] - conf.startPoseNuScenes[1], 0])
        x_t = x_t - x_t_diff
        
        timeSLAM = time.time()

        # Save SLAM results
        if conf.isSLAM:
            x_t_SLAM_array.append(x_t)

        # Compute instantaneous grid map with inverse sensor model
        sM.updateBasedOnPose(x_t)
        if conf.freeUpGroundDetections:
            x_t_prime = np.array([x_t[0], x_t[1], 0])
            gm = sM.generateGridMap(z_t, x_t_prime, z_t_ground, conf.rayTraceGround)
        else:
            x_t_prime = np.array([x_t[0], x_t[1], 0])
            gm = sM.generateGridMap(z_t, x_t_prime)
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

        # Print times
        print('Data:    ' + str(timeData - timeStart))
        print('SLAM:    ' + str(timeSLAM - timeData))
        print('InvSenM: ' + str(timeSensorModel - timeSLAM))
        print('TGM:     ' + str(timeTGM - timeSensorModel))
        print('Plots:   ' + str(timePlot - timeTGM))
        print('Total:   ' + str(time.time() - timeStart))
        print('')

        # Update nuScemes' sample data token
        sample_data_token = sample_data['next']

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