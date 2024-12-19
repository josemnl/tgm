import numpy as np
import matplotlib.pyplot as plt
import time

from utilities import readLidarNuScenes, createVideo, loadConfigAsDict, listFilesExt
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching
from metrics import computeMetrics, IoU
from gridMap import gridMap, frame

# NuScenes stuff
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

# NN stuff
import torch
from unet_model import UNet

def run():
    # NuScenes: Load the first scene
    nusc = NuScenes(version='v1.0-trainval', dataroot='./nuscenes', verbose=True)
    scene = nusc.scene[1]
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

    # Load NN model
    model = UNet(2,3)
    #model.load_state_dict(torch.load('./trainRuns/UNet_batchSize_10_lr_1e-05_epochs_10_date_20241104-183629/checkpoint3_13000.pt'))    # Masked
    #model.load_state_dict(torch.load('./trainRuns/UNet_batchSize_10_lr_1e-05_epochs_10_date_20241108-173804/model.pt'))                # Without mask
    #model.load_state_dict(torch.load('./trainRuns/UNet_batchSize_10_lr_1e-05_epochs_10_date_20241114-115655/checkpoint0_18000.pt'))    # Without sat limit
    #model.load_state_dict(torch.load('./trainRuns/UNet_batchSize_10_lr_1e-05_epochs_10_date_20241114-161520/checkpoint3_21000.pt'))    # Biased
    #model.load_state_dict(torch.load('./trainRuns/UNet_batchSize_10_lr_1e-05_epochs_10_date_20241115-185109/model.pt'))                # Masked and biased
    #model.load_state_dict(torch.load('./trainRuns/UNet_batchSize_10_lr_1e-05_epochs_10_date_20241119-163322/model.pt'))                # NAN - Masked and biased, without softmax the target
    #model.load_state_dict(torch.load('./trainRuns/UNet_batchSize_10_lr_1e-05_epochs_10_date_20241125-161602/checkpoint3_17000.pt'))     # Masked and biased, without softmax the target
    #model.load_state_dict(torch.load('./trainRuns/UNet_batchSize_10_lr_1e-05_epochs_10_date_20241204-104234/checkpoint2_15000.pt'))     # Masked and biased, without softmax the target
    model.load_state_dict(torch.load('./trainRuns/UNet_batchSize_10_lr_1e-05_epochs_10_date_20241218-114305/checkpoint0_21000.pt'))     # TRAINED WITH NEW CODE

    # Create Sensor Model and TGM
    sM = sensorModel(conf.origin, conf.smWidth, conf.smHeight, conf.resolution, conf.sensorRange, conf.invModel, conf.occPrior)
    tgmFrame = frame(conf.origin[0], conf.origin[1], conf.width, conf.height, conf.resolution)
    tgm = TGM(tgmFrame, conf.staticPrior, conf.dynamicPrior, conf.weatherPrior, conf.maxVelocity, conf.saturationLimits, conf.fftConv, predictionMode = 'random', model = model)

    # Empty arrays for the results
    x_t_SLAM_array = []
    intersection_array = []
    union_array = []
    IoU_array = []

    # Initial guess for the velocity
    v_t = [0, 0, 0]

    # Main loop
    fig= plt.figure()
    i = 0
    while sample_data_token != '':
        i += 1
        if i == 1:
            tgm.switchPredictions(predictionMode='NNDynamic')
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

        # Remove points in box                                  # PLACING THIS HERE IS A HACK
        z_t_3D.removePointsInBox(conf.box)

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

        '''
        EVALUATION
        '''
        if sample_data['is_key_frame']:
            # Create an empty grid map with the same size and resolution as gm
            gm_gt = gridMap(gm.frame, np.zeros((gm.frame.w, gm.frame.h)))
            # Get the annotation
            anns = []
            if sample_data['is_key_frame']:
                # Update the sample corresponding to the keyframe
                sample = nusc.get('sample', sample_data['sample_token'])
                for ann_token in sample['anns']:
                    ann = nusc.get('sample_annotation', ann_token)
                    # If the category name contains 'vehicle' or 'human', add it to the list
                    if 'vehicle' in ann['category_name'] or 'human' in ann['category_name']:
                        anns.append(ann)
            # Draw the bounding boxes in the ground truth grid map
            for ann in anns:
                # Get position, orientation and bounding box
                position = ann['translation']
                orientation = ann['rotation']
                size = ann['size']

                # Compute the orientation as an angle and discard the z component of the position
                orientation = -Rotation.from_quat(orientation).as_euler('zyx')[2] + np.pi/2
                position = np.array([position[0], position[1]])

                # Correct position by x_t_diff
                position = position - np.array([x_t_diff[0], x_t_diff[1]])

                # Draw the bounding box
                gm_gt.drawFilledRectangle(position[0], position[1], orientation, size[0], size[1], 1.0)

            # Compute IoU for dynamic grid map
            #dynamic = tgm.oneLayer('dynamic', gm.frame).toBool(0.5).toCPU()
            dynamic = tgm.maxLayer('dynamic', gm.frame).toBool(0.5).toCPU()
            dynamic_gt = gm_gt.toBool(0.5)
            # The mask is the visible area, which is the free or occupied area in the instantaneous grid map
            mask = gridMap(gm.frame, np.logical_or(gm.data == conf.invModel[0], gm.data == conf.invModel[1]))
            intersection, union, IoU_result = IoU(dynamic, dynamic_gt)
            intersection_array.append(intersection)
            union_array.append(union)
            IoU_array.append(IoU_result)

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

    # Save IoU results
    np.savetxt(videoPath + 'intersection.csv', intersection_array, delimiter=',')
    np.savetxt(videoPath + 'union.csv', union_array, delimiter=',')
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