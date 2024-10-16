import numpy as np
import matplotlib.pyplot as plt
import time
import os

from utilities import readLidarNuScenes, createVideo, loadConfigAsDict, listFilesExt
from sensorModel import sensorModel
from TGM import TGM
from gridMap import gridMap
from SLAM import lsqnl_matching
from metrics import computeMetrics, IoU

# NuScenes stuff
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

def run():
    # Load NuScenes class
    nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=True)

    # Check if the folder exists
    if not os.path.exists('./Dataset'):
        os.makedirs('./Dataset')
    
    # Config file
    configPath = './config/'
    defConfFile = 'config'
    logID = 'nuscenes'

    # Load parameters
    conf = loadConfigAsDict(configPath, defConfFile)
    specificConf = loadConfigAsDict(configPath, logID)
    conf.__dict__.update(specificConf.__dict__)
    
    # For each scene
    for scene in nusc.scene:
        # Get the first sample_data of the lidar sensor
        sample_token = scene['first_sample_token']
        sample = nusc.get('sample', sample_token)
        sample_data_token = sample['data']['LIDAR_TOP']

        print('Scene: ' + scene['name'])

        # Scene path
        scenePath = './Dataset/' + scene['name'] + '/'

        # Create folder if it does not exist
        if not os.path.exists(scenePath):
            os.makedirs(scenePath)

        
        # Create Sensor Model and TGM
        sM = sensorModel(conf.origin, conf.smWidth, conf.smHeight, conf.resolution, conf.sensorRange, conf.invModel, conf.occPrior)
        '''
        tgm = TGM(conf.origin, conf.width, conf.height, conf.resolution, conf.staticPrior, conf.dynamicPrior, conf.weatherPrior, conf.maxVelocity, conf.saturationLimits, conf.fftConv)
        '''

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

            # Transform from sensor to ego vehicle
            calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            translation = calibrated_sensor['translation']
            rotation = calibrated_sensor['rotation']
            z_t_3D.rotate(Quaternion(rotation).rotation_matrix)
            z_t_3D.translate(np.array(translation))

            # Transform from ego vehicle to global (Only the rotation)
            ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
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
                else:
                    raise ValueError('Invalid ground filter')
                
                # Transform detections to 2D
                z_t = z_t_objects_3D.convertTo2D()
                if conf.freeUpGroundDetections:
                    z_t_ground = z_t_ground_3D.convertTo2D()
                else:
                    z_t_ground = None

                # Voxel grid filter
                if conf.isVoxelGridFilter:
                    z_t.voxelGridFilter(conf.voxelGridSize)
                    if conf.freeUpGroundDetections and conf.rayTraceGround:
                        z_t_ground.voxelGridFilter(conf.voxelGridSize)

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
            '''
            tgm.update(gm, x_t)
            '''
            timeTGM = time.time()

            # Save grid map
            fig.clear()
            #gm.plot()
            gm.saveState(scenePath + 'frame_' + str(i) + '.grid')
            '''
            tgm.plot(fig, saveMap=conf.saveMap, savePNG=conf.saveVideo, saveSvg=conf.saveSvg, imgName= scenePath + 'frame_' + str(i-conf.initialTimeStep+1), section = conf.videoSection, width=conf.videoWidth, height=conf.videoHeight, origin=conf.videoOrigin, style=conf.style)
            '''
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

if __name__ == '__main__':
    run()