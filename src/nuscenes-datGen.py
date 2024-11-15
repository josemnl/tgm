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

# Rotations
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

# Parallel processing
import concurrent.futures

# Import json
import json

# For each scene
def process_scene(scene_name, conf):

    # Load scene data
    with open('./Dataset/' + scene_name + '/scene_data.json', 'r') as f:
        scene_data = json.load(f)
        lidar_paths = scene_data['lidar_paths']
        sensor_rotations = scene_data['sensor_rotations']
        sensor_translations = scene_data['sensor_translations']
        ego_poses = scene_data['ego_poses']

    print('Processing scene: ' + scene_name)

    # Scene path
    scenePath = './Dataset/' + scene_name + '/'

    # Create folder if it does not exist
    if not os.path.exists(scenePath):
        os.makedirs(scenePath)

    
    # Create Sensor Model and TGM
    sM = sensorModel(conf.origin, conf.smWidth, conf.smHeight, conf.resolution, conf.sensorRange, conf.invModel, conf.occPrior)
    tgm = TGM(conf.origin, conf.width, conf.height, conf.resolution, conf.staticPrior, conf.dynamicPrior, conf.weatherPrior, conf.maxVelocity, conf.saturationLimits, conf.fftConv)

    # Main loop
    #fig= plt.figure()
    print('Hello')
    i = 0
    for lidar_path, sensor_rotation, sensor_translation, ego_pose in zip(lidar_paths, sensor_rotations, sensor_translations, ego_poses):
        i += 1
        timeStart = time.time()

        # Import sensor data
        z_t_3D = readLidarNuScenes('./nuscenes/' + lidar_path)

        # Transform from sensor to ego vehicle
        z_t_3D.rotate(Quaternion(sensor_rotation).rotation_matrix)
        z_t_3D.translate(np.array(sensor_translation))

        # Remove points in box                                  # PLACING THIS HERE IS A HACK
        z_t_3D.removePointsInBox(conf.box)

        # Transform from ego vehicle to global (Only the rotation)
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
            
            # Transform detections to 2D
            z_t = z_t_objects_3D.convertTo2D()
            if conf.freeUpGroundDetections:
                z_t_ground = z_t_ground_3D.convertTo2D()
            else:
                z_t_ground = None

            # Voxel grid filter
            if conf.isVoxelGridFilter:
                z_t.fastVoxelGridFilter(conf.voxelGridSize)
                if conf.freeUpGroundDetections and conf.rayTraceGround:
                    z_t_ground.fastVoxelGridFilter(conf.voxelGridSize)

            # Order by angle
            z_t.orderByAngle()
        
        timeData = time.time()

        # Correct pose
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
        tgm.update(gm, x_t)
        timeTGM = time.time()

        # Save grid map
        #fig.clear()
        #gm.plot()
        origin_x = gm.origin_x
        origin_y = gm.origin_y
        width = gm.width
        height = gm.height
        static = tgm.oneLayer2('static',origin_x,origin_y,width,height)
        dynamic = tgm.oneLayer2('dynamic',origin_x,origin_y,width,height)
        gm.saveState(scenePath + 'frame_' + str(i) + '_instant.grid')
        static.saveState(scenePath + 'frame_' + str(i) + '_static.grid')
        dynamic.saveState(scenePath + 'frame_' + str(i) + '_dynamic.grid')

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

# Main execution block
if __name__ == "__main__":
    # Load scene names
    with open('./Dataset/scene_names.json', 'r') as f:
        scene_names = json.load(f)

    print('Scene names: ', scene_names)

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

    # Find which scenes have already been processed
    processed_scenes = []
    for scene_name in scene_names:
        if os.path.exists('./Dataset/' + scene_name + '/frame_1_instant.grid'):
            processed_scenes.append(scene_name)
    
    # Remove already processed scenes
    scene_names = list(set(scene_names) - set(processed_scenes))

    # Reorder the scenes
    scene_names.sort()

    # Use ProcessPoolExecutor to parallelize scene processing
    max_workers = 7
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_scene, scene_name, conf) for scene_name in scene_names]
        concurrent.futures.wait(futures)
    print('Total time: ', time.time() - start_time)