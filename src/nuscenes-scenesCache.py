import time
import os

# NuScenes stuff
from nuscenes.nuscenes import NuScenes

# Import json
import json


def cache_scene_data(scene, nusc):
    # Get the first sample_data of the lidar sensor
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    sample_data_token = sample['data']['LIDAR_TOP']

    scene_name = scene['name']

    lidar_paths = []
    sensor_rotations = []
    sensor_translations = []
    ego_poses = []
    isKeyFrames = []
    annotations = []

    while sample_data_token != '':
        # Get sample data
        sample_data = nusc.get('sample_data', sample_data_token)

        # Find lidar path
        lidar_path = nusc.get('sample_data', sample_data_token)['filename']
        lidar_paths.append(lidar_path)

        # Get sensor translation and rotation
        calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        sensor_translation = calibrated_sensor['translation']
        sensor_translations.append(sensor_translation)
        sensor_rotation = calibrated_sensor['rotation']
        sensor_rotations.append(sensor_rotation)

        # Get ego pose
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        ego_poses.append(ego_pose)

        # Check if frame is a keyframe
        isKeyFrames.append(sample_data['is_key_frame'])

        # If the frame is a keyframe, get the annotation
        anns = []
        if sample_data['is_key_frame']:
            # Update the sample corresponding to the keyframe
            sample = nusc.get('sample', sample_data['sample_token'])
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                # If the category name contains 'vehicle' or 'human', add it to the list
                if 'vehicle' in ann['category_name'] or 'human' in ann['category_name']:
                    anns.append(ann)
        annotations.append(anns)

        # Update nuScemes' sample data token
        sample_data_token = sample_data['next']

    # Check if the folder exists
    if not os.path.exists('./Dataset'):
        os.makedirs('./Dataset')

    # Check if the scene folder exists
    if not os.path.exists('./Dataset/' + scene_name):
        os.makedirs('./Dataset/' + scene_name)

    # Save scene_name, lidar_paths, sensor_rotations, sensor_translations, ego_poses, isKeyFrame and annotations as a json file
    with open('./Dataset/' + scene_name + '/scene_data.json', 'w') as f:
        json.dump({'scene_name': scene_name, 'lidar_paths': lidar_paths, 'sensor_rotations': sensor_rotations, 'sensor_translations': sensor_translations, 'ego_poses': ego_poses, 'isKeyFrames': isKeyFrames, 'annotations': annotations}, f)


if __name__ == '__main__':
    # Load NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot='./nuscenes', verbose=True)

    # Get all the scene names
    scene_names = [scene['name'] for scene in nusc.scene]

    # Check if the folder exists
    if not os.path.exists('./Dataset'):
        os.makedirs('./Dataset')

    # Save the scene names as a json file
    with open('./Dataset/scene_names.json', 'w') as f:
        json.dump(scene_names, f)

    # Timer
    start = time.time()

    # Preprocess scenes
    for scene in nusc.scene:
        scene_name = scene['name']
        if not os.path.exists('./Dataset/' + scene_name + '/scene_data.json'):
            cache_scene_data(scene, nusc)

    print('Time to cache all scenes: ', time.time() - start)