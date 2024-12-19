from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from utilities import read3DLidarBIN, loadConfigAsDict
import matplotlib.pyplot as plt

# Config file
configPath = './config/'
defConfFile = 'config'
logID = 'nuscenes'

# Load parameters
conf = loadConfigAsDict(configPath, defConfFile)
specificConf = loadConfigAsDict(configPath, logID)
conf.__dict__.update(specificConf.__dict__)

nusc = NuScenes(version='v1.0-mini', dataroot='./nuscenes', verbose=True)

'''
nusc.list_scenes()
my_scene = nusc.scene[0]

# Loop through each sample in the scene
my_sample_token = my_scene['first_sample_token']
i = 0
while my_sample_token != '':
    my_sample = nusc.get('sample', my_sample_token)
    print(my_sample['timestamp'])
    i += 1
    print(i)
    my_sample_token = my_sample['next']

my_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', my_sample_token)

#print(my_sample['data'])

# Loop through each sample_data in the scene
for my_sample_data_token in my_sample['data']:
    sensor = 'CAM_FRONT'
    my_sample_data = nusc.get('sample_data', my_sample['data'][sensor])
    # Check if that sample_data is a keyframe
    print(my_sample_data['is_key_frame'])


'''
# Get the first scene in the scene table
# Get the first sample in the scene and from that sample, 
# get the first sample_data of the lidar sensor

scene = nusc.scene[0]
sample_token = scene['first_sample_token']

sample = nusc.get('sample', sample_token)
sample_data_token = sample['data']['LIDAR_TOP']
i = 1

fig, ax = plt.subplots()

# Loop through each sample data in the scene
while sample_data_token != '':
    sample_data = nusc.get('sample_data', sample_data_token)
    print(i)
    i += 1
    print(sample_data['is_key_frame'])
    # Render the images in the sample data
    #nusc.render_sample_data(sample_data_token)
    # Get the name of the file
    filename = sample_data['filename']
    # Load the Lidar point cloud data
    lidar_path = nusc.get('sample_data', sample_data_token)['filename']
    z_t_3D = read3DLidarBIN('./nuscenes/' + lidar_path)
    z_t = z_t_3D.convertTo2D()
    #z_t.plot()
    lidar = LidarPointCloud.from_file('./nuscenes/' + lidar_path)
    # Render the Lidar point cloud
    lidar.render_height(ax=ax)
    # Pause to show the image
    
    # Force to draw the image
    plt.draw()
    plt.pause(1)
    # Print pose
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    print(ego_pose['translation'])
    print(ego_pose['rotation'])
    # Get next sample data token
    sample_data_token = sample_data['next']