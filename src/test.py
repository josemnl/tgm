from gridMap import gridMap
import numpy as np
import os


scene_path = './Dataset/scene-0002/'
images_path = scene_path + 'images/'

if not os.path.exists(images_path):
    os.makedirs(images_path)

# List all files in the scene folder
filenames = os.listdir(scene_path)

# Order the files
filenames.sort()

# For each grid map file in the scene folder
for filename in filenames:
    print('Processing', filename)
    if filename.endswith('.grid'):
        # Load the grid map
        grid = gridMap.loadState(scene_path + filename, data_type = np.float32)
        # Save the grid map as an image
        grid.savePNG(images_path + filename + '.png')
