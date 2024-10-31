import os

import torch
from torch.utils.data import Dataset
from gridMap import gridMap
import numpy as np
from splits import train, val, test

class NuScenesDataset(Dataset):
    def __init__(self, root_dir = './Dataset', mode = 'train', transform=None):
        assert mode in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.transform = transform

        # List all the folders in the root directory (each folder is a scene)
        #self.scenes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]

        # Load the scenes based on the mode
        if mode == 'train':
            self.scenes = train
        elif mode == 'val':
            self.scenes = val
        elif mode == 'test':
            self.scenes = test

        # For each of the folders, list all the files that start with 'frame_'
        self.in_static_grids = []
        self.in_dynamic_grids = []
        self.out_static_grids = []
        self.out_dynamic_grids = []
        self.instant_grids = []

        for scene in self.scenes:
            scene_path = os.path.join(self.root_dir, scene)
            # Grid are all the files that start with 'frame_' making sure that they are ordered by the frame number
            static_grids = sorted([f for f in os.listdir(scene_path) if f.startswith('frame_') and f.endswith('_static.grid')], key=lambda x: int(x.split('_')[1]))
            dynamic_grids = sorted([f for f in os.listdir(scene_path) if f.startswith('frame_') and f.endswith('_dynamic.grid')], key=lambda x: int(x.split('_')[1]))
            instant_grids = sorted([f for f in os.listdir(scene_path) if f.startswith('frame_') and f.endswith('_instant.grid')], key=lambda x: int(x.split('_')[1]))
            
            for i in range(len(static_grids) - 1):
                self.in_static_grids.append(os.path.join(scene_path, static_grids[i]))
                self.in_dynamic_grids.append(os.path.join(scene_path, dynamic_grids[i]))
                self.out_static_grids.append(os.path.join(scene_path, static_grids[i + 1]))
                self.out_dynamic_grids.append(os.path.join(scene_path, dynamic_grids[i + 1]))
                self.instant_grids.append(os.path.join(scene_path, instant_grids[i + 1]))

    def load_grid(self, path):
        grid = gridMap.loadState(path, data_type = np.float32)
        return grid

    def __len__(self):
        return len(self.in_static_grids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        in_static_grid = self.load_grid(self.in_static_grids[idx])
        in_dynamic_grid = self.load_grid(self.in_dynamic_grids[idx])
        out_static_grid = self.load_grid(self.out_static_grids[idx])
        out_dynamic_grid = self.load_grid(self.out_dynamic_grids[idx])
        instant_grid = self.load_grid(self.instant_grids[idx])

        # Assert that the input_static and input_dynamic have the same shape
        assert in_static_grid.width == in_dynamic_grid.width
        assert in_static_grid.height == in_dynamic_grid.height
        assert in_static_grid.origin_x == in_dynamic_grid.origin_x
        assert in_static_grid.origin_y == in_dynamic_grid.origin_y
        assert in_static_grid.resolution == in_dynamic_grid.resolution

        # Reshape the output and instant grids to match the input grids
        out_static_grid = out_static_grid.reshape(in_static_grid.origin_x, in_static_grid.origin_y, in_static_grid.width, in_static_grid.height, 0.5)
        out_dynamic_grid = out_dynamic_grid.reshape(in_dynamic_grid.origin_x, in_dynamic_grid.origin_y, in_dynamic_grid.width, in_dynamic_grid.height, 0.5)
        instant_grid = instant_grid.reshape(in_static_grid.origin_x, in_static_grid.origin_y, in_static_grid.width, in_static_grid.height, 0.5)

        # Transform grids to tensors
        in_static_grid = torch.tensor(in_static_grid.data)
        in_dynamic_grid = torch.tensor(in_dynamic_grid.data)
        out_static_grid = torch.tensor(out_static_grid.data)
        out_dynamic_grid = torch.tensor(out_dynamic_grid.data)
        instant_grid = torch.tensor(instant_grid.data)

        # Create the sample
        sample = {'input_static': in_static_grid, 'input_dynamic': in_dynamic_grid, 'output_static': out_static_grid, 'output_dynamic': out_dynamic_grid, 'output_instant': instant_grid}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
if __name__ == "__main__":
    dataset = NuScenesDataset(mode='val')

    # Print the number of samples
    print(len(dataset))

    # Load the first sample
    sample = dataset[0]

    # Plot the input_static grid
    sample['input_static'].plot(isPause=True)
    sample['input_dynamic'].plot(isPause=True)