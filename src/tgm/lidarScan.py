import matplotlib.pyplot as plt
import numpy as np

class lidarScan:
    def __init__(self, angles, ranges):
        self.ranges = ranges
        self.angles = angles
        self.numReadings = len(ranges)

    def computeCartesian(self):
        return np.column_stack([self.ranges * np.cos(self.angles), self.ranges * np.sin(self.angles)])
    
    def removeNoReturn(self, maxRange):
        angles = self.angles[self.ranges < maxRange]
        ranges = self.ranges[self.ranges < maxRange]
        return lidarScan(angles, ranges)

    def computeRelativeCartesian(self, relPose):
        angles = self.angles + relPose[2]
        x = self.ranges * np.cos(angles) + relPose[0]
        y = self.ranges * np.sin(angles) + relPose[1]
        return np.column_stack([x, y])

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.computeCartesian()[:, 0], self.computeCartesian()[:, 1], 'r.')
        ax.axis('equal')
        plt.show()

    def removeClosePoints(self, minRange):
        return lidarScan(self.angles[self.ranges > minRange], self.ranges[self.ranges > minRange])

    def orderByAngle(self):
        idx = np.argsort(self.angles)
        return lidarScan(self.angles[idx], self.ranges[idx])

    def voxelGridFilter(self, voxel_size):
        points = self.computeCartesian()
        # Determine the grid indices for each point
        grid_indices = np.floor(points / voxel_size).astype(int)

        # Create a dictionary to store points in each voxel
        voxel_dict = {}
        for i, idx in enumerate(grid_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(points[i])

        # Create a list to store the downsampled points
        downsampled_points = []

        # Iterate through each voxel and average the points inside
        for key, points in voxel_dict.items():
            average_point = np.mean(points, axis=0)
            downsampled_points.append(average_point)

        downsampled_points = np.array(downsampled_points)

        return lidarScan(np.arctan2(downsampled_points[:, 1], downsampled_points[:, 0]), np.sqrt(downsampled_points[:, 0]**2 + downsampled_points[:, 1]**2))

class lidarScan3D:
    def __init__(self, points3D):
        self.points3D = points3D
        self.numReadings = len(points3D)

    def removeGround(self, groundThreshold):
        return lidarScan3D(self.points3D[self.points3D[:, 2] > groundThreshold])
    
    def removeSky(self, skyThreshold):
        return lidarScan3D(self.points3D[self.points3D[:, 2] < skyThreshold])
    
    def convertTo2D(self):
        return lidarScan(np.arctan2(self.points3D[:, 1], self.points3D[:, 0]), np.sqrt(self.points3D[:, 0]**2 + self.points3D[:, 1]**2))
    
    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax = plt.axes(projection='3d')  # Add this line to create a 3D projection
        ax.scatter(self.points3D[:, 0], self.points3D[:, 1], self.points3D[:,2], 'r')
        ax.axis('equal')
        plt.show()