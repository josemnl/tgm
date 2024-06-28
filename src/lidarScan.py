import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

class lidarScan:
    def __init__(self, angles, ranges, labels=None):
        assert len(angles) == len(ranges)
        if labels is not None:
            assert len(angles) == len(labels)
        
        self.ranges = ranges
        self.angles = angles
        self.numReadings = len(ranges)
        self.labels = labels

    def computeCartesian(self):
        return np.column_stack([self.ranges * np.cos(self.angles), self.ranges * np.sin(self.angles)])

    def computeRelativeCartesian(self, relPose):
        angles = self.angles + relPose[2]
        x = self.ranges * np.cos(angles) + relPose[0]
        y = self.ranges * np.sin(angles) + relPose[1]
        return np.column_stack([x, y])

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        # Plot the lidar scan, marking the points based on their labels
        if self.labels is not None:
            # Compute Cartesian coordinates once
            cartesian_coords = self.computeCartesian()
            
            # Define color for each label
            colors = {0: 'k.', 1: 'b.', 2: 'g.', 3: 'y.'}
            
            # Group points by label
            for label, color in colors.items():
                # Get indices of points with the current label
                indices = [i for i, lbl in enumerate(self.labels) if lbl == label]
                
                # Plot all points with the same label in one call
                ax.plot(cartesian_coords[indices, 0], cartesian_coords[indices, 1], color)
        else:
            ax.plot(self.computeCartesian()[:, 0], self.computeCartesian()[:, 1], 'k.')
        #ax.plot(self.computeCartesian()[:, 0], self.computeCartesian()[:, 1], 'k.', markersize=1)
        ax.axis('equal')
        plt.show()

    def removeClosePoints(self, minRange):
        mask = self.ranges > minRange
        self.angles = self.angles[mask]
        self.ranges = self.ranges[mask]
        if self.labels is not None:
            self.labels = self.labels[mask]
    
    def removeFarPoints(self, maxRange):
        mask = self.ranges < maxRange
        self.angles = self.angles[mask]
        self.ranges = self.ranges[mask]
        if self.labels is not None:
            self.labels = self.labels[mask]

    def orderByAngle(self):
        idx = np.argsort(self.angles)
        self.angles = self.angles[idx]
        self.ranges = self.ranges[idx]
        if self.labels is not None:
            self.labels = self.labels[idx]

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

        self.angles = np.arctan2(downsampled_points[:, 1], downsampled_points[:, 0])
        self.ranges = np.sqrt(downsampled_points[:, 0]**2 + downsampled_points[:, 1]**2)

        # Remove labels if they exist
        if self.labels is not None:
            self.labels = None

class lidarScan3D:
    def __init__(self, points3D, labels=None):
        self.points3D = points3D
        self.numReadings = len(points3D)
        self.labels = labels

    def removeGround(self, groundThreshold):
        mask = self.points3D[:, 2] > groundThreshold
        self.points3D = self.points3D[mask]
        if self.labels is not None:
            self.labels = self.labels[mask]
    
    def removeSky(self, skyThreshold):
        mask = self.points3D[:, 2] < skyThreshold
        self.points3D = self.points3D[mask]
        if self.labels is not None:
            self.labels = self.labels[mask]
    
    def splitByHeight(self, height):
        if self.labels is not None:
            bottom = lidarScan3D(self.points3D[self.points3D[:, 2] < height], self.labels[self.points3D[:, 2] < height])
            top = lidarScan3D(self.points3D[self.points3D[:, 2] >= height], self.labels[self.points3D[:, 2] >= height])
        else:
            bottom = lidarScan3D(self.points3D[self.points3D[:, 2] < height])
            top = lidarScan3D(self.points3D[self.points3D[:, 2] >= height])
        return bottom, top
    
    def removeClosePoints(self, minRange):
        mask = np.sqrt(self.points3D[:, 0]**2 + self.points3D[:, 1]**2) > minRange
        self.points3D = self.points3D[mask]
        if self.labels is not None:
            self.labels = self.labels[mask]
    
    def convertTo2D(self):
        return lidarScan(np.arctan2(self.points3D[:, 1], self.points3D[:, 0]), np.sqrt(self.points3D[:, 0]**2 + self.points3D[:, 1]**2), self.labels)
    
    def convertTo2D_new(self, angRes, maxRange):
        # This function converts the 3D scan to a 2D scan taking only the closest point in each angular sector
        # Create lidarScan object with the specified angular resolution and maximum range
        z_t = lidarScan(np.linspace(-np.pi, np.pi, angRes), np.ones(angRes)*maxRange)
        # Iterate through each point in the 3D scan
        for point in self.points3D:
            # Compute the angle and range of the point
            angle = np.arctan2(point[1], point[0])
            range = np.sqrt(point[0]**2 + point[1]**2)
            # Find the closest index in the 2D scan
            idx = np.argmin(np.abs(z_t.angles - angle))
            # Update the range if the new range is smaller
            if range < z_t.ranges[idx]:
                z_t.ranges[idx] = range
        return z_t
    
    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax = plt.axes(projection='3d')  # Add this line to create a 3D projection
        if self.labels is not None:
            # Plot the lidar scan, marking the points based on their labels
            # Define color for each label
            colors = {0: 'k.', 1: 'b.', 2: 'g.', 3: 'y.'}
            # Group points by label
            for label, color in colors.items():
                # Get indices of points with the current label
                indices = [i for i, lbl in enumerate(self.labels) if lbl == label]
                # Plot all points with the same label in one call
                ax.scatter(self.points3D[indices, 0], self.points3D[indices, 1], self.points3D[indices, 2], color)
        else:
            ax.scatter(self.points3D[:, 0], self.points3D[:, 1], self.points3D[:,2], 'r')
        ax.axis('equal')
        plt.show()

    def ROR(self, k, r):
        # This function removes outliers from the 3D scan by comparing the distance to the k-th nearest neighbor to a specified radius
        # Create a KDTree object with the 3D points
        tree = sp.spatial.KDTree(self.points3D)
        # Compute the distance to the k-th nearest neighbor for each point
        distances, _ = tree.query(self.points3D, k=k+1)
        k_distance = distances[:, k]
        # Remove points that are further than the specified radius from their k-th nearest neighbor
        self.points3D = self.points3D[k_distance < r]
        # Remove labels that correspond to removed points
        if self.labels is not None:
            self.labels = self.labels[k_distance < r]

    def SOR(self, k, s):
        # This function removes outliers from the 3D scan by comparing the distance to the k-th nearest neighbor
        # Create a KDTree object with the 3D points
        tree = sp.spatial.KDTree(self.points3D)
        # Compute the distance to the k-th nearest neighbor for each point
        distances, _ = tree.query(self.points3D, k=k+1)
        k_distance = distances[:, k]
        # Compute the mean and standard deviation of the k-th nearest neighbor distances
        mean = np.mean(k_distance)
        std = np.std(k_distance)
        # Remove points that are further than the specified number of standard deviations from the mean
        self.points3D = self.points3D[k_distance < mean + s * std]
        # Remove labels that correspond to removed points
        if self.labels is not None:
            self.labels = self.labels[k_distance < mean + s * std]

    def DROR(self, k, rho):
        # This function removes outliers from the 3D scan by comparing the distance to the k-th nearest neighbor to a radius proportional to the distance to the origin
        # Create a KDTree object with the 3D points
        tree = sp.spatial.KDTree(self.points3D)
        # Compute the distance to the k-th nearest neighbor for each point
        distances, _ = tree.query(self.points3D, k=k+1)
        k_distance = distances[:, k]
        # Compute the distance to the origin for each point
        origin_distance = np.linalg.norm(self.points3D, axis=1)
        # Remove points that are further than a radius (rho * origin_distance) from their k-th nearest neighbor
        self.points3D = self.points3D[k_distance < rho * origin_distance]
        # Remove labels that correspond to removed points
        if self.labels is not None:
            self.labels = self.labels[k_distance < rho * origin_distance]