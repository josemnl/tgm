import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

class lidarScan:
    def __init__(self, angles, ranges, labels=None):
        assert len(angles) == len(ranges)
        if labels is not None:
            assert len(angles) == len(labels)
            assert angles.shape == ranges.shape == labels.shape
        
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

    def plot(self, ax=None, byLabel=False):
        if ax is None:
            ax = plt.gca()
        # Plot the lidar scan, marking the points based on their labels
        if self.labels is not None and byLabel:
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

    def fastVoxelGridFilter(self, voxel_size):
        # Compute a voxel grid without averaging the points, so that the dictionary is not needed
        points = self.computeCartesian()
        grid_indices = np.floor(points / voxel_size).astype(int)

        # remove duplicates
        unique_indices = np.unique(grid_indices, axis=0)

        # From indices to points
        downsampled_points = unique_indices * voxel_size

        self.angles = np.arctan2(downsampled_points[:, 1], downsampled_points[:, 0])
        self.ranges = np.sqrt(downsampled_points[:, 0]**2 + downsampled_points[:, 1]**2)

        # Remove labels if they exist
        if self.labels is not None:
            self.labels = None

    def filterOutByLabel(self, label):
        assert self.labels is not None
        mask = self.labels != label
        angles = self.angles[mask]
        ranges = self.ranges[mask]
        labels = self.labels[mask]
        return lidarScan(angles, ranges, labels)

    def filterInByLabel(self, label):
        assert self.labels is not None
        mask = self.labels == label
        angles = self.angles[mask]
        ranges = self.ranges[mask]
        labels = self.labels[mask]
        return lidarScan(angles, ranges, labels)

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

    def removePointsInBox(self, box):
        x_min = box[0]
        y_min = box[1]
        mask = (np.abs(self.points3D[:, 0]) > x_min) | (np.abs(self.points3D[:, 1]) > y_min)
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
    
    def removeFarPoints(self, maxRange):
        mask = np.sqrt(self.points3D[:, 0]**2 + self.points3D[:, 1]**2) < maxRange
        self.points3D = self.points3D[mask]
        if self.labels is not None:
            self.labels = self.labels[mask]

    def filterOutByLabel(self, label):
        assert self.labels is not None
        mask = self.labels != label
        points3D = self.points3D[mask]
        labels = self.labels[mask]
        return lidarScan3D(points3D, labels)
    
    def filterInByLabel(self, label):
        assert self.labels is not None
        mask = self.labels == label
        points3D = self.points3D[mask]
        labels = self.labels[mask]
        return lidarScan3D(points3D, labels)
    
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
        #ax.axis('equal')
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

    def DSOR(self, k, s, rho):
        # This function removes outliers from the 3D scan by comparing the distance to the k-th nearest neighbor to a radius proportional to the distance to the origin
        # Create a KDTree object with the 3D points
        tree = sp.spatial.KDTree(self.points3D)
        # Compute the distance to the k-th nearest neighbor for each point
        distances, _ = tree.query(self.points3D, k=k+1)
        k_distance = distances[:, k]
        # Compute the mean and standard deviation of the k-th nearest neighbor distances
        mean = np.mean(k_distance)
        std = np.std(k_distance)
        # Compute the distance to the origin for each point
        origin_distance = np.linalg.norm(self.points3D, axis=1)
        # Remove points that are further than a radius (mean + s * std) * rho * origin_distance from their k-th nearest neighbor
        self.points3D = self.points3D[k_distance < (mean + s * std) * rho * origin_distance]
        # Remove labels that correspond to removed points
        if self.labels is not None:
            self.labels = self.labels[k_distance < (mean + s * std) * rho * origin_distance]

    def translate(self, translation):
        # This function translates the 3D points by a specified translation
        self.points3D += translation

    def rotate(self, rotationMatrix):
        # This function rotates the 3D points by a specified rotation matrix
        self.points3D = np.dot(rotationMatrix, self.points3D.T).T

    def RANSAC(self, maxDistance, maxIterations):
        # This function performs RANSAC on the 3D points to find the best plane
        bestInliers = []
        bestPlane = None
        bestError = np.inf
        for _ in range(maxIterations):
            # Randomly sample three points
            indices = np.random.choice(len(self.points3D), 3, replace=False)
            points = self.points3D[indices]
            # Compute the plane parameters
            v1 = points[1] - points[0]
            v2 = points[2] - points[0]
            normal = np.cross(v1, v2)
            normal /= np.linalg.norm(normal)
            d = -np.dot(normal, points[0])
            # Compute the distance to the plane for each point
            distances = np.abs(np.dot(self.points3D, normal) + d)
            # Compute the inliers
            inliers = np.where(distances < maxDistance)[0]
            # Update the best model if the current model is better
            error = np.sum(distances[inliers])/len(inliers)
            if error < bestError:
                bestInliers = inliers
                bestPlane = (normal, d)
                bestError = error
        # Force the normal to point upwards
        if bestPlane[0][2] < 0:
            bestPlane = (-bestPlane[0], -bestPlane[1])
        # Move the plane up by maxDistance
        bestPlane = (bestPlane[0], bestPlane[1] - maxDistance)
        # Ground are the points below the plane
        groundMask = np.where(np.dot(self.points3D, bestPlane[0]) + bestPlane[1] < 0)[0]
        ground = lidarScan3D(self.points3D[groundMask])
        objectsMask = np.where(np.dot(self.points3D, bestPlane[0]) + bestPlane[1] >= 0)[0]
        objects = lidarScan3D(self.points3D[objectsMask])
        return ground, objects

if __name__ == "__main__":
    # Create a 3D lidar scan with only one point
    points3D = np.array([[1, 2, 3]])
    labels = np.array([0])
    z_t_3D = lidarScan3D(points3D, labels)