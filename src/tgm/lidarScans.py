import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from .spatial import pose, orientation
import open3d as o3d

class lidarScan2D:
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

    def computeRelativeCartesian(self, relPose: pose):
        angles = self.angles + relPose.orientation.yaw
        x = self.ranges * np.cos(angles) + relPose.position.x
        y = self.ranges * np.sin(angles) + relPose.position.y
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

    def voxelGridFilter_o3d(self, voxel_size):
        if len(self.points3D) == 0:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.points3D))

        down_pcd = pcd.voxel_down_sample(voxel_size) # It uses centroid method(avg) instead of center of voxel or random point in voxel
        
        self.points3D = np.asarray(down_pcd.points, dtype=np.float32)

    def filterOutByLabel(self, label):
        assert self.labels is not None
        mask = self.labels != label
        angles = self.angles[mask]
        ranges = self.ranges[mask]
        labels = self.labels[mask]
        return lidarScan2D(angles, ranges, labels)

    def filterInByLabel(self, label):
        assert self.labels is not None
        mask = self.labels == label
        angles = self.angles[mask]
        ranges = self.ranges[mask]
        labels = self.labels[mask]
        return lidarScan2D(angles, ranges, labels)
    
    def convertTo3D(self, height=0.0):
        points3D = np.column_stack([self.ranges * np.cos(self.angles), self.ranges * np.sin(self.angles), np.ones(self.ranges.shape) * height])
        return lidarScan3D(points3D, self.labels)

class lidarScan3D:
    def __init__(self, points3D, labels=None):
        assert points3D.shape[1] == 3
        if labels is not None:
            assert points3D.shape[0] == labels.shape[0]
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
        return lidarScan2D(np.arctan2(self.points3D[:, 1], self.points3D[:, 0]), np.sqrt(self.points3D[:, 0]**2 + self.points3D[:, 1]**2), self.labels)
    
    def convertTo2D_new(self, angRes, maxRange):
        # This function converts the 3D scan to a 2D scan taking only the closest point in each angular sector
        # Create lidarScan object with the specified angular resolution and maximum range
        z_t = lidarScan2D(np.linspace(-np.pi, np.pi, angRes), np.ones(angRes)*maxRange)
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
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('equal')
        plt.show()
    
    def plot_o3d(self, title='Open3D_LiDAR_Scan', show_plane=False, compare_scan=None):

        def _build_geometries(scan_obj, include_plane=False):
            # 1. Create Point Cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scan_obj.points3D[:, :3])

            # 2. Handle Colors
            if hasattr(scan_obj, "colors") and scan_obj.colors is not None:
                c = scan_obj.colors
                if c.max() > 1.0:
                    c = c / 255.0
                pcd.colors = o3d.utility.Vector3dVector(c)
            else:
                pcd.paint_uniform_color([0.2, 0.5, 1.0])

            geometries_local = [pcd]

            # 3. Optional Plane
            if include_plane and hasattr(scan_obj, "_plane_normal") and hasattr(scan_obj, "_plane_d"):
                n = scan_obj._plane_normal
                d = scan_obj._plane_d
                x_min, x_max = np.percentile(scan_obj.points3D[:, 0], [2, 98])
                y_min, y_max = np.percentile(scan_obj.points3D[:, 1], [2, 98])

                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 2),
                    np.linspace(y_min, y_max, 2)
                )
                zz = (-n[0] * xx - n[1] * yy - d) / (n[2] + 1e-12)

                plane_vertices = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=1)
                triangles = [[0, 2, 1], [1, 2, 3]]

                plane_mesh = o3d.geometry.TriangleMesh()
                plane_mesh.vertices = o3d.utility.Vector3dVector(plane_vertices)
                plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
                plane_mesh.paint_uniform_color([0.2, 1.0, 0.2])
                geometries_local.append(plane_mesh)

            # 4. Add Axes
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            geometries_local.append(axes)
            return geometries_local

        # Single-view mode (original behavior)
        if compare_scan is None:
            geometries = _build_geometries(self, include_plane=show_plane)

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=title, width=1024, height=768)

            for geom in geometries:
                vis.add_geometry(geom)

            opt = vis.get_render_option()
            opt.background_color = np.asarray([1, 1, 1])
            opt.point_size = 2.0

            vis.run()
            vis.destroy_window()
            return

        # Compare mode: two Open3D views side-by-side (same visualizer style)
        left_geometries = _build_geometries(self, include_plane=show_plane)
        right_geometries = _build_geometries(compare_scan, include_plane=False)

        win_w = 900
        win_h = 768
        left_pos = 40
        top_pos = 60
        right_pos = left_pos + win_w + 20

        vis_left = o3d.visualization.Visualizer()
        vis_left.create_window(window_name=f"{title} | Ground", width=win_w, height=win_h, left=left_pos, top=top_pos)
        for geom in left_geometries:
            vis_left.add_geometry(geom)

        vis_right = o3d.visualization.Visualizer()
        vis_right.create_window(window_name=f"{title} | Original", width=win_w, height=win_h, left=right_pos, top=top_pos)
        for geom in right_geometries:
            vis_right.add_geometry(geom)

        opt_left = vis_left.get_render_option()
        opt_left.background_color = np.asarray([1, 1, 1])
        opt_left.point_size = 2.0

        opt_right = vis_right.get_render_option()
        opt_right.background_color = np.asarray([1, 1, 1])
        opt_right.point_size = 2.0

        while True:
            alive_left = vis_left.poll_events()
            if alive_left:
                vis_left.update_renderer()

            alive_right = vis_right.poll_events()
            if alive_right:
                vis_right.update_renderer()

            if (not alive_left) or (not alive_right):
                break

        vis_left.destroy_window()
        vis_right.destroy_window()

    def BEV_GroundSeg(self, grid_res=0.5, height_thresh=0.2,
                        smooth_kernel=5, abs_height_limit=1.5):

        points_numpy = self.points3D
        if points_numpy.size == 0:
            return lidarScan3D(np.empty((0, 3))), lidarScan3D(np.empty((0, 3)))


        min_x = float(np.min(points_numpy[:, 0]))
        max_x = float(np.max(points_numpy[:, 0]))
        min_y = float(np.min(points_numpy[:, 1]))
        max_y = float(np.max(points_numpy[:, 1]))
        span_pad = float(grid_res)
        min_x -= span_pad
        max_x += span_pad
        min_y -= span_pad
        max_y += span_pad
        pts = points_numpy
        outside_pts = np.empty((0, 3), dtype=points_numpy.dtype)

        if pts.shape[0] == 0:
            return lidarScan3D(np.empty((0, 3))), lidarScan3D(points_numpy.copy())

        grid_w = int(np.ceil((max_x - min_x) / grid_res)) + 1
        grid_h = int(np.ceil((max_y - min_y) / grid_res)) + 1

        idx_x = np.clip(((pts[:, 0] - min_x) / grid_res).astype(np.int64), 0, grid_w - 1)
        idx_y = np.clip(((pts[:, 1] - min_y) / grid_res).astype(np.int64), 0, grid_h - 1)
        flat_idx = idx_x * grid_h + idx_y

        # Get min Z per cell (empty cells stay inf)
        min_z = np.full((grid_w * grid_h,), np.inf, dtype=np.float32)
        np.minimum.at(min_z, flat_idx, pts[:, 2].astype(np.float32))

        min_z_2d = min_z.reshape(grid_w, grid_h)
        pad = smooth_kernel // 2

        # Fill empty cells (Morphological Dilation)
        filled = sp.ndimage.minimum_filter(
            min_z_2d,
            size=(smooth_kernel * 2 + 1, smooth_kernel * 2 + 1),
            mode='constant',
            cval=np.inf
        )
        min_z_2d = np.where(np.isinf(min_z_2d), filled, min_z_2d)

        # Erosion (Min-Pool): Pulls curbs and walls down to road level locally
        eroded_z_2d = sp.ndimage.minimum_filter(
            min_z_2d,
            size=(3, 3),
            mode='constant',
            cval=np.inf
        )

        # Smooth the surface (Avg-Pool)
        valid_mask = np.isfinite(eroded_z_2d).astype(np.float32)
        finite_values = np.where(np.isfinite(eroded_z_2d), eroded_z_2d, 0.0).astype(np.float32)
        window_area = float(smooth_kernel * smooth_kernel)
        sum_z = sp.ndimage.uniform_filter(
            finite_values,
            size=(smooth_kernel, smooth_kernel),
            mode='constant',
            cval=0.0
        ) * window_area
        count_z = sp.ndimage.uniform_filter(
            valid_mask,
            size=(smooth_kernel, smooth_kernel),
            mode='constant',
            cval=0.0
        ) * window_area
        smoothed_z_2d = np.divide(
            sum_z,
            count_z,
            out=np.full_like(sum_z, np.nan, dtype=np.float32),
            where=count_z > 0
        )

        smoothed_z = smoothed_z_2d.reshape(-1)
        point_sloped_z = smoothed_z[flat_idx]

        ground_mask = (pts[:, 2] - point_sloped_z <= height_thresh) & \
                    (pts[:, 2] < abs_height_limit)

        nonground_pts = np.concatenate([pts[~ground_mask], outside_pts], axis=0)
        return lidarScan3D(pts[ground_mask].copy()), lidarScan3D(nonground_pts.copy())

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
    
    
    def voxelGridFilter(self, voxel_size):
        # Determine the grid indices for each point
        grid_indices = np.floor(self.points3D / voxel_size).astype(int)

        # Create a dictionary to store points in each voxel
        voxel_dict = {}
        for i, idx in enumerate(grid_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(self.points3D[i])

        # Create a list to store the downsampled points
        downsampled_points = []

        # Iterate through each voxel and average the points inside
        for key, points in voxel_dict.items():
            average_point = np.mean(points, axis=0)
            downsampled_points.append(average_point)
        
        downsampled_points = np.array(downsampled_points)
        self.points3D = downsampled_points

        # Remove labels if they exist
        if self.labels is not None:
            self.labels = None
        
    def transform(self, x_t: pose) -> 'lidarScan3D':
        if self.points3D is None or self.points3D.size == 0:
            empty_labels = None if self.labels is None else self.labels[:0]
            return lidarScan3D(np.empty((0, 3)), empty_labels)
        
        R = orientation.to_R_zyx(x_t.orientation)          # Rz(yaw) @ Ry(pitch) @ Rx(roll)
        t = np.array([x_t.position.x, x_t.position.y, x_t.position.z])
        return lidarScan3D(self.points3D @ R.T + t, self.labels)
    
    def __matmul__(self, other):
        # This function allows to use the @ operator to transform the scan by a pose
        if not isinstance(other, pose):
            raise ValueError("The @ operator can only be used to transform the scan by a pose")
        return self.transform(other)

if __name__ == "__main__":
    # Create a 3D lidar scan with only one point
    points3D = np.array([[1, 2, 3]])
    labels = np.array([0])
    z_t_3D = lidarScan3D(points3D, labels)