import numpy as np
import cupy as cp
import scipy.signal as sp
import cupyx.scipy.signal as csp
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from skimage.morphology import disk
import matplotlib

from gridMap import discreteDist, gridMap
from spatial import frame, origin, size, pose, position, orientation

matplotlib.use('Qt5Agg')

class TGM:
    def __init__(self, tgmFrame: frame, priors: list, maxVelocity: int, saturationLimits: list, fftConv=False, isGPU=True):
        assert isinstance(tgmFrame, frame)
        assert isinstance(priors[0], (float, int))
        assert isinstance(priors[1], (float, int))
        assert isinstance(priors[2], (float, int))
        assert isinstance(maxVelocity, int)
        assert isinstance(saturationLimits, list) and len(saturationLimits) == 4

        # Choose numpy or cupy
        self.GPU = isGPU
        if self.GPU:
            self.xp = cp
        else:
            self.xp = np

        self.frame = tgmFrame
        self.staticPrior = priors[0]
        self.dynamicPrior = priors[1]
        self.weatherPrior = priors[2]
        self.freePrior = 1 - priors[0] - priors[1] - priors[2]
        self.sdwPrior = self.staticPrior + self.dynamicPrior + self.weatherPrior
        self.staticMap = gridMap(self.frame, self.xp.ones((self.frame.size.w, self.frame.size.h, self.frame.size.d)) * priors[0])
        self.dynamicMap = gridMap(self.frame, self.xp.ones((self.frame.size.w, self.frame.size.h, self.frame.size.d)) * priors[1])
        self.weatherMap = gridMap(self.frame, self.xp.ones((self.frame.size.w, self.frame.size.h, self.frame.size.d)) * priors[2])

        # Conv shape 2D
        shape2D = disk(maxVelocity).astype(float)
        self.D0_2D = 1 / np.sum(shape2D)
        shape2D /= np.sum(shape2D)
        shape2D[len(shape2D)//2, len(shape2D)//2] = 0
        self.convShape2D = shape2D
        self.convShape2D = self.xp.asarray(self.convShape2D) # Ensure it's a cupy array if using GPU

        # Conv shape 3D
        # (For now, it's just a cuboid)
        shape3D = np.ones((2*maxVelocity+1, 2*maxVelocity+1, 2*maxVelocity+1))
        self.D0_3D = 1 / np.sum(shape3D)
        shape3D /= np.sum(shape3D)
        shape3D[maxVelocity, maxVelocity, maxVelocity] = 0
        self.convShape3D = shape3D
        self.convShape3D = self.xp.asarray(self.convShape3D) # Ensure it's a cupy array if using GPU

        self.satLowS = saturationLimits[0]
        self.satHighS = saturationLimits[1]
        self.satLowD = saturationLimits[2]
        self.satHighD = saturationLimits[3]
        self.fftConv = fftConv

        self.x_t = None
        self.prev_region = [0, 0, 0, 0, 0, 0]

    @property
    def freeMap(self) -> gridMap:
        return gridMap(self.frame, 1 - self.staticMap.data - self.dynamicMap.data - self.weatherMap.data)

    @property
    def is2D(self) -> bool:
        return self.frame.is2D
    
    @property
    def is3D(self) -> bool:
        return self.frame.is3D
    
    @property
    def dynamicCardinality(self) -> discreteDist:
        return self.dynamicMap.cardinality()

    def update(self, instGridMap: gridMap, x_t: pose | None):
        assert isinstance(instGridMap, gridMap)
        assert instGridMap.frame.r == self.frame.r
        assert isinstance(x_t, pose) or x_t is None

        # Update ego position (used for visualization purposes only)
        self.x_t = x_t

        # Compute overlaping grid between the instantaneous map and the TGM
        overlap = self.staticMap.computeOverlap(instGridMap.frame)

        # Crop the instantaneous map to the overlapping region
        instMap = instGridMap.crop(overlap).data

        if self.GPU:
            instMap = cp.asarray(instMap)

        # Split the instantaneous map into static, dynamic, weather and free maps
        instStaticMap = instMap * self.staticPrior / self.sdwPrior
        instDynamicMap = instMap * self.dynamicPrior / self.sdwPrior
        instWeatherMap = instMap * self.weatherPrior / self.sdwPrior
        instFreeMap = 1 - instStaticMap - instDynamicMap - instWeatherMap

        # Predict based on previous measurements
        predStaticMap, predDynamicMap, predWeatherMap = self.predict(overlap)
        predFreeMap = 1 - predStaticMap - predDynamicMap - predWeatherMap

        # Compute the updated maps
        if self.staticPrior != 0:
            staticMatrix = instStaticMap * predStaticMap / self.staticPrior
        else:
            staticMatrix = self.xp.zeros_like(instStaticMap)
        if self.dynamicPrior != 0:
            dynamicMatrix = instDynamicMap * predDynamicMap / self.dynamicPrior
        else:
            dynamicMatrix = self.xp.zeros_like(instDynamicMap)
        if self.weatherPrior != 0:
            weatherMatrix = instWeatherMap * predWeatherMap / self.weatherPrior
        else:
            weatherMatrix = self.xp.zeros_like(instWeatherMap)
        freeMatrix = instFreeMap * predFreeMap / self.freePrior

        # Normalize the maps
        total = staticMatrix + dynamicMatrix + weatherMatrix + freeMatrix
        staticMatrix /= total
        dynamicMatrix /= total
        weatherMatrix /= total

        # Apply saturation limits
        staticMatrix = self.xp.clip(staticMatrix, self.satLowS, self.satHighS)
        dynamicMatrix = self.xp.clip(dynamicMatrix, self.satLowD, self.satHighD)

        # Set the cells that were visible to the prior
        x0, y0, x1, y1, z0, z1 = self.prev_region
        self.dynamicMap.data[x0:x1, y0:y1, z0:z1] = (1 - self.staticMap.data[x0:x1, y0:y1, z0:z1]) * self.dynamicPrior / (self.dynamicPrior + self.freePrior + self.weatherPrior)

        # Compute visible mask as the portion of the TGM that overlaps with the instantaneous map
        x0_new = overlap.origin.x - self.frame.origin.x
        y0_new = overlap.origin.y - self.frame.origin.y
        z0_new = overlap.origin.z - self.frame.origin.z
        x1_new = x0_new + overlap.size.w
        y1_new = y0_new + overlap.size.h
        z1_new = z0_new + overlap.size.d

        # Save the visible cells
        self.staticMap.data[x0_new:x1_new, y0_new:y1_new, z0_new:z1_new] = staticMatrix
        self.dynamicMap.data[x0_new:x1_new, y0_new:y1_new, z0_new:z1_new] = dynamicMatrix
        self.weatherMap.data[x0_new:x1_new, y0_new:y1_new, z0_new:z1_new] = weatherMatrix

        # Save the previous visible mask
        self.prev_region = [x0_new, y0_new, x1_new, y1_new, z0_new, z1_new]

    def predict(self, predictFrame=None):
        # Crop the maps if necessary
        if predictFrame is None:
            staticMap = self.staticMap.data
            dynamicMap = self.dynamicMap.data
        else:
            staticMap = self.staticMap.crop(predictFrame).data
            dynamicMap = self.dynamicMap.crop(predictFrame).data

        # Compute static prediction
        predStaticMap = staticMap

        # Compute dynamic prediction
        if self.dynamicPrior != 0:
            if self.is3D:
                print("3D convolution")
                dynamicStay = dynamicMap * self.D0_3D
                bounceBack = conv3prior(staticMap, self.convShape3D, self.staticPrior, self.fftConv, self.GPU) * dynamicMap
                dynamicMove = conv3prior(dynamicMap, self.convShape3D, self.dynamicPrior, self.fftConv, self.GPU) * (1 - staticMap)
            else:
                print("2D convolution")
                dynamicStay = dynamicMap * self.D0_2D
                bounceBack = conv2prior(staticMap, self.convShape2D, self.staticPrior, self.fftConv, self.GPU) * dynamicMap
                dynamicMove = conv2prior(dynamicMap, self.convShape2D, self.dynamicPrior, self.fftConv, self.GPU) * (1 - staticMap)
            predDynamicMap = dynamicStay + bounceBack + dynamicMove
        else:
            predDynamicMap = cp.zeros_like(dynamicMap) if self.GPU else np.zeros_like(dynamicMap)

        # Compute weather prediction
        predWeatherMap = (1 - predStaticMap - predDynamicMap) * self.weatherPrior / (self.weatherPrior + self.freePrior)

        return predStaticMap, predDynamicMap, predWeatherMap
    
    def contains(self, otherFrame: frame):
        assert otherFrame.r == self.frame.r
        return self.staticMap.contains(otherFrame)
    
    def reshape(self, newFrame: frame):
        '''
        Update the origin and size of the TGM, reshaping the maps and updating the previous region.
        '''
        self.prev_region[0] = self.prev_region[0] + self.frame.origin.x - newFrame.origin.x
        self.prev_region[1] = self.prev_region[1] + self.frame.origin.y - newFrame.origin.y
        self.prev_region[2] = self.prev_region[2] + self.frame.origin.x - newFrame.origin.x
        self.prev_region[3] = self.prev_region[3] + self.frame.origin.y - newFrame.origin.y
        self.prev_region[4] = self.prev_region[4] + self.frame.origin.z - newFrame.origin.z
        self.prev_region[5] = self.prev_region[5] + self.frame.origin.z - newFrame.origin.z

        self.staticMap = self.staticMap.reshape(newFrame, self.staticPrior)
        self.dynamicMap = self.dynamicMap.reshape(newFrame, self.dynamicPrior)
        self.weatherMap = self.weatherMap.reshape(newFrame, self.weatherPrior)

        self.frame = newFrame

        # Make sure the previous region is within the new map
        self.prev_region[0] = max(0, self.prev_region[0])
        self.prev_region[1] = max(0, self.prev_region[1])
        self.prev_region[2] = min(newFrame.size.w, self.prev_region[2])
        self.prev_region[3] = min(newFrame.size.h, self.prev_region[3])

    def oneLayer(self, layer, layerFrame):
        overlap = self.frame.computeOverlap(layerFrame)
        return self._get_layer_map(layer).crop(overlap)
    
    def maxLayer(self, layer, layerFrame = None):
        '''
        Return a map with ones in the cells where probability of layer is bigger than probability of all the others.
        '''
        layers = ['static', 'dynamic', 'weather']
        layers.remove(layer)
        if layerFrame is None:
            layerFrame = self.frame
        return gridMap(self.frame,
                       (self._get_layer_map(layer).data > self._get_layer_map(layers[0]).data) &
                       (self._get_layer_map(layer).data > self._get_layer_map(layers[1]).data) &
                       (self._get_layer_map(layer).data > self.freeMap.data)).crop(layerFrame)

    def computeStaticDynamicGridMap(self):
        combined_data = self.staticMap.data + self.dynamicMap.data
        return gridMap(self.frame, combined_data)
    
    def dynamicRebalance(self, targetDynamicCardinality: discreteDist):
        """
        Rebalance the dynamic map to match the target dynamic cardinality distribution.
        This is done without changing the static map and weather map, by adjusting only the ratio
        between dynamic and free space in each cell. I.e., we adjust p(dynamic) / (p(dynamic) + p(free)).
        Uses an iterative approach to adjust the value of the shift applied to the log-odds.
        """
        shift_low = -10.0
        shift_high = 10.0
        tolerance = 1e-3
        max_iterations = 20

        expected_target = targetDynamicCardinality.expected_value()
        
        dynamic_map = self.dynamicMap.data
        free_map = self.freeMap.data
        
        for _ in range(max_iterations):
            shift = (shift_low + shift_high) / 2.0
            denom = dynamic_map + free_map
            if self.GPU:
                denom = cp.where(denom <= 0, 1e-12, denom)
            else:
                denom = np.where(denom <= 0, 1e-12, denom)
            dynamic_free_ratio_map = gridMap(self.frame, dynamic_map / denom)
            new_dynamic_free_ratio_map = dynamic_free_ratio_map.logOddShift(shift).data
            new_dynamic_map = new_dynamic_free_ratio_map * (dynamic_map + free_map)

            # Compute the cardinality distribution of the new dynamic map
            new_dynamic_map = gridMap(self.dynamicMap.frame, new_dynamic_map)
            expected_dynamic = new_dynamic_map.cardinality().expected_value()

            if abs(expected_dynamic - expected_target) < tolerance:
                break
            elif expected_dynamic < expected_target:
                shift_low = shift
            else:
                shift_high = shift

        # Update the dynamic map
        self.dynamicMap = new_dynamic_map

    
    def plot(self, fig=None, frame = None, saveMap=False, savePNG=False, saveSvg=False, imgName='', style='combined', egoStyle='rectangle'):
        assert style in ['combined', 'static', 'dynamic', 'weather']
        assert egoStyle in ['none', 'dot', 'rectangle']
        if frame is None:
            frame = self.frame
        if fig is None:
            fig = plt.figure()
        overlap = self.frame.computeOverlap(frame)
        staticMap = self.staticMap.crop(overlap).toCPU().data
        dynamicMap = self.dynamicMap.crop(overlap).toCPU().data
        weatherMap = self.weatherMap.crop(overlap).toCPU().data

        # Plot the map according to the style
        if style == 'combined':
            I = np.zeros((overlap.size.h, overlap.size.w, 3))
            I[:,:,0] = 1 - np.transpose(1.0*staticMap + 0.0*dynamicMap + 2.0*weatherMap/np.square(1-weatherMap))
            I[:,:,1] = 1 - np.transpose(0.5*staticMap + 0.5*dynamicMap + 0.0*weatherMap/np.square(1-weatherMap))
            I[:,:,2] = 1 - np.transpose(0.0*staticMap + 1.0*dynamicMap + 2.0*weatherMap/np.square(1-weatherMap))
            # Make sure the values are between 0 and 1
            I = np.clip(I, 0, 1)
        elif style == 'static':
            I = 1 - np.transpose(staticMap)
        elif style == 'dynamic':
            I = 1 - np.transpose(dynamicMap)
        elif style == 'weather':
            I = 1 - np.transpose(weatherMap)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                extent=(overlap.origin.x*self.frame.r, (overlap.origin.x + overlap.size.w)*self.frame.r,
                        overlap.origin.y*self.frame.r, (overlap.origin.y + overlap.size.h)*self.frame.r))

        # Plot the ego pose
        if self.x_t is not None:
            if egoStyle == 'dot':
                plt.plot(self.x_t.position.x, self.x_t.position.y, 'ro')
            elif egoStyle == 'rectangle':
                x = self.x_t.position.x
                y = self.x_t.position.y
                theta = self.x_t.orientation.yaw
                car_length = 4.953
                car_width = 1.923
                x1 = x + car_length/2 * np.cos(theta) + car_width/2 * np.cos(theta + np.pi/2)
                y1 = y + car_length/2 * np.sin(theta) + car_width/2 * np.sin(theta + np.pi/2)
                x2 = x + car_length/2 * np.cos(theta) - car_width/2 * np.cos(theta + np.pi/2)
                y2 = y + car_length/2 * np.sin(theta) - car_width/2 * np.sin(theta + np.pi/2)
                x3 = x - car_length/2 * np.cos(theta) - car_width/2 * np.cos(theta + np.pi/2)
                y3 = y - car_length/2 * np.sin(theta) - car_width/2 * np.sin(theta + np.pi/2)
                x4 = x - car_length/2 * np.cos(theta) + car_width/2 * np.cos(theta + np.pi/2)
                y4 = y - car_length/2 * np.sin(theta) + car_width/2 * np.sin(theta + np.pi/2)
                rectangle = plt.Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], closed=True, facecolor='white', edgecolor='black')
                plt.gca().add_patch(rectangle)
                # Plot the heading as a triangle
                x1 = x + car_length/2 * np.cos(theta)
                y1 = y + car_length/2 * np.sin(theta)
                x2 = x + (car_length/2-car_width) * np.cos(theta) + car_width/2 * np.sin(theta)
                y2 = y + (car_length/2-car_width) * np.sin(theta) - car_width/2 * np.cos(theta)
                x3 = x + (car_length/2-car_width) * np.cos(theta) - car_width/2 * np.sin(theta)
                y3 = y + (car_length/2-car_width) * np.sin(theta) + car_width/2 * np.cos(theta)
                triangle = plt.Polygon([[x1, y1], [x2, y2], [x3, y3]], closed=True, facecolor='white', edgecolor='black')
                plt.gca().add_patch(triangle)

        if saveMap:
            imsave(imgName + '_map.png', I, origin ="lower", cmap='gray')
        if savePNG:
            plt.savefig(imgName + '.png', format='png')
        if saveSvg:
            plt.savefig(imgName + '.svg', format='svg', dpi=1200)
        
        # Pause to show the image
        plt.pause(0.01)

    def plot3D(self, fig: plt.Figure = None, frame: frame = None, isPause=False, value_min: float = 0.0, value_max: float = 1.0) -> None:
        """
        3D plot of the TGM using scatter plot.
        Very similar to the one in gridMap.py, but plotting the 3 layers each using
        a different RGB layer for the color, similarly to the 2D case.
        """
        if frame is None:
            frame = self.frame
        if fig is None:
            fig = plt.figure()
        overlap = self.frame.computeOverlap(frame)
        staticMap = self.staticMap.crop(overlap).toCPU().data
        dynamicMap = self.dynamicMap.crop(overlap).toCPU().data
        weatherMap = self.weatherMap.crop(overlap).toCPU().data

        # Create a 3D axis
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid for the coordinates
        x = np.arange(overlap.origin.x, overlap.origin.x + overlap.size.w) * self.frame.r
        y = np.arange(overlap.origin.y, overlap.origin.y + overlap.size.h) * self.frame.r
        z = np.arange(overlap.origin.z, overlap.origin.z + overlap.size.d) * self.frame.r
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Flatten the arrays for plotting
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        staticMap = staticMap.flatten()
        dynamicMap = dynamicMap.flatten()
        weatherMap = weatherMap.flatten()

        # Create a color array based on the probabilities
        colors = np.zeros((len(X), 3))
        # Same color coding as in the 2D case
        colors[:, 0] = 1 - (staticMap + 0.0*dynamicMap + 2.0*weatherMap/np.square(1-weatherMap))  # Red channel
        colors[:, 1] = 1 - (0.5*staticMap + 0.5*dynamicMap + 0.0*weatherMap/np.square(1-weatherMap))  # Green channel
        colors[:, 2] = 1 - (0.0*staticMap + 1.0*dynamicMap + 2.0*weatherMap/np.square(1-weatherMap))  # Blue channel

        # Normalize colors to be between 0 and 1
        colors = np.clip(colors, 0, 1)

        # Mask out low and high values
        mask = (staticMap + dynamicMap + weatherMap > value_min) & (staticMap + dynamicMap + weatherMap < value_max)
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        colors = colors[mask]

        # Before plotting, enforce equal data scale across X/Y/Z using the true extents (in meters)
        # Compute extents from the overlap frame (not from masked points)
        x_min = overlap.origin.x * self.frame.r
        x_max = (overlap.origin.x + overlap.size.w) * self.frame.r
        y_min = overlap.origin.y * self.frame.r
        y_max = (overlap.origin.y + overlap.size.h) * self.frame.r
        z_min = overlap.origin.z * self.frame.r
        z_max = (overlap.origin.z + overlap.size.d) * self.frame.r

        rx = max(x_max - x_min, 0.0)
        ry = max(y_max - y_min, 0.0)
        rz = max(z_max - z_min, 0.0)

        # Set explicit limits first (so autoscale doesn't change box-aspect)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Try to preserve the real aspect ratios (so short Z is not stretched)
        ax.set_box_aspect((rx, ry, rz))

        # Scatter plot
        ax.scatter(X, Y, Z, c=colors, marker='o', s=1)

        # Set labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # Plot the ego pose
        if self.x_t is not None:
            ax.scatter(self.x_t.position.x, self.x_t.position.y, self.x_t.position.z, c='red', marker='o', s=50)

        # Plot field of view of the sensor (90 degrees horizontal, 40 degrees vertical and 15 m range)
        # The plot displays the sides of the pyramid
        if self.x_t is not None:
            sensor_x = self.x_t.position.x
            sensor_y = self.x_t.position.y
            sensor_z = self.x_t.position.z
            sensor_roll = self.x_t.orientation.roll
            sensor_pitch = self.x_t.orientation.pitch
            sensor_yaw = self.x_t.orientation.yaw

            fov_range = 3.0
            fov_hfov = np.deg2rad(45.0)  # Half horizontal FOV
            fov_vfov = np.deg2rad(20.0)  # Half vertical FOV

            # Rotation matrix from sensor to world frame
            R_yaw = np.array([[np.cos(sensor_yaw), -np.sin(sensor_yaw), 0],
                              [np.sin(sensor_yaw), np.cos(sensor_yaw), 0],
                              [0, 0, 1]])
            R_pitch = np.array([[np.cos(sensor_pitch), 0, np.sin(sensor_pitch)],
                                [0, 1, 0],
                                [-np.sin(sensor_pitch), 0, np.cos(sensor_pitch)]])

            R_roll = np.array([[1, 0, 0],
                               [0, np.cos(sensor_roll), -np.sin(sensor_roll)],
                               [0, np.sin(sensor_roll), np.cos(sensor_roll)]])
            R = R_yaw @ R_pitch @ R_roll

            # Define the 4 corner rays in the sensor frame (yaw/pitch offsets)
            # Order: top-left, top-right, bottom-right, bottom-left
            angles = [(-fov_hfov,  fov_vfov),
                      ( fov_hfov,  fov_vfov),
                      ( fov_hfov, -fov_vfov),
                      (-fov_hfov, -fov_vfov)]
            corners_sensor = []
            for yaw_off, pitch_off in angles:
                cp = np.cos(pitch_off)
                dir_sensor = np.array([cp * np.cos(yaw_off), cp * np.sin(yaw_off), np.sin(pitch_off)])
                corners_sensor.append(fov_range * dir_sensor)
            corners = np.vstack(corners_sensor)

            # Rotate and translate corners to world frame
            world_corners = (R @ corners.T).T + np.array([sensor_x, sensor_y, sensor_z])

            # Plot the 4 sides of the pyramid
            for i in range(4):
                x_vals = [sensor_x, world_corners[i, 0]]
                y_vals = [sensor_y, world_corners[i, 1]]
                z_vals = [sensor_z, world_corners[i, 2]]
                ax.plot(x_vals, y_vals, z_vals, color='blue', linestyle='--', linewidth=1)
            
            # Plot the base of the pyramid
            for i in range(4):
                x_vals = [world_corners[i, 0], world_corners[(i+1)%4, 0]]
                y_vals = [world_corners[i, 1], world_corners[(i+1)%4, 1]]
                z_vals = [world_corners[i, 2], world_corners[(i+1)%4, 2]]
                ax.plot(x_vals, y_vals, z_vals, color='blue', linestyle='--', linewidth=1)

        # Plot the coordinate frame of the sensor at the ego position
        if self.x_t is not None:
            sensor_x = self.x_t.position.x
            sensor_y = self.x_t.position.y
            sensor_z = self.x_t.position.z
            sensor_roll = self.x_t.orientation.roll
            sensor_pitch = self.x_t.orientation.pitch
            sensor_yaw = self.x_t.orientation.yaw

            # Rotation matrix from sensor to world frame
            R_yaw = np.array([[np.cos(sensor_yaw), -np.sin(sensor_yaw), 0],
                              [np.sin(sensor_yaw), np.cos(sensor_yaw), 0],
                              [0, 0, 1]])
            R_pitch = np.array([[np.cos(sensor_pitch), 0, np.sin(sensor_pitch)],
                                [0, 1, 0],
                                [-np.sin(sensor_pitch), 0, np.cos(sensor_pitch)]])

            R_roll = np.array([[1, 0, 0],
                               [0, np.cos(sensor_roll), -np.sin(sensor_roll)],
                               [0, np.sin(sensor_roll), np.cos(sensor_roll)]])
            R = R_yaw @ R_pitch @ R_roll

            # Define the axes in the sensor frame
            axis_length = 1.0
            axes_sensor = np.array([[axis_length, 0, 0],
                                    [0, axis_length, 0],
                                    [0, 0, axis_length]])
            axes_world = (R @ axes_sensor.T).T + np.array([sensor_x, sensor_y, sensor_z])

            # Plot the axes
            ax.plot([sensor_x, axes_world[0, 0]], [sensor_y, axes_world[0, 1]], [sensor_z, axes_world[0, 2]], color='red', linewidth=2)   # X-axis
            ax.plot([sensor_x, axes_world[1, 0]], [sensor_y, axes_world[1, 1]], [sensor_z, axes_world[1, 2]], color='green', linewidth=2) # Y-axis
            ax.plot([sensor_x, axes_world[2, 0]], [sensor_y, axes_world[2, 1]], [sensor_z, axes_world[2, 2]], color='blue', linewidth=2)  # Z-axis
            
        plt.show(block=isPause)
        plt.pause(0.01)

    def _get_layer_map(self, layer):
        if layer == 'static':
            return self.staticMap
        elif layer == 'dynamic':
            return self.dynamicMap
        elif layer == 'weather':
            return self.weatherMap
        else:
            raise ValueError("Invalid layer specified.")

def conv2prior(map, convShape, prior, fftConv=False, GPU=False):
    # Assert that the map is 2D or has depth 1
    assert map.ndim == 2 or (map.ndim == 3 and map.shape[2] == 1)

    # Choose numpy/scipy or cupy/cupyx
    if GPU:
        map = cp.asarray(map)
        xp = cp
        xsp = csp
    else:
        xp = np
        xsp = sp

    # If the map is 3D with depth 1, squeeze it to 2D
    if map.ndim == 3:
        map = map[:,:,0]
        is_3D = True
    else:
        is_3D = False

    # Pad the map with the prior before making the convolution
    sx, sy = convShape.shape
    px = (sx - 1) // 2
    py = (sy - 1) // 2
    paddedMap = xp.pad(map, ((px, px), (py, py)), constant_values=prior)

    # Make the convolution
    if fftConv:
        conv = xsp.fftconvolve(paddedMap, convShape, mode='valid')
    else:
        conv = xsp.convolve2d(paddedMap, convShape, mode='valid')

    # If the input was 3D with depth 1, add back the depth dimension
    if is_3D:
        conv = conv[:,:,xp.newaxis]
    return conv

def conv3prior(map, convShape, prior, fftConv=False, GPU=False):
    print("THIS FUNCTION IS USING 2D CONVOLUTIONS INSTEAD OF 3D. IT SHOULD BE FIXED.")

    # Choose numpy/scipy or cupy/cupyx
    if GPU:
        map = cp.asarray(map)
        xp = cp
        xsp = csp
    else:
        xp = np
        xsp = sp
    
    # Pad the map with the prior before making the convolution
    sx, sy, sz = convShape.shape
    px = (sx - 1) // 2
    py = (sy - 1) // 2
    pz = (sz - 1) // 2
    paddedMap = xp.pad(map, ((px, px), (py, py), (pz, pz)), constant_values=prior)

    # Make the convolution
    if fftConv:
        conv = xsp.fftconvolve(paddedMap, convShape, mode='valid')
    else:
        conv = xsp.convolve(paddedMap, convShape, mode='valid')
    return conv

if __name__ == '__main__':
    from lidarScan import lidarScan3D
    from utilities import read3DLidarCSV
    from sensorModel import sensorModel3D
    # Example of usage
    # Create a TGM
    tgmOrigin = origin(10, 10, 0)
    tgmSize = size(100, 100, 25)
    resolution = 0.5
    tgmFrame = frame(tgmOrigin, tgmSize, resolution)
    priors = [0.3, 0.3, 0.01]
    maxVelocity = 1
    saturationLimits = [0, 1, 0, 1]
    tgm = TGM(tgmFrame, priors, maxVelocity, saturationLimits, fftConv=True, isGPU=False)

    # Create a sensor model (using the same frame as the TGM for simplicity)
    # The occPrior of the sensor model MUST be the sum of the priors of the TGM
    sM = sensorModel3D(tgmFrame, [0.1, 0.9], sum(priors))

    # Import a sensor measurement and pose
    z_t = read3DLidarCSV("./logs/2024-02-13-10-35-56/z_1.csv")

    assert isinstance(z_t, lidarScan3D)
    
    x_t = pose(position(25, 25, 2.0), orientation(0.0, 0.0, 0.0))

    # Apply voxel grid filter to the lidar scan
    z_t.voxelGridFilter(resolution)

    assert isinstance(z_t, lidarScan3D)

    # Create the instantaneous grid map
    instGridMap = sM.generateGridMap(z_t, x_t)
    instGridMap.plot3D_scatter(isPause=True, value_min=0.7, value_max=1.0)

    # Update the TGM with the instantaneous grid map and the pose
    tgm.update(instGridMap, x_t)

    # Plot the TGM
    tgm.plot3D(isPause=True, value_min=0.7, value_max=1.0)


    # 2D convolution test
    # To test of the 2D convolution works, we are going to create a small 2D map with a single dynamic cell in the center
    testFrame = frame(origin(0, 0, 0), size(11, 11, 1), 1.0)
    testTGM = TGM(testFrame, priors, maxVelocity=1, saturationLimits=saturationLimits, fftConv=False, isGPU=False)
    testTGM.dynamicMap.data[:] = 0.0 # Set all cells to 0
    testTGM.staticMap.data[:] = 0.0 # Set all cells to 0
    testTGM.dynamicMap.data[5, 5, 0] = 0.9  # Set the center cell to be highly dynamic

    # plot the initial map
    testTGM.plot3D(isPause=True, value_min=0.01, value_max=1.0)

    print("data", testTGM.dynamicMap.data[:,:,0])

    # Update the TGM without any new measurements (to see the prediction only)
    testTGM.update(gridMap(testFrame, np.ones((11, 11, 1))*sum(priors)), None)

    # Plot the predicted map
    testTGM.plot3D(isPause=True, value_min=0.1, value_max=1.0)

    print("data after prediction", testTGM.dynamicMap.data[:,:,0])

    # 3D convolution test
    # To test of the 3D convolution works, we are going to create a small 3D map with a single dynamic cell in the center
    testFrame = frame(origin(0, 0, 0), size(11, 11, 11), 1.0)
    testTGM = TGM(testFrame, priors, maxVelocity=1, saturationLimits=saturationLimits, fftConv=False, isGPU=False)
    testTGM.dynamicMap.data[:] = 0.0 # Set all cells to 0
    testTGM.staticMap.data[:] = 0.0 # Set all cells to 0
    testTGM.dynamicMap.data[5, 5, 5] = 0.9  # Set the center cell to be highly dynamic

    # plot the initial map
    testTGM.plot3D(isPause=True, value_min=0.1, value_max=1.0)

    print("data", testTGM.dynamicMap.data[:,:,5])

    # Update the TGM without any new measurements (to see the prediction only)
    testTGM.update(gridMap(testFrame, np.ones((11, 11, 11))*sum(priors)), None)

    # Plot the predicted map
    testTGM.plot3D(isPause=True, value_min=0.01, value_max=1.0)

    print("data after prediction", np.array2string(np.asarray(testTGM.dynamicMap.data[:, :, 5]), formatter={'float_kind': lambda x: f"{x:.4f}"}))


    # Test the dynamic rebalance function
    cframe = frame(origin(0, 0, 0), size(2, 2, 1), 1.0)
    ddata = np.array([[[0.0], [0.5]],
                      [[0.0], [1.0]]])
    sdata = np.array([[[0.0], [0.2]],
                      [[0.0], [0.0]]])
    wdata = np.array([[[0.0], [0.0]],
                      [[0.0], [0.0]]])
    testTGM = TGM(cframe, priors, maxVelocity=1, saturationLimits=saturationLimits, fftConv=False, isGPU=False)
    testTGM.dynamicMap = gridMap(cframe, ddata)
    testTGM.staticMap = gridMap(cframe, sdata)
    testTGM.weatherMap = gridMap(cframe, wdata)
    print("Before rebalance dynamic map data:", testTGM.dynamicMap.data[:,:,0])
    targetDist = discreteDist([0.0, 0.0, 1.0, 0.0, 0.0])
    testTGM.dynamicRebalance(targetDist)
    print("After rebalance dynamic map data:", testTGM.dynamicMap.data[:,:,0])