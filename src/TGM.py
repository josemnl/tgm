import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from gridMap import gridMap, frame, orientation, origin, position, size, pose
from skimage.morphology import disk
from scipy.signal import convolve2d, fftconvolve
from cupyx.scipy.signal import convolve2d as cp_convolve2d
from cupyx.scipy.signal import fftconvolve as cp_fftconvolve
import cupy as cp
import matplotlib

from lidarScan import lidarScan, lidarScan3D
from sensorModel import sensorModel3D
from utilities import read3DLidarCSV
matplotlib.use('Qt5Agg')

class TGM:
    def __init__(self, tgmFrame: frame, priors: list, velocities: list, saturationLimits: list, fftConv=False, isGPU=True):
        assert isinstance(tgmFrame, frame)
        assert isinstance(priors[0], (float, int))
        assert isinstance(priors[1], (float, int))
        assert isinstance(priors[2], (float, int))
        assert isinstance(velocities[0], int)
        assert isinstance(velocities[1], int)
        assert isinstance(velocities[2], int)
        assert isinstance(saturationLimits, list) and len(saturationLimits) == 4

        self.frame = tgmFrame
        self.staticPrior = priors[0]
        self.dynamicPrior = priors[1]
        self.weatherPrior = priors[2]
        self.freePrior = 1 - priors[0] - priors[1] - priors[2]
        self.sdwPrior = self.staticPrior + self.dynamicPrior + self.weatherPrior
        self.staticMap = gridMap(self.frame, np.ones((self.frame.size.w, self.frame.size.h, self.frame.size.d)) * priors[0])
        self.dynamicMap = gridMap(self.frame, np.ones((self.frame.size.w, self.frame.size.h, self.frame.size.d)) * priors[1])
        self.weatherMap = gridMap(self.frame, np.ones((self.frame.size.w, self.frame.size.h, self.frame.size.d)) * priors[2])

        """
        r = int(maxVelocity / self.frame.r)
        shape = disk(r).astype(float)
        self.D0 = 1 / np.sum(shape)
        shape /= np.sum(shape)
        shape[len(shape)//2, len(shape)//2] = 0
        self.convShape = shape
        """

        # Create a 3D convolution shape based on the provided velocities
        # (For now, it's just a cuboid)
        shape = np.ones((2*velocities[0]+1, 2*velocities[1]+1, 2*velocities[2]+1))
        self.D0 = 1 / np.sum(shape)
        shape /= np.sum(shape)
        shape[velocities[0], velocities[1], velocities[2]] = 0
        self.convShape = shape

        self.satLowS = saturationLimits[0]
        self.satHighS = saturationLimits[1]
        self.satLowD = saturationLimits[2]
        self.satHighD = saturationLimits[3]
        self.fftConv = fftConv

        self.x_t = None
        self.prev_region = [0, 0, 0, 0, 0, 0]

        self.GPU = isGPU
        if self.GPU:
            self.staticMap = self.staticMap.toGPU()
            self.dynamicMap = self.dynamicMap.toGPU()
            self.weatherMap = self.weatherMap.toGPU()
            self.convShape = cp.asarray(self.convShape)

    @property
    def freeMap(self):
        return gridMap(self.frame, 1 - self.staticMap.data - self.dynamicMap.data - self.weatherMap.data)

    def update(self, instGridMap, x_t: pose | None):
        assert isinstance(instGridMap, gridMap)
        assert instGridMap.frame.r == self.frame.r

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
            staticMatrix = cp.zeros_like(instStaticMap) if self.GPU else np.zeros_like(instStaticMap)
        if self.dynamicPrior != 0:
            dynamicMatrix = instDynamicMap * predDynamicMap / self.dynamicPrior
        else:
            dynamicMatrix = cp.zeros_like(instDynamicMap) if self.GPU else np.zeros_like(instDynamicMap)
        if self.weatherPrior != 0:
            weatherMatrix = instWeatherMap * predWeatherMap / self.weatherPrior
        else:
            weatherMatrix = cp.zeros_like(instWeatherMap) if self.GPU else np.zeros_like(instWeatherMap)
        freeMatrix = instFreeMap * predFreeMap / self.freePrior

        # Normalize the maps
        total = staticMatrix + dynamicMatrix + weatherMatrix + freeMatrix
        staticMatrix /= total
        dynamicMatrix /= total
        weatherMatrix /= total

        # Apply saturation limits
        if self.GPU:
            staticMatrix = cp.clip(staticMatrix, self.satLowS, self.satHighS)
            dynamicMatrix = cp.clip(dynamicMatrix, self.satLowD, self.satHighD)
        else:
            staticMatrix = np.clip(staticMatrix, self.satLowS, self.satHighS)
            dynamicMatrix = np.clip(dynamicMatrix, self.satLowD, self.satHighD)

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
            dynamicStay = dynamicMap * self.D0
            bounceBack = conv3prior(staticMap, self.convShape, self.staticPrior, self.fftConv, self.GPU) * dynamicMap
            dynamicMove = conv3prior(dynamicMap, self.convShape, self.dynamicPrior, self.fftConv, self.GPU) * (1 - staticMap)
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

    def plot3D(self, frame: frame = None, isPause=False, value_min: float = 0.0, value_max: float = 1.0) -> None:
        """
        3D plot of the TGM using scatter plot.
        Very similar to the one in gridMap.py, but plotting the 3 layers each using
        a different RGB layer for the color, similarly to the 2D case.
        """
        if frame is None:
            frame = self.frame
        overlap = self.frame.computeOverlap(frame)
        staticMap = self.staticMap.crop(overlap).toCPU().data
        dynamicMap = self.dynamicMap.crop(overlap).toCPU().data
        weatherMap = self.weatherMap.crop(overlap).toCPU().data

        # Create a figure and a 3D axis
        fig = plt.figure()
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
        mask = (staticMap + dynamicMap + weatherMap >= value_min) & (staticMap + dynamicMap + weatherMap <= value_max)
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        colors = colors[mask]

        # Scatter plot
        ax.scatter(X, Y, Z, c=colors, marker='o', s=1)

        # Set labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

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
    # Pad the map with the prior before making the convolution
    sx, sy = convShape.shape
    px = (sx - 1) // 2
    py = (sy - 1) // 2
    if fftConv:
        if GPU:
            paddedMap = cp.pad(cp.asarray(map), ((px, px), (py, py)), constant_values=prior)
            conv = cp_fftconvolve(paddedMap, convShape, mode='valid')
        else:
            paddedMap = np.pad(map, ((px, px), (py, py)), constant_values=prior)
            conv = fftconvolve(paddedMap, convShape, mode='valid')
    else:
        if GPU:
            paddedMap = cp.pad(cp.asarray(map), ((px, px), (py, py)), constant_values=prior)
            conv = cp_convolve2d(paddedMap, convShape, mode='valid')
        else:
            paddedMap = np.pad(map, ((px, px), (py, py)), constant_values=prior)
            conv = convolve2d(paddedMap, convShape, mode='valid')
    return conv

def conv3prior(map, convShape, prior, fftConv=False, GPU=False):
    print("THIS FUNCTION IS USING 2D CONVOLUTIONS INSTEAD OF 3D. IT SHOULD BE FIXED.")
    # Pad the map with the prior before making the convolution
    sx, sy, sz = convShape.shape
    px = (sx - 1) // 2
    py = (sy - 1) // 2
    pz = (sz - 1) // 2
    if fftConv:
        if GPU:
            paddedMap = cp.pad(cp.asarray(map), ((px, px), (py, py), (pz, pz)), constant_values=prior)
            conv = cp_fftconvolve(paddedMap, convShape, mode='valid')
        else:
            paddedMap = np.pad(map, ((px, px), (py, py), (pz, pz)), constant_values=prior)
            conv = fftconvolve(paddedMap, convShape, mode='valid')
    else:
        if GPU:
            paddedMap = cp.pad(cp.asarray(map), ((px, px), (py, py), (pz, pz)), constant_values=prior)
            conv = cp_convolve2d(paddedMap, convShape, mode='valid')
        else:
            paddedMap = np.pad(map, ((px, px), (py, py), (pz, pz)), constant_values=prior)
            conv = convolve2d(paddedMap, convShape, mode='valid')
    return conv

if __name__ == '__main__':
    # Example of usage
    # Create a TGM
    tgmOrigin = origin(10, 10, 0)
    tgmSize = size(100, 100, 25)
    resolution = 0.5
    tgmFrame = frame(tgmOrigin, tgmSize, resolution)
    priors = [0.3, 0.3, 0.01]
    velocities = [1, 1, 0]
    saturationLimits = [0.1, 0.9, 0.1, 0.9]
    tgm = TGM(tgmFrame, priors, velocities, saturationLimits, fftConv=True, isGPU=False)

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
