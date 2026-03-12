import numpy as np
import cupy as cp
import scipy.signal as sp
import cupyx.scipy.signal as csp
from skimage.morphology import disk

from gridMap import discreteDist, gridMap
from spatial import frame, origin, size, pose, position, orientation

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
        self.prev_dynamic_frame = None

        # Open3D persistent visualization state
        self._o3d_vis = None
        self._o3d_pcd = None
        self._o3d_ego = None
        self._o3d_frame = None
        self._o3d_fov_lines = []
        self._o3d_last_bounds = None
        self._o3d_view_initialized = False

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

        if self.GPU and not isinstance(instMap, cp.ndarray):
            instMap = cp.asarray(instMap)
            print('WARNING: The instantaneous map is not a cupy array, but the TGM is using GPU. Converting the instantaneous map to a cupy array, which may cause a slowdown.')
        elif not self.GPU and isinstance(instMap, cp.ndarray):
            instMap = cp.asnumpy(instMap)
            print('WARNING: The instantaneous map is a cupy array, but the TGM is not using GPU. Converting the instantaneous map to a numpy array, which may cause a slowdown.')
        
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
        if self.prev_dynamic_frame is not None:
            self.dynamicMap.update(self.prev_dynamic_frame, (1 - self.staticMap.crop(self.prev_dynamic_frame).data) * self.dynamicPrior / (self.dynamicPrior + self.freePrior + self.weatherPrior))

        # Update the maps in the overlapping region
        self.staticMap.update(overlap, staticMatrix)
        self.dynamicMap.update(overlap, dynamicMatrix)
        self.weatherMap.update(overlap, weatherMatrix)

        # Save the previous dynamic frame for the next update
        self.prev_dynamic_frame = overlap

        # Clean up GPU memory if using GPU
        if self.GPU:
            cp._default_memory_pool.free_all_blocks()

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
                dynamicStay = dynamicMap * self.D0_3D
                bounceBack = conv3prior(staticMap, self.convShape3D, self.staticPrior, self.fftConv, self.GPU) * dynamicMap
                dynamicMove = conv3prior(dynamicMap, self.convShape3D, self.dynamicPrior, self.fftConv, self.GPU) * (1 - staticMap)
            else:
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
        Update the TGM frame and reshape the maps accordingly. New cells are initialized with the prior values.
        '''
        self.staticMap = self.staticMap.reshape(newFrame, self.staticPrior)
        self.dynamicMap = self.dynamicMap.reshape(newFrame, self.dynamicPrior)
        self.weatherMap = self.weatherMap.reshape(newFrame, self.weatherPrior)

        self.frame = newFrame

        # Make sure the previous dynamic frame is within the new map
        if self.prev_dynamic_frame is not None:
            self.prev_dynamic_frame = self.prev_dynamic_frame.computeOverlap(newFrame)

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

    
    def plot(self, ax = None, frame = None, saveMap=False, savePNG=False, saveSvg=False, imgName='', style='combined', egoStyle='rectangle'):
        from plotting import tgm_plot2D
        return tgm_plot2D(self, ax, frame, saveMap, savePNG, saveSvg, imgName, style, egoStyle)

    def plot3D(self, ax = None, frame: frame = None, value_min: float = 0.0, value_max: float = 1.0) -> None:
        from plotting import tgm_plot3D
        return tgm_plot3D(self, ax, frame, value_min, value_max)

    def plot3D_open3d(self, frame: frame = None, isPause=False, value_min: float = 0.0, value_max: float = 1.0) -> None:
        from plotting import tgm_plot3D_open3d
        return tgm_plot3D_open3d(self, frame, isPause, value_min, value_max)

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
        conv = xsp.convolve(paddedMap, convShape, mode='valid', method='direct')
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
    instGridMap.plot3D(isPause=True, value_min=0.7, value_max=1.0)

    # Update the TGM with the instantaneous grid map and the pose
    tgm.update(instGridMap, x_t)

    # Plot the TGM
    tgm.plot3D(value_min=0.7, value_max=1.0)

    # Plot using Open3D
    tgm.plot3D_open3d(isPause=True, value_min=0.7, value_max=1.0)


    # 2D convolution test
    # To test of the 2D convolution works, we are going to create a small 2D map with a single dynamic cell in the center
    testFrame = frame(origin(0, 0, 0), size(11, 11, 1), 1.0)
    testTGM = TGM(testFrame, priors, maxVelocity=1, saturationLimits=saturationLimits, fftConv=False, isGPU=False)
    testTGM.dynamicMap.data[:] = 0.0 # Set all cells to 0
    testTGM.staticMap.data[:] = 0.0 # Set all cells to 0
    testTGM.dynamicMap.data[5, 5, 0] = 0.9  # Set the center cell to be highly dynamic

    # plot the initial map
    testTGM.plot3D(value_min=0.01, value_max=1.0)

    print("data", testTGM.dynamicMap.data[:,:,0])

    # Update the TGM without any new measurements (to see the prediction only)
    testTGM.update(gridMap(testFrame, np.ones((11, 11, 1))*sum(priors)), None)

    # Plot the predicted map
    testTGM.plot3D(value_min=0.1, value_max=1.0)

    print("data after prediction", testTGM.dynamicMap.data[:,:,0])

    # 3D convolution test
    # To test of the 3D convolution works, we are going to create a small 3D map with a single dynamic cell in the center
    testFrame = frame(origin(0, 0, 0), size(11, 11, 11), 1.0)
    testTGM = TGM(testFrame, priors, maxVelocity=1, saturationLimits=saturationLimits, fftConv=False, isGPU=False)
    testTGM.dynamicMap.data[:] = 0.0 # Set all cells to 0
    testTGM.staticMap.data[:] = 0.0 # Set all cells to 0
    testTGM.dynamicMap.data[5, 5, 5] = 0.9  # Set the center cell to be highly dynamic

    # plot the initial map
    testTGM.plot3D(value_min=0.1, value_max=1.0)

    print("data", testTGM.dynamicMap.data[:,:,5])

    # Update the TGM without any new measurements (to see the prediction only)
    testTGM.update(gridMap(testFrame, np.ones((11, 11, 11))*sum(priors)), None)

    # Plot the predicted map
    testTGM.plot3D(value_min=0.01, value_max=1.0)

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