import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from gridMap import gridMap
from skimage.morphology import disk
from scipy.signal import convolve2d, fftconvolve
import time
from cupyx.scipy.signal import convolve2d as cp_convolve2d
from cupyx.scipy.signal import fftconvolve as cp_fftconvolve
import cupy as cp

class TGM:
    def __init__(self, origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits, fftConv = False, GPU = True):
        assert isinstance(origin[0], int)
        assert isinstance(origin[1], int)
        assert isinstance(width, int)
        assert isinstance(height, int)

        self.origin_x = origin[0]
        self.origin_y = origin[1]
        self.width = width
        self.height = height
        self.resolution = resolution

        self.staticPrior = staticPrior
        self.dynamicPrior = dynamicPrior
        self.weatherPrior = weatherPrior
        self.freePrior = 1 - staticPrior - dynamicPrior - weatherPrior

        self.sdwPrior = self.staticPrior + self.dynamicPrior + self.weatherPrior

        r = int(maxVelocity / self.resolution)
        shape = disk(r).astype(float)
        self.D0 = 1 / np.sum(shape)
        shape /= np.sum(shape)
        shape[len(shape)//2, len(shape)//2] = 0
        self.convShape = shape

        self.staticMap = np.ones((self.width, self.height)) * staticPrior
        self.dynamicMap = np.ones((self.width, self.height)) * dynamicPrior
        self.weatherMap = np.ones((self.width, self.height)) * weatherPrior

        self.satLowS = saturationLimits[0]
        self.satHighS = saturationLimits[1]
        self.satLowD = saturationLimits[2]
        self.satHighD = saturationLimits[3]

        self.fftConv = fftConv

        self.x_t = []

        self.prev_x0 = 0
        self.prev_y0 = 0
        self.prev_x1 = 0
        self.prev_y1 = 0

        self.GPU = GPU
        if self.GPU:
            self.staticMap = cp.asarray(self.staticMap)
            self.dynamicMap = cp.asarray(self.dynamicMap)
            self.weatherMap = cp.asarray(self.weatherMap)
            self.convShape = cp.asarray(self.convShape)

    def update(self, instGridMap, x_t):
        assert isinstance(instGridMap, gridMap)
        print(instGridMap.resolution)
        print(self.resolution)
        assert instGridMap.resolution == self.resolution

        timeStart = time.time()

        # Update ego position (used for visualization purposes only)
        self.x_t = x_t

        # Compute overlaping grid between the instantaneous map and the TGM
        overlapOrigin_x = max(self.origin_x, instGridMap.origin_x)
        overlapOrigin_y = max(self.origin_y, instGridMap.origin_y)
        overlapWidth = min(self.origin_x + self.width, instGridMap.origin_x + instGridMap.width) - overlapOrigin_x
        overlapHeight = min(self.origin_y + self.height, instGridMap.origin_y + instGridMap.height) - overlapOrigin_y
        assert overlapWidth > 0 and overlapHeight > 0

        # Crop the instantaneous map to the overlapping region
        instMap = instGridMap.crop(overlapOrigin_x, overlapOrigin_y, overlapWidth, overlapHeight).data

        if self.GPU:
            instMap = cp.asarray(instMap)

        # Split the instantaneous map into static, dynamic, weather and free maps
        instStaticMap = instMap * self.staticPrior / self.sdwPrior
        instDynamicMap = instMap * self.dynamicPrior / self.sdwPrior
        instWeatherMap = instMap * self.weatherPrior / self.sdwPrior
        instFreeMap = 1 - instStaticMap - instDynamicMap - instWeatherMap

        timeSplit = time.time()

        # Predict based on previous measurements
        predStaticMap, predDynamicMap, predWeatherMap = self.predict(overlapOrigin_x, overlapOrigin_y, overlapWidth, overlapHeight)
        predFreeMap = 1 - predStaticMap - predDynamicMap - predWeatherMap

        timePredict = time.time()

        # Compute the updated maps
        if self.staticPrior != 0:
            staticMatrix = instStaticMap * predStaticMap / self.staticPrior
        else:
            staticMatrix = np.zeros_like(instStaticMap)
        if self.dynamicPrior != 0:
            dynamicMatrix = instDynamicMap * predDynamicMap / self.dynamicPrior
        else:
            dynamicMatrix = np.zeros_like(instDynamicMap)
        if self.weatherPrior != 0:
            weatherMatrix = instWeatherMap * predWeatherMap / self.weatherPrior
        else:
            weatherMatrix = np.zeros_like(instWeatherMap)
        freeMatrix = instFreeMap * predFreeMap / self.freePrior

        # Normalize the maps

        total = staticMatrix + dynamicMatrix + weatherMatrix + freeMatrix
        staticMatrix /= total
        dynamicMatrix /= total
        weatherMatrix /= total

        timeUpdate = time.time()

        # Apply saturation limits
        if self.GPU:
            staticMatrix = cp.clip(staticMatrix, self.satLowS, self.satHighS)
            dynamicMatrix = cp.clip(dynamicMatrix, self.satLowD, self.satHighD)
        else:
            staticMatrix[staticMatrix > self.satHighS] = self.satHighS
            staticMatrix[staticMatrix < self.satLowS] = self.satLowS
            dynamicMatrix[dynamicMatrix > self.satHighD] = self.satHighD
            dynamicMatrix[dynamicMatrix < self.satLowD] = self.satLowD

        timeSat = time.time()

        # Set the cells that were visible to the prior
        self.dynamicMap[self.prev_x0:self.prev_x1, self.prev_y0:self.prev_y1] = 1 - self.staticMap[self.prev_x0:self.prev_x1, self.prev_y0:self.prev_y1] * self.dynamicPrior/(self.dynamicPrior + self.freePrior + self.weatherPrior)

        timeVisible = time.time()

        # Compute visible mask as the portion of the TGM that overlaps with the instantaneous map
        x0 = overlapOrigin_x - self.origin_x
        y0 = overlapOrigin_y - self.origin_y
        x1 = x0 + overlapWidth
        y1 = y0 + overlapHeight

        # Save the visible cells
        self.staticMap[x0:x1, y0:y1] = staticMatrix
        self.dynamicMap[x0:x1, y0:y1] = dynamicMatrix
        self.weatherMap[x0:x1, y0:y1] = weatherMatrix

        # Save the previous visible mask
        self.prev_x0 = x0
        self.prev_y0 = y0
        self.prev_x1 = x1
        self.prev_y1 = y1

        timeSave = time.time()

        # Print times
        print('Split:    ' + str(timeSplit - timeStart))
        print('Predict:  ' + str(timePredict - timeSplit))
        print('Update:   ' + str(timeUpdate - timePredict))
        print('Sat:      ' + str(timeSat - timeUpdate))
        print('Visible:  ' + str(timeVisible - timeSat))
        print('Save:     ' + str(timeSave - timeVisible))
        print('Total:    ' + str(time.time() - timeStart))
        print('')

    def predict(self, overlapOrigin_x=None, overlapOrigin_y=None, overlapWidth=None, overlapHeight=None):
        if overlapOrigin_x is None:
            overlapOrigin_x = self.origin_x
            overlapOrigin_y = self.origin_y
            overlapWidth = self.width
            overlapHeight = self.height

        # Computed cropped maps
        staticMap = self.cropMap('static',overlapOrigin_x, overlapOrigin_y, overlapWidth, overlapHeight)
        dynamicMap = self.cropMap('dynamic',overlapOrigin_x, overlapOrigin_y, overlapWidth, overlapHeight)
        
        # Compute static prediction
        predStaticMap = staticMap

        # Compute dynamic prediction
        dynamicStay = dynamicMap * self.D0
        bounceBack = conv2prior(staticMap, self.convShape, self.staticPrior, self.fftConv, self.GPU) * dynamicMap
        dynamicMove = conv2prior(dynamicMap, self.convShape, self.dynamicPrior, self.fftConv, self.GPU) * (1 - staticMap)

        predDynamicMap = dynamicStay + bounceBack + dynamicMove

        # Compute weather prediction
        predWeatherMap = (1 - predStaticMap - predDynamicMap) * self.weatherPrior / (self.weatherPrior + self.freePrior)

        return predStaticMap, predDynamicMap, predWeatherMap
    
    def computeStaticGridMap(self, following=False, width=0, height=0):

        # If following is True
        if following:
            # Compute the origin
            origin_x = int((self.x_t[0] - width/2) / self.resolution)
            origin_y = int((self.x_t[1] - height/2) / self.resolution)

            # Convert width and height to grid units
            width = int(width/self.resolution)
            height = int(height/self.resolution)
        
            # Compute overlaping grid between the instantaneous map and the TGM
            overlapOrigin_x = max(self.origin_x, origin_x)
            overlapOrigin_y = max(self.origin_y, origin_y)
            overlapWidth = min(self.origin_x + self.width, origin_x + width) - overlapOrigin_x
            overlapHeight = min(self.origin_y + self.height, origin_y + height) - overlapOrigin_y
            assert overlapWidth > 0 and overlapHeight > 0

            # Crop the static map to the overlapping region
            staticMap = self.cropMap('static', overlapOrigin_x, overlapOrigin_y, overlapWidth, overlapHeight)

        else:
            # Origin, width and height are the same as the TGM
            origin_x = self.origin_x
            origin_y = self.origin_y
            width = self.width
            height = self.height

            # Use the full static map
            staticMap = self.staticMap

        if self.GPU:
            return gridMap(overlapOrigin_x, overlapOrigin_y, overlapWidth, overlapHeight, self.resolution, cp.asnumpy(staticMap))
        else:
            return gridMap(overlapOrigin_x, overlapOrigin_y, overlapWidth, overlapHeight, self.resolution, staticMap)

    def plot(self, fig=None, saveImg=False, saveSvg=False, imgName='', section = 'Full', width = 0, height = 0, origin = None, style='combined', egoStyle='rectangle'):
        origin_x = int(origin[0]/self.resolution) if origin is not None else None
        origin_y = int(origin[1]/self.resolution) if origin is not None else None
        width = int(width/self.resolution) if width != 0 else 0
        height = int(height/self.resolution) if height != 0 else 0
        # Assert that the style is valid
        assert style in ['combined', 'static', 'dynamic', 'weather']

        # Assert that the egoStyle is valid
        assert egoStyle in ['none', 'dot', 'rectangle']

        # If fig is None, create a new figure
        if fig is None:
            fig = plt.figure()

        # If section is Following, compute the origin
        if section == 'Following':
            origin_x = int((self.x_t[0] - width/2) / self.resolution)
            origin_y = int((self.x_t[1] - height/2) / self.resolution)

        # If section is Constant, assert that the origin is not None and compute origin
        if section == 'Constant':
            assert origin_x is not None
            assert origin_y is not None

        # If section is Following or Constant, compute the overlaping grid and crop the maps
        if section == 'Following' or section == 'Constant':
            assert width != 0 and height != 0
            # Compute overlaping grid
            overlapOrigin_x = max(self.origin_x, origin_x)
            overlapOrigin_y = max(self.origin_y, origin_y)
            overlapWidth = min(self.origin_x + self.width, origin_x + width) - overlapOrigin_x
            overlapHeight = min(self.origin_y + self.height, origin_y + height) - overlapOrigin_y
            assert overlapWidth > 0 and overlapHeight > 0
            # Crop the maps
            staticMap = self.cropMap('static',overlapOrigin_x, overlapOrigin_y, overlapWidth, overlapHeight)
            dynamicMap = self.cropMap('dynamic',overlapOrigin_x, overlapOrigin_y, overlapWidth, overlapHeight)
            weatherMap = self.cropMap('weather',overlapOrigin_x, overlapOrigin_y, overlapWidth, overlapHeight)
        # Otherwise, use the full maps
        else:
            overlapOrigin_x = self.origin_x
            overlapOrigin_y = self.origin_y
            overlapWidth = self.width
            overlapHeight = self.height
            staticMap = self.staticMap
            dynamicMap = self.dynamicMap
            weatherMap = self.weatherMap

        if self.GPU:
            staticMap = cp.asnumpy(staticMap)
            dynamicMap = cp.asnumpy(dynamicMap)
            weatherMap = cp.asnumpy(weatherMap)

        # Plot the map according to the style
        if style == 'combined':
            I = np.zeros((overlapHeight, overlapWidth, 3))
            I[:,:,0] = 1 - np.transpose(1.0*staticMap + 0.0*dynamicMap + 1.0*weatherMap)
            I[:,:,1] = 1 - np.transpose(0.5*staticMap + 0.5*dynamicMap + 0.0*weatherMap)
            I[:,:,2] = 1 - np.transpose(0.0*staticMap + 1.0*dynamicMap + 1.0*weatherMap)
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(I, vmin=0, vmax=1, origin ="lower",
                    extent=(overlapOrigin_x*self.resolution, (overlapOrigin_x + overlapWidth)*self.resolution,
                            overlapOrigin_y*self.resolution, (overlapOrigin_y + overlapHeight)*self.resolution))
        elif style == 'static':
            I = 1 - np.transpose(staticMap)
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                    extent=(overlapOrigin_x*self.resolution, (overlapOrigin_x + overlapWidth)*self.resolution,
                            overlapOrigin_y*self.resolution, (overlapOrigin_y + overlapHeight)*self.resolution))
        elif style == 'dynamic':
            I = 1 - np.transpose(dynamicMap)
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                    extent=(overlapOrigin_x*self.resolution, (overlapOrigin_x + overlapWidth)*self.resolution,
                            overlapOrigin_y*self.resolution, (overlapOrigin_y + overlapHeight)*self.resolution))
        elif style == 'weather':
            I = 1 - np.transpose(weatherMap)
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                    extent=(overlapOrigin_x*self.resolution, (overlapOrigin_x + overlapWidth)*self.resolution,
                            overlapOrigin_y*self.resolution, (overlapOrigin_y + overlapHeight)*self.resolution))
            
        # Plot the ego pose
        if self.x_t is not None and len(self.x_t) != 0:
            if egoStyle == 'dot':
                plt.plot(self.x_t[0], self.x_t[1], 'ro')
            elif egoStyle == 'rectangle':
                x = self.x_t[0]
                y = self.x_t[1]
                theta = self.x_t[2]
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
                plt.fill([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], color='white', edgecolor='black')
                # Plot the heading as a triangle
                x1 = x + car_length/2 * np.cos(theta)
                y1 = y + car_length/2 * np.sin(theta)
                x2 = x + (car_length/2-car_width) * np.cos(theta) + car_width/2 * np.sin(theta)
                y2 = y + (car_length/2-car_width) * np.sin(theta) - car_width/2 * np.cos(theta)
                x3 = x + (car_length/2-car_width) * np.cos(theta) - car_width/2 * np.sin(theta)
                y3 = y + (car_length/2-car_width) * np.sin(theta) + car_width/2 * np.cos(theta)
                plt.fill([x1, x2, x3, x1], [y1, y2, y3, y1], color='white', edgecolor='black')

        # If saveImg is True, save the image
        if saveImg:
            imsave(imgName + '.png', I, origin ="lower", cmap='gray')

        # If saveSvg is True, save the plot
        if saveSvg:
            plt.savefig(imgName + '.svg', format='svg', dpi=1200)
        
        # Pause to show the image
        plt.pause(0.01)

    def cropMap(self, layer, origin_x, origin_y, width, height):
        assert layer in ['static', 'dynamic', 'weather']
        assert isinstance(origin_x, int)
        assert isinstance(origin_y, int)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert origin_x >= self.origin_x
        assert origin_y >= self.origin_y
        assert origin_x + width <= self.origin_x + self.width
        assert origin_y + height <= self.origin_y + self.height
        x0 = origin_x - self.origin_x
        y0 = origin_y - self.origin_y
        x1 = x0 + width
        y1 = y0 + height
        if layer == 'static':
            return self.staticMap[x0:x1, y0:y1]
        elif layer == 'dynamic':
            return self.dynamicMap[x0:x1, y0:y1]
        elif layer == 'weather':
            return self.weatherMap[x0:x1, y0:y1]
    
def conv2prior(map, convShape, prior, fftConv = False, GPU = False):
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

if __name__ == '__main__':
    origin_x = 10
    origin_y = 10
    width = 20
    height = 10
    resolution = 2
    staticPrior = 0.3
    dynamicPrior = 0.3
    weatherPrior = 0.01
    maxVelocity = 1
    saturationLimits = [0.1, 0.9, 0.1, 0.9]
    tgm = TGM([origin_x, origin_y], width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits)
    tgm.plot()