import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from tgm.gridMap import gridMap
from skimage.morphology import disk
from scipy.signal import convolve2d, fftconvolve

class TGM:
    def __init__(self,origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits, fftConv = False):
        self.origin = origin
        self.width = width
        self.height = height
        self.resolution = resolution

        self.staticPrior = staticPrior
        self.dynamicPrior = dynamicPrior
        self.weatherPrior = weatherPrior
        self.freePrior = 1 - staticPrior - dynamicPrior - weatherPrior

        r = int(maxVelocity * resolution)
        shape = disk(r).astype(float)
        self.D0 = 1 / np.sum(shape)
        shape /= np.sum(shape)
        shape[len(shape)//2, len(shape)//2] = 0
        self.convShape = shape

        self.staticMap = np.ones((width*resolution, height*resolution)) * staticPrior
        self.dynamicMap = np.ones((width*resolution, height*resolution)) * dynamicPrior
        self.weatherMap = np.ones((width*resolution, height*resolution)) * weatherPrior

        self.satLowS = saturationLimits[0]
        self.satHighS = saturationLimits[1]
        self.satLowD = saturationLimits[2]
        self.satHighD = saturationLimits[3]

        self.fftConv = fftConv

        self.x_t = []

    def update(self, instGridMap, x_t):
        assert isinstance(instGridMap, gridMap)
        assert instGridMap.resolution == self.resolution

        # Update ego position (used for visualization purposes only)
        self.x_t = x_t

        # Compute overlaping grid between the instantaneous map and the TGM
        overlapOrigin = [max(self.origin[0], instGridMap.origin[0]), max(self.origin[1], instGridMap.origin[1])]
        overlapWidth = min(self.origin[0] + self.width, instGridMap.origin[0] + instGridMap.width) - overlapOrigin[0]
        overlapHeight = min(self.origin[1] + self.height, instGridMap.origin[1] + instGridMap.height) - overlapOrigin[1]
        assert overlapWidth > 0 and overlapHeight > 0

        # Crop the instantaneous map to the overlapping region
        instMap = instGridMap.crop(overlapOrigin, overlapWidth, overlapHeight).data

        # Split the instantaneous map into static, dynamic, weather and free maps
        instStaticMap = instMap * self.staticPrior / (self.staticPrior + self.dynamicPrior + self.weatherPrior)
        instDynamicMap = instMap * self.dynamicPrior / (self.staticPrior + self.dynamicPrior + self.weatherPrior)
        instWeatherMap = instMap * self.weatherPrior / (self.staticPrior + self.dynamicPrior + self.weatherPrior)
        instFreeMap = 1 - instStaticMap - instDynamicMap - instWeatherMap

        # Predict based on previous measurements
        predStaticMap, predDynamicMap, predWeatherMap = self.predict(overlapOrigin, overlapWidth, overlapHeight)
        predFreeMap = 1 - predStaticMap - predDynamicMap - predWeatherMap

        # Compute the normalized maps
        nStatic = instStaticMap * predStaticMap / self.staticPrior
        nDynamic = instDynamicMap * predDynamicMap / self.dynamicPrior
        if self.weatherPrior != 0:
            nWeather = instWeatherMap * predWeatherMap / self.weatherPrior
        else:
            nWeather = np.zeros_like(instWeatherMap)
        nFree = instFreeMap * predFreeMap / self.freePrior

        total = nStatic + nDynamic + nWeather + nFree
        staticMatrix = nStatic / total
        dynamicMatrix = nDynamic / total
        weatherMatrix = nWeather / total

        # Apply saturation limits
        staticMatrix[staticMatrix > self.satHighS] = self.satHighS
        staticMatrix[staticMatrix < self.satLowS] = self.satLowS

        dynamicMatrix[dynamicMatrix > self.satHighD] = self.satHighD
        dynamicMatrix[dynamicMatrix < self.satLowD] = self.satLowD

        # Set the dynamic map to the prior (TODO: improve this)
        self.dynamicMap = (1 - self.staticMap) * self.dynamicPrior/(self.dynamicPrior + self.freePrior + self.weatherPrior)

        # Update the portion of the TGM that overlaps with the instantaneous map
        x0 = int((overlapOrigin[0] - self.origin[0]) * self.resolution)
        y0 = int((overlapOrigin[1] - self.origin[1]) * self.resolution)
        x1 = int((overlapOrigin[0] + overlapWidth - self.origin[0]) * self.resolution)
        y1 = int((overlapOrigin[1] + overlapHeight - self.origin[1]) * self.resolution)
        self.staticMap[x0:x1, y0:y1] = staticMatrix
        self.dynamicMap[x0:x1, y0:y1] = dynamicMatrix
        self.weatherMap[x0:x1, y0:y1] = weatherMatrix

    def predict(self, overlapOrigin=None, overlapWidth=None, overlapHeight=None):
        if overlapOrigin is None:
            overlapOrigin = self.origin
            overlapWidth = self.width
            overlapHeight = self.height

        # Computed cropped maps
        staticMap = self.cropStaticMap(overlapOrigin, overlapWidth, overlapHeight)
        dynamicMap = self.cropDynamicMap(overlapOrigin, overlapWidth, overlapHeight)
        
        # Compute static prediction
        predStaticMap = staticMap

        # Compute dynamic prediction
        dynamicStay = dynamicMap * self.D0
        bounceBack = conv2prior(staticMap, self.convShape, self.staticPrior, self.fftConv) * dynamicMap
        dynamicMove = conv2prior(dynamicMap, self.convShape, self.dynamicPrior, self.fftConv) * (1 - staticMap)

        predDynamicMap = dynamicStay + bounceBack + dynamicMove

        # Compute weather prediction
        predWeatherMap = (1 - predStaticMap - predDynamicMap) * self.weatherPrior / (self.weatherPrior + self.freePrior)

        return predStaticMap, predDynamicMap, predWeatherMap
    
    def computeStaticGridMap(self):
        return gridMap(self.origin, self.width, self.height, self.resolution, self.staticMap)
    
    def plotStaticMap(self, fig=None):
        if fig is None:
            fig = plt.figure()
        I = 1 - np.transpose(self.staticMap)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.origin[0], self.origin[0] + self.width,
                           self.origin[1], self.origin[1] + self.height))
        if self.x_t is not None and len(self.x_t) != 0:
            plt.plot(self.x_t[0], self.x_t[1], 'ro')
        plt.show()

    def plotDynamicMap(self, fig=None):
        if fig is None:
            fig = plt.figure()
        I = 1 - np.transpose(self.dynamicMap)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.origin[0], self.origin[0] + self.width,
                           self.origin[1], self.origin[1] + self.height))
        if self.x_t is not None and len(self.x_t) != 0:
            plt.plot(self.x_t[0], self.x_t[1], 'ro')
        plt.show()

    def plotCombinedMap(self, fig=None, saveImg=False, imgName='', following = False, width = 0, height = 0):
        if fig is None:
            fig = plt.figure()
        if following:
            assert width != 0 and height != 0
            origin = int((self.x_t[0] - width/2) * self.resolution) / self.resolution, int((self.x_t[1] - height/2) * self.resolution) / self.resolution
            # Compute overlaping grid
            overlapOrigin = [max(self.origin[0], origin[0]), max(self.origin[1], origin[1])]
            overlapWidth = min(self.origin[0] + self.width, origin[0] + width) - overlapOrigin[0]
            overlapHeight = min(self.origin[1] + self.height, origin[1] + height) - overlapOrigin[1]
            assert overlapWidth > 0 and overlapHeight > 0
            # Crop the maps
            staticMap = self.cropStaticMap(overlapOrigin, overlapWidth, overlapHeight)
            dynamicMap = self.cropDynamicMap(overlapOrigin, overlapWidth, overlapHeight)
            weatherMap = self.cropWeatherMap(overlapOrigin, overlapWidth, overlapHeight)
        else:
            overlapOrigin = self.origin
            overlapWidth = self.width
            overlapHeight = self.height
            staticMap = self.staticMap
            dynamicMap = self.dynamicMap
            weatherMap = self.weatherMap
        I = np.zeros((int(overlapHeight*self.resolution), int(overlapWidth*self.resolution), 3))
        I[:,:,0] = 1 - np.transpose(1.0*staticMap + 0.0*dynamicMap + 1.0*weatherMap)
        I[:,:,1] = 1 - np.transpose(0.5*staticMap + 0.5*dynamicMap + 0.0*weatherMap)
        I[:,:,2] = 1 - np.transpose(0.0*staticMap + 1.0*dynamicMap + 1.0*weatherMap)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(I, vmin=0, vmax=1, origin ="lower",
                   extent=(overlapOrigin[0], overlapOrigin[0] + overlapWidth,
                           overlapOrigin[1], overlapOrigin[1] + overlapHeight))
        if self.x_t is not None and len(self.x_t) != 0:
            plt.plot(self.x_t[0], self.x_t[1], 'ro')
        if saveImg:
            #plt.savefig(imgName + '.png')
            imsave(imgName + '.png', I, origin ="lower")
        plt.pause(0.01)

    def cropStaticMap(self, origin, width, height):
        assert len(origin) == 2
        assert origin[0] >= self.origin[0]
        assert origin[1] >= self.origin[1]
        assert origin[0] + width <= self.origin[0] + self.width
        assert origin[1] + height <= self.origin[1] + self.height
        x0 = int((origin[0] - self.origin[0]) * self.resolution)
        y0 = int((origin[1] - self.origin[1]) * self.resolution)
        x1 = int((origin[0] + width - self.origin[0]) * self.resolution)
        y1 = int((origin[1] + height - self.origin[1]) * self.resolution)
        return self.staticMap[x0:x1, y0:y1]
    
    def cropDynamicMap(self, origin, width, height):
        assert len(origin) == 2
        assert origin[0] >= self.origin[0]
        assert origin[1] >= self.origin[1]
        assert origin[0] + width <= self.origin[0] + self.width
        assert origin[1] + height <= self.origin[1] + self.height
        x0 = int((origin[0] - self.origin[0]) * self.resolution)
        y0 = int((origin[1] - self.origin[1]) * self.resolution)
        x1 = int((origin[0] + width - self.origin[0]) * self.resolution)
        y1 = int((origin[1] + height - self.origin[1]) * self.resolution)
        return self.dynamicMap[x0:x1, y0:y1]
    
    def cropWeatherMap(self, origin, width, height):
        assert len(origin) == 2
        assert origin[0] >= self.origin[0]
        assert origin[1] >= self.origin[1]
        assert origin[0] + width <= self.origin[0] + self.width
        assert origin[1] + height <= self.origin[1] + self.height
        x0 = int((origin[0] - self.origin[0]) * self.resolution)
        y0 = int((origin[1] - self.origin[1]) * self.resolution)
        x1 = int((origin[0] + width - self.origin[0]) * self.resolution)
        y1 = int((origin[1] + height - self.origin[1]) * self.resolution)
        return self.weatherMap[x0:x1, y0:y1]
    
def conv2prior(map, convShape, prior, fftConv = False):
    # Pad the map with the prior before making the convolution
    sx, sy = convShape.shape
    px = (sx - 1) // 2
    py = (sy - 1) // 2
    paddedMap = np.pad(map, ((px, px), (py, py)), constant_values=prior)
    if fftConv:
        conv = fftconvolve(paddedMap, convShape, mode='valid')
    else:
        conv = convolve2d(paddedMap, convShape, mode='valid')
    return conv

if __name__ == '__main__':
    origin = [0, 0]
    width = 10
    height = 5
    resolution = 2
    staticPrior = 0.3
    dynamicPrior = 0.3
    weatherPrior = 0.01
    maxVelocity = 1
    saturationLimits = [0.1, 0.9, 0.1, 0.9]
    tgm = TGM(origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits)
    tgm.plotCombinedMap()