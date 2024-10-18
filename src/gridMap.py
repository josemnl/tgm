import matplotlib.pyplot as plt
import numpy as np
import pickle

class gridMap:
    def __init__(self, origin_x, origin_y, width, height, resolution, data):
        """
        Origin, width, and height are in grid cells
        Resolution is in meters per grid cell
        """
        assert isinstance(origin_x, int)
        assert isinstance(origin_y, int)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert data.shape[0] == width
        assert data.shape[1] == height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height
        self.resolution = resolution
        self.data = data

    def plot(self, isPause = False):
        """
        Plot the grid map
        """
        I = 1 - np.transpose(self.data)
        plt.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.origin_x*self.resolution, (self.origin_x + self.width)*self.resolution,
                           self.origin_y*self.resolution, (self.origin_y + self.height)*self.resolution))
        plt.show(block=isPause)
        plt.pause(0.0001)

    def crop(self, origin_x, origin_y, width, height):
        """
        Crop the grid map
        """
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
        return gridMap(origin_x, origin_y, width, height, self.resolution, self.data[x0:x1, y0:y1])
    
    def occupancy(self, x, y):
        """
        Get the occupancy of the cell where the point (x, y) is
        """
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert x >= self.origin_x*self.resolution
        assert y >= self.origin_y*self.resolution
        assert x <= (self.origin_x + self.width)*self.resolution
        assert y <= (self.origin_y + self.height)*self.resolution

        ix = np.round((x - self.origin_x*self.resolution)/self.resolution).astype(int)
        iy = np.round((y - self.origin_y*self.resolution)/self.resolution).astype(int)

        return self.data[ix][iy]
    
    def saveState(self, filename):
        original_data = self.data
        self.data = self.data.astype(np.float16)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        self.data = original_data

    @classmethod
    def loadState(cls, filename):
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
            obj.data = obj.data.astype(np.float64)
            return obj

def main():
    origin_x = 0
    origin_y = 0
    width = 10*2
    height = 5*2
    resolution = 0.5

    data = np.zeros((width, height))
    data[0][0] = 1
    data[19][0] = 0.5
    
    grid = gridMap(origin_x, origin_y, width, height, resolution, data)
    grid.plot(isPause=True)

    grid.crop(10, 0, 10, 6).plot(isPause=True)

if __name__ == '__main__':
    main()