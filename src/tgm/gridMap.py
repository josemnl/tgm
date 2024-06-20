import matplotlib.pyplot as plt
import numpy as np

class gridMap:
    def __init__(self, origin, width, height, resolution, data):
        assert len(origin) == 2
        assert data.shape[0] == width*resolution
        assert data.shape[1] == height*resolution
        self.origin = origin
        self.width = width
        self.height = height
        self.resolution = resolution
        self.data = data

    def plot(self):
        """
        Plot the grid map
        """
        I = 1 - np.transpose(self.data)
        plt.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.origin[0], self.origin[0] + self.width,
                           self.origin[1], self.origin[1] + self.height))
        plt.show()

    def crop(self, origin, width, height):
        """
        Crop the grid map
        """
        assert len(origin) == 2
        assert origin[0] >= self.origin[0]
        assert origin[1] >= self.origin[1]
        assert origin[0] + width <= self.origin[0] + self.width
        assert origin[1] + height <= self.origin[1] + self.height
        x0 = int((origin[0] - self.origin[0]) * self.resolution)
        y0 = int((origin[1] - self.origin[1]) * self.resolution)
        x1 = int((origin[0] + width - self.origin[0]) * self.resolution)
        y1 = int((origin[1] + height - self.origin[1]) * self.resolution)
        return gridMap(origin, width, height, self.resolution, self.data[x0:x1, y0:y1])

def main():
    origin = [0, 0]
    width = 10
    height = 5
    resolution = 2

    data = np.zeros((width*resolution, height*resolution))
    data[0][0] = 1
    data[19][1] = 0.5
    
    grid = gridMap(origin, width, height, resolution, data)
    grid.plot()

    grid.crop([5, 0], 5, 3).plot()

if __name__ == '__main__':
    main()