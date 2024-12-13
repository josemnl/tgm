import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import cupy as cp

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

    def toCPU(self):
        if isinstance(self.data, cp.ndarray):
            return gridMap(self.origin_x, self.origin_y, self.width, self.height, self.resolution, cp.asnumpy(self.data))
        return self

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

    def savePNG(self, filename):
        """
        Save the grid map as a PNG image
        """
        I = 1 - np.transpose(self.data)
        plt.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.origin_x*self.resolution, (self.origin_x + self.width)*self.resolution,
                           self.origin_y*self.resolution, (self.origin_y + self.height)*self.resolution))
        plt.savefig(filename)

    def crop(self, origin_x, origin_y, width, height):
        """
        Crop the grid map to a new grid map with the specified origin and size.
        Throws an error if the new grid is outside the old one.
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
    
    def reshape(self, origin_x, origin_y, width, height, fill_value):
        """
        Reshape the grid map.
        If the new grid is partially outside the old one, the new cells are initialized with the fill value.
        """
        assert isinstance(origin_x, int)
        assert isinstance(origin_y, int)
        assert isinstance(width, int)
        assert isinstance(height, int)
        overlap_origin_x, overlap_origin_y, overlap_width, overlap_height = self.computeOverlap(origin_x, origin_y, width, height)
        new_data = np.full((width, height), fill_value)
        ix_0 = overlap_origin_x - origin_x
        iy_0 = overlap_origin_y - origin_y
        ix_1 = ix_0 + overlap_width
        iy_1 = iy_0 + overlap_height
        nx_0 = overlap_origin_x - self.origin_x
        ny_0 = overlap_origin_y - self.origin_y
        nx_1 = nx_0 + overlap_width
        ny_1 = ny_0 + overlap_height

        new_data[ix_0:ix_1, iy_0:iy_1] = self.data[nx_0:nx_1, ny_0:ny_1]
        return gridMap(origin_x, origin_y, width, height, self.resolution, new_data)

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

    def computeOverlap(self, other_origin_x, other_origin_y, other_width, other_height):
        """
        Compute the overlap between this grid and another grid.
        
        Parameters:
        other_origin_x (int): Origin x of the other grid.
        other_origin_y (int): Origin y of the other grid.
        other_width (int): Width of the other grid.
        other_height (int): Height of the other grid.
        
        Returns:
        overlap_origin_x, overlap_origin_y, overlap_width, overlap_height
        """
        assert isinstance(other_origin_x, int)
        assert isinstance(other_origin_y, int)
        assert isinstance(other_width, int)
        assert isinstance(other_height, int)
        
        overlap_origin_x = max(self.origin_x, other_origin_x)
        overlap_origin_y = max(self.origin_y, other_origin_y)
        overlap_width = min(self.origin_x + self.width, other_origin_x + other_width) - overlap_origin_x
        overlap_height = min(self.origin_y + self.height, other_origin_y + other_height) - overlap_origin_y
        
        return overlap_origin_x, overlap_origin_y, overlap_width, overlap_height
    
    def drawFilledRectangle(self, x, y, theta, length, width, fill_value):
        """
        Draw a filled rectangle in the grid map.
        
        Parameters:
        x (float): x coordinate of the center of the rectangle.
        y (float): y coordinate of the center of the rectangle.
        theta (float): angle of the rectangle.
        length (float): length of the rectangle.
        width (float): width of the rectangle.
        fill_value (float): value to fill the rectangle with.
        """
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(theta, float)
        assert isinstance(length, float)
        assert isinstance(width, float)
        assert isinstance(fill_value, float)
        
        # Compute the corners of the rectangle
        corners = np.array([[x + length/2, y + width/2],
                             [x - length/2, y + width/2],
                             [x - length/2, y - width/2],
                             [x + length/2, y - width/2]])
        
        # Rotate the corners around the center of the rectangle
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        rotated_corners = np.dot(rotation_matrix, (corners - np.array([x, y])).T).T + np.array([x, y])

        # Translate the corners to the grid map
        rotated_corners[:, 0] = (rotated_corners[:, 0] - self.origin_x*self.resolution)/self.resolution
        rotated_corners[:, 1] = (rotated_corners[:, 1] - self.origin_y*self.resolution)/self.resolution

        # Swap x and y (for consistency with openCV)
        rotated_corners[:, 0], rotated_corners[:, 1] = rotated_corners[:, 1], rotated_corners[:, 0].copy()
        
        # Draw the rectangle using OpenCV fillPoly
        points = rotated_corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(self.data, [points], fill_value)

    @classmethod
    def loadState(cls, filename, data_type = np.float64):
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
            obj.data = obj.data.astype(data_type)
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
    grid.drawFilledRectangle(0.0, 2.0, 0.0, 2.0, 1.0, 1.0)
    grid.plot(isPause=True)

    grid.crop(10, 0, 10, 6).plot(isPause=True)

if __name__ == '__main__':
    main()