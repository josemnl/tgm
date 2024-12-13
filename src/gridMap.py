import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import cupy as cp
from typing import Tuple, Union

class gridMap:
    def __init__(self, origin_x: int, origin_y: int, width: int, height: int, resolution: float, data: Union[np.ndarray, cp.ndarray]):
        """
        Origin, width, and height are in grid cells
        Resolution is in meters per grid cell
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height
        self.resolution = resolution
        self.data = data

    def toCPU(self) -> 'gridMap':
        if isinstance(self.data, cp.ndarray):
            return gridMap(self.origin_x, self.origin_y, self.width, self.height, self.resolution, cp.asnumpy(self.data))
        return self

    def plot(self, isPause: bool = False) -> None:
        I = 1 - np.transpose(self.data)
        plt.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.origin_x*self.resolution, (self.origin_x + self.width)*self.resolution,
                           self.origin_y*self.resolution, (self.origin_y + self.height)*self.resolution))
        plt.show(block=isPause)
        plt.pause(0.0001)

    def savePNG(self, filename: str) -> None:
        I = 1 - np.transpose(self.data)
        plt.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.origin_x*self.resolution, (self.origin_x + self.width)*self.resolution,
                           self.origin_y*self.resolution, (self.origin_y + self.height)*self.resolution))
        plt.savefig(filename)

    def contains(self, origin_x: int, origin_y: int, width: int, height: int) -> bool:
        return origin_x >= self.origin_x and origin_y >= self.origin_y and origin_x + width <= self.origin_x + self.width and origin_y + height <= self.origin_y + self.height

    def crop(self, origin_x: int, origin_y: int, width: int, height: int) -> 'gridMap':
        """
        Crop the grid map to a new grid map with the specified origin and size.
        Throws an error if the new grid is outside the old one.
        """
        if not self.contains(origin_x, origin_y, width, height):
            raise ValueError("New grid is outside the old one")
        x0 = origin_x - self.origin_x
        y0 = origin_y - self.origin_y
        x1 = x0 + width
        y1 = y0 + height
        return gridMap(origin_x, origin_y, width, height, self.resolution, self.data[x0:x1, y0:y1])
    
    def reshape(self, origin_x: int, origin_y: int, width: int, height: int, fill_value: float) -> 'gridMap':
        """
        Reshape the grid map.
        If the new grid is partially outside the old one, the new cells are initialized with the fill value.
        """
        overlap_origin_x, overlap_origin_y, overlap_width, overlap_height = self.computeOverlap(origin_x, origin_y, width, height)
        if isinstance(self.data, cp.ndarray):
            new_data = cp.full((width, height), fill_value)
        else:
            new_data = np.full((width, height), fill_value)
        ix_0 = overlap_origin_x - origin_x
        iy_0 = overlap_origin_y - origin_y
        ix_1 = ix_0 + overlap_width - 1
        iy_1 = iy_0 + overlap_height - 1
        nx_0 = overlap_origin_x - self.origin_x
        ny_0 = overlap_origin_y - self.origin_y
        nx_1 = nx_0 + overlap_width - 1
        ny_1 = ny_0 + overlap_height - 1

        new_data[ix_0:ix_1, iy_0:iy_1] = self.data[nx_0:nx_1, ny_0:ny_1]
        return gridMap(origin_x, origin_y, width, height, self.resolution, new_data)

    def occupancy(self, x: float, y: float) -> float:
        ix = np.round((x - self.origin_x*self.resolution)/self.resolution).astype(int)
        iy = np.round((y - self.origin_y*self.resolution)/self.resolution).astype(int)
        return self.data[ix][iy]
    
    def saveState(self, filename: str) -> None:
        original_data = self.data
        self.data = self.data.astype(np.float16)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        self.data = original_data

    def computeOverlap(self, other_origin_x: int, other_origin_y: int, other_width: int, other_height: int) -> Tuple[int, int, int, int]:
        """
        Compute the overlap between this grid and another grid.
        """
        overlap_origin_x = max(self.origin_x, other_origin_x)
        overlap_origin_y = max(self.origin_y, other_origin_y)
        overlap_width = min(self.origin_x + self.width, other_origin_x + other_width) - overlap_origin_x
        overlap_height = min(self.origin_y + self.height, other_origin_y + other_height) - overlap_origin_y
        
        return overlap_origin_x, overlap_origin_y, overlap_width, overlap_height
    
    def drawFilledRectangle(self, x: float, y: float, theta: float, length: float, width: float, fill_value: float) -> None:
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
    def loadState(cls, filename: str, data_type: np.dtype = np.float64) -> 'gridMap':
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
            obj.data = obj.data.astype(data_type)
            return obj

def main() -> None:
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