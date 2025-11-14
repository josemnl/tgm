import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
try:
    import cupy as cp
    _cupy_available = True
except ImportError:
    cp = None
    _cupy_available = False
from typing import Tuple, Union, Any

class frame:
    def __init__(self, origin_x: int, origin_y: int, width: int, height: int, resolution: float):
        """
        Origin, width, and height are in grid cells
        Resolution is in meters per grid cell
        """
        assert isinstance(origin_x, int)
        assert isinstance(origin_y, int)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(resolution, float)
        assert width > 0
        assert height > 0
        assert resolution > 0
        self.ox = origin_x
        self.oy = origin_y
        self.w = width
        self.h = height
        self.r = resolution

    def contains(self, other: 'frame') -> bool:
        return self.ox <= other.ox and self.oy <= other.oy and self.ox + self.w >= other.ox + other.w and self.oy + self.h >= other.oy + other.h

    def computeOverlap(self, other: 'frame') -> 'frame':
        """
        Compute the overlap between this frame and another frame.
        """
        overlap_origin_x = max(self.ox, other.ox)
        overlap_origin_y = max(self.oy, other.oy)
        overlap_width = min(self.ox + self.w, other.ox + other.w) - overlap_origin_x
        overlap_height = min(self.oy + self.h, other.oy + other.h) - overlap_origin_y
        
        return frame(overlap_origin_x, overlap_origin_y, overlap_width, overlap_height, self.r)

    @classmethod
    def frameAroundPose(cls, x: float, y: float, width: int, height: int, resolution: float) -> 'frame':
        """
        Create a frame centered around a pose with the specified width and height.
        """
        origin_x = int((x / resolution) - width/2)
        origin_y = int((y / resolution) - height/2)
        return cls(origin_x, origin_y, width, height, resolution)

class gridMap:
    def __init__(self, gridFrame: frame, data: Union[np.ndarray, Any]):
        """
        Origin, width, and height are in grid cells
        Resolution is in meters per grid cell
        """
        self.frame = gridFrame
        self.data = data

    @property
    def isGPU(self) -> bool:
        return _cupy_available and isinstance(self.data, cp.ndarray)
    
    @property
    def isBool(self) -> bool:
        return self.data.dtype == bool

    def toCPU(self) -> 'gridMap':
        if self.isGPU and cp is not None:
            return gridMap(self.frame, cp.asnumpy(self.data))
        return self
    
    def toBool(self, threshold: float) -> 'gridMap':
        return gridMap(self.frame, self.data > threshold)

    def plot(self, isPause: bool = False) -> None:
        I = 1 - np.transpose(self.data)
        plt.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.frame.ox*self.frame.r, (self.frame.ox + self.frame.w)*self.frame.r,
                           self.frame.oy*self.frame.r, (self.frame.oy + self.frame.h)*self.frame.r))
        plt.show(block=isPause)
        plt.pause(0.0001)

    def savePNG(self, filename: str) -> None:
        I = 1 - np.transpose(self.data)
        plt.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.frame.ox*self.frame.r, (self.frame.ox + self.frame.w)*self.frame.r,
                           self.frame.oy*self.frame.r, (self.frame.oy + self.frame.h)*self.frame.r))
        plt.savefig(filename)

    def contains(self, frame) -> bool:
        return self.frame.contains(frame)

    def crop(self, newFrame: frame) -> 'gridMap':
        """
        Crop the grid map to a new grid map with the specified origin and size.
        Throws an error if the new grid is outside the old one.
        """
        if not self.contains(newFrame):
            raise ValueError("New grid is outside the old one")
        x0 = newFrame.ox - self.frame.ox
        y0 = newFrame.oy - self.frame.oy
        x1 = x0 + newFrame.w
        y1 = y0 + newFrame.h
        return gridMap(newFrame, self.data[x0:x1, y0:y1])
    
    def reshape(self, newFrame: frame, fill_value: float) -> 'gridMap':
        """
        Reshape the grid map.
        If the new grid is partially outside the old one, the new cells are initialized with the fill value.
        """
        overlap = self.computeOverlap(newFrame)
        if self.isGPU:
            newData = cp.full((newFrame.w, newFrame.h), fill_value)
        else:
            newData = np.full((newFrame.w, newFrame.h), fill_value)
        ix_0 = overlap.ox - newFrame.ox
        iy_0 = overlap.oy - newFrame.oy
        ix_1 = ix_0 + overlap.w - 1
        iy_1 = iy_0 + overlap.h - 1
        nx_0 = overlap.ox - self.frame.ox
        ny_0 = overlap.oy - self.frame.oy
        nx_1 = nx_0 + overlap.w - 1
        ny_1 = ny_0 + overlap.h - 1

        newData[ix_0:ix_1, iy_0:iy_1] = self.data[nx_0:nx_1, ny_0:ny_1]
        return gridMap(newFrame, newData)

    def occupancy(self, x: float, y: float) -> float:
        ix = np.round((x - self.frame.ox*self.frame.r)/self.frame.r).astype(int)
        iy = np.round((y - self.frame.oy*self.frame.r)/self.frame.r).astype(int)
        return self.data[ix][iy]
    
    def saveState(self, filename: str) -> None:
        original_data = self.data
        self.data = self.data.astype(np.float16)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        self.data = original_data

    def computeOverlap(self, frame: frame) -> 'frame':
        """
        Compute the overlap between this grid and another grid.
        """
        return self.frame.computeOverlap(frame)
    
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
        rotated_corners[:, 0] = (rotated_corners[:, 0] - self.frame.ox*self.frame.r)/self.frame.r
        rotated_corners[:, 1] = (rotated_corners[:, 1] - self.frame.oy*self.frame.r)/self.frame.r

        # Swap x and y (for consistency with openCV)
        rotated_corners[:, 0], rotated_corners[:, 1] = rotated_corners[:, 1], rotated_corners[:, 0].copy()
        
        # Draw the rectangle using OpenCV fillPoly
        points = rotated_corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(self.data, [points], fill_value)

    def diff(self, otherGM: 'gridMap') -> 'gridMap':
        # Implements the set difference between two grid maps
        assert self.frame.w == otherGM.frame.w
        assert self.frame.h == otherGM.frame.h
        assert self.frame.r == otherGM.frame.r
        assert self.frame.ox == otherGM.frame.ox
        assert self.frame.oy == otherGM.frame.oy
        assert self.isBool
        assert otherGM.isBool
        if self.isGPU:
            return gridMap(self.frame, cp.logical_and(self.data, cp.logical_not(otherGM.data)))
        return gridMap(self.frame, np.logical_and(self.data, np.logical_not(otherGM.data)))
    
    def union(self, otherGM: 'gridMap') -> 'gridMap':
        # Implements the set union between two grid maps
        assert self.frame.w == otherGM.frame.w
        assert self.frame.h == otherGM.frame.h
        assert self.frame.r == otherGM.frame.r
        assert self.frame.ox == otherGM.frame.ox
        assert self.frame.oy == otherGM.frame.oy
        assert self.isBool
        assert otherGM.isBool
        if self.isGPU:
            return gridMap(self.frame, cp.logical_or(self.data, otherGM.data))
        return gridMap(self.frame, np.logical_or(self.data, otherGM.data))

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
    currentFrame = frame(origin_x, origin_y, width, height, resolution)

    data = np.zeros((width, height))
    data[0][0] = 1
    data[19][0] = 0.5
    
    grid = gridMap(currentFrame, data)
    grid.drawFilledRectangle(0.0, 2.0, 0.0, 2.0, 1.0, 1.0)
    grid.plot(isPause=True)

    newFrame = frame(10, 0, 10, 6, 0.5)

    grid.crop(newFrame).plot(isPause=True)

if __name__ == '__main__':
    main()