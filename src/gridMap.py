from turtle import width
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import cupy as cp
from typing import Tuple, Union

class position:
    def __init__(self, x: float, y: float, z: float = 0.0):
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        if not isinstance(other, position):
            return NotImplemented
        return position(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        if not isinstance(other, position):
            return NotImplemented
        return position(self.x - other.x, self.y - other.y, self.z - other.z)

class orientation:
    def __init__(self, roll: float, pitch: float, yaw: float):
        assert isinstance(roll, float)
        assert isinstance(pitch, float)
        assert isinstance(yaw, float)
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __add__(self, other):
        if not isinstance(other, orientation):
            return NotImplemented
        return orientation(self.roll + other.roll, self.pitch + other.pitch, self.yaw + other.yaw)
    
    def __sub__(self, other):
        if not isinstance(other, orientation):
            return NotImplemented
        return orientation(self.roll - other.roll, self.pitch - other.pitch, self.yaw - other.yaw)

class pose:
    def __init__(self, pose_position: position, pose_orientation: orientation):
        assert isinstance(pose_position, position)
        assert isinstance(pose_orientation, orientation)
        self.position = pose_position
        self.orientation = pose_orientation

    def __add__(self, other):
        if not isinstance(other, pose):
            return NotImplemented
        new_position = self.position + other.position
        new_orientation = self.orientation + other.orientation
        return pose(new_position, new_orientation)
    
    def __sub__(self, other):
        if not isinstance(other, pose):
            return NotImplemented
        new_position = self.position - other.position
        new_orientation = self.orientation - other.orientation
        return pose(new_position, new_orientation)

class origin:
    def __init__(self, x: int, y: int, z: int = 0):
        """
        Origin coordinates of a grid, measured in cells
        """
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(z, int)
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        if not isinstance(other, origin):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

class size:
    def __init__(self, width: int, height: int, depth: int = 1):
        """
        Size of a grid, measured in number of cells
        """
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(depth, int)
        assert width >= 0
        assert height >= 0
        assert depth >= 0
        self.w = width
        self.h = height
        self.d = depth

    def __eq__(self, other):
        if not isinstance(other, size):
            return NotImplemented
        return self.w == other.w and self.h == other.h and self.d == other.d

class frame:
    def __init__(self, frame_origin: origin, frame_size: size, resolution: float):
        """
        Origin, width, and height are in grid cells
        Resolution is in meters per grid cell
        """
        # Validate input types
        assert isinstance(frame_origin, origin)
        assert isinstance(frame_size, size)
        assert isinstance(resolution, float)

        # Validate input values
        assert resolution > 0

        # Save parameters
        self.ox = frame_origin.x
        self.oy = frame_origin.y
        self.oz = frame_origin.z
        self.w = frame_size.w
        self.h = frame_size.h
        self.d = frame_size.d
        self.r = resolution

    def __eq__(self, other):
        if not isinstance(other, frame):
            return NotImplemented
        return self.ox == other.ox and self.oy == other.oy and self.oz == other.oz and \
               self.w == other.w and self.h == other.h and self.d == other.d and \
               self.r == other.r

    def contains(self, other: 'frame') -> bool:
        return self.ox <= other.ox and \
               self.oy <= other.oy and \
               self.oz <= other.oz and \
               self.ox + self.w >= other.ox + other.w and \
               self.oy + self.h >= other.oy + other.h and \
               self.oz + self.d >= other.oz + other.d

    def computeOverlap(self, other: 'frame') -> 'frame':
        """
        Compute the overlap between this frame and another frame.
        """
        overlap_origin_x = max(self.ox, other.ox)
        overlap_origin_y = max(self.oy, other.oy)
        overlap_origin_z = max(self.oz, other.oz)
        overlap_width = min(self.ox + self.w, other.ox + other.w) - overlap_origin_x
        overlap_height = min(self.oy + self.h, other.oy + other.h) - overlap_origin_y
        overlap_depth = min(self.oz + self.d, other.oz + other.d) - overlap_origin_z

        overlap_origin = origin(overlap_origin_x, overlap_origin_y, overlap_origin_z)
        overlap_size = size(overlap_width, overlap_height, overlap_depth)

        return frame(overlap_origin, overlap_size, self.r)

    @classmethod
    def frameAroundPosition(cls, pos: position, frame_size: size, resolution: float) -> 'frame':
        """
        Create a frame centered around a pose with the specified width and height.
        """
        assert isinstance(pos, position)
        assert isinstance(frame_size, size)
        origin_x = int((pos.x / resolution) - frame_size.w/2)
        origin_y = int((pos.y / resolution) - frame_size.h/2)
        origin_z = int((pos.z / resolution) - frame_size.d/2)

        frame_origin = origin(origin_x, origin_y, origin_z)

        return cls(frame_origin, frame_size, resolution)

class gridMap:
    def __init__(self, gridFrame: frame, data: Union[np.ndarray, cp.ndarray]):
        """
        Origin, width, and height are in grid cells
        Resolution is in meters per grid cell
        """
        self.frame = gridFrame
        self.data = data

    @property
    def isGPU(self) -> bool:
        return isinstance(self.data, cp.ndarray)
    
    @property
    def isBool(self) -> bool:
        return self.data.dtype == bool

    def toCPU(self) -> 'gridMap':
        if self.isGPU:
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
        assert self.frame == otherGM.frame
        assert self.isBool
        assert otherGM.isBool
        if self.isGPU:
            return gridMap(self.frame, cp.logical_and(self.data, cp.logical_not(otherGM.data)))
        return gridMap(self.frame, np.logical_and(self.data, np.logical_not(otherGM.data)))
    
    def union(self, otherGM: 'gridMap') -> 'gridMap':
        # Implements the set union between two grid maps
        assert self.frame == otherGM.frame
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
    orig = origin(origin_x, origin_y, 0)
    frame_size = size(width, height, 1)
    currentFrame = frame(orig, frame_size, resolution)

    data = np.zeros((width, height))
    data[0][0] = 1
    data[19][0] = 0.5
    
    grid = gridMap(currentFrame, data)
    grid.drawFilledRectangle(0.0, 2.0, 0.0, 2.0, 1.0, 1.0)
    grid.plot(isPause=True)

    newFrame = frame(origin(10, 0, 0), size(10, 6, 1), 0.5)

    grid.crop(newFrame).plot(isPause=True)

if __name__ == '__main__':
    main()