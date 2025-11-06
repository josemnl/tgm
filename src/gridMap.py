from turtle import width
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import cupy as cp
from typing import Tuple, Union
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class position:
    def __init__(self, x: float, y: float, z: float = 0.0):
        assert isinstance(x, (float, int))
        assert isinstance(y, (float, int))
        assert isinstance(z, (float, int))
        if isinstance(x, int):
            x = float(x)
        if isinstance(y, int):
            y = float(y)
        if isinstance(z, int):
            z = float(z)
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
        assert isinstance(roll, (float, int))
        assert isinstance(pitch, (float, int))
        assert isinstance(yaw, (float, int))
        if isinstance(roll, int):
            roll = float(roll)
        if isinstance(pitch, int):
            pitch = float(pitch)
        if isinstance(yaw, int):
            yaw = float(yaw)
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __add__(self, other):
        if not isinstance(other, orientation):
            return NotImplemented
        two_pi = 2 * np.pi
        r = (self.roll + other.roll) % two_pi
        p = (self.pitch + other.pitch) % two_pi
        y = (self.yaw + other.yaw) % two_pi
        return orientation(r, p, y)
    
    def __sub__(self, other):
        if not isinstance(other, orientation):
            return NotImplemented
        two_pi = 2 * np.pi
        r = (self.roll - other.roll) % two_pi
        p = (self.pitch - other.pitch) % two_pi
        y = (self.yaw - other.yaw) % two_pi
        return orientation(r, p, y)

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
    
    @staticmethod
    def _euler_zyx_to_R(o: 'orientation') -> np.ndarray:
        cr, sr = np.cos(o.roll), np.sin(o.roll)
        cp, sp = np.cos(o.pitch), np.sin(o.pitch)
        cy, sy = np.cos(o.yaw), np.sin(o.yaw)
        # Rz(yaw) * Ry(pitch) * Rx(roll)
        return np.array([
            [cy*cp,                cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,                sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,                  cp*sr,             cp*cr           ]
        ])
    
    @staticmethod
    def R_zyx(o: 'orientation') -> np.ndarray:
        # Alias to your existing implementation
        return pose._euler_zyx_to_R(o)

    @staticmethod
    def _R_to_euler_zyx(R: np.ndarray) -> 'orientation':
        # Robust ZYX extraction with gimbal-lock handling
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        eps = 1e-9
        if sy > eps:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock: yaw set to 0, roll absorbs heading
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)  # = ±pi/2
            yaw = 0.0
        return orientation(roll, pitch, yaw)

    def transform(self, other: 'pose') -> 'pose':
        """Compose poses: result = other ⊕ self (apply 'other' to 'self')."""
        # Rotation/translation of 'other'
        R_other = self._euler_zyx_to_R(other.orientation)
        t_other = np.array([other.position.x, other.position.y, other.position.z])

        # Rotate and translate position
        p_self = np.array([self.position.x, self.position.y, self.position.z])
        p_out = R_other @ p_self + t_other

        # Compose orientations: R_out = R_other @ R_self
        R_self = self._euler_zyx_to_R(self.orientation)
        R_out = R_other @ R_self
        o_out = self._R_to_euler_zyx(R_out)

        return pose(position(p_out[0], p_out[1], p_out[2]), o_out)
    
    def compose(self, right: 'pose') -> 'pose':
        """
        Return self ⊕ right (apply self, then right). Matches matrix A @ B semantics.
        Implemented via the existing left-multiply transform to stay consistent.
        """
        return right.transform(self)

    def __matmul__(self, right: 'pose') -> 'pose':
        """
        Python @ operator: A @ B == A ⊕ B
        """
        return self.compose(right)

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
        self.origin = frame_origin
        self.size = frame_size
        self.r = resolution

    def __eq__(self, other):
        if not isinstance(other, frame):
            return NotImplemented
        return self.origin.x == other.origin.x and self.origin.y == other.origin.y and self.origin.z == other.origin.z and \
               self.size.w == other.size.w and self.size.h == other.size.h and self.size.d == other.size.d and \
               self.r == other.r

    def contains(self, other: 'frame') -> bool:
        return self.origin.x <= other.origin.x and \
               self.origin.y <= other.origin.y and \
               self.origin.z <= other.origin.z and \
               self.origin.x + self.size.w >= other.origin.x + other.size.w and \
               self.origin.y + self.size.h >= other.origin.y + other.size.h and \
               self.origin.z + self.size.d >= other.origin.z + other.size.d

    def computeOverlap(self, other: 'frame') -> 'frame':
        """
        Compute the overlap between this frame and another frame.
        """
        overlap_origin_x = max(self.origin.x, other.origin.x)
        overlap_origin_y = max(self.origin.y, other.origin.y)
        overlap_origin_z = max(self.origin.z, other.origin.z)
        overlap_width = min(self.origin.x + self.size.w, other.origin.x + other.size.w) - overlap_origin_x
        overlap_height = min(self.origin.y + self.size.h, other.origin.y + other.size.h) - overlap_origin_y
        overlap_depth = min(self.origin.z + self.size.d, other.origin.z + other.size.d) - overlap_origin_z

        overlap_origin = origin(overlap_origin_x, overlap_origin_y, overlap_origin_z)
        overlap_size = size(overlap_width, overlap_height, overlap_depth)

        return frame(overlap_origin, overlap_size, self.r)
    
    def world_to_idx(self, x: float, y: float, z: float = 0.0) -> Tuple[int, int, int]:
        """
        Convert world coordinates (in meters) to grid indices (in cells).
        """
        ix = int(np.round((x / self.r) - self.origin.x))
        iy = int(np.round((y / self.r) - self.origin.y))
        iz = int(np.round((z / self.r) - self.origin.z))
        return ix, iy, iz
    
    def in_bounds(self, ix: int, iy: int, iz: int = 0) -> bool:
        """
        Check if the given grid indices are within the bounds of the frame.
        """
        return (0 <= ix < self.size.w) and (0 <= iy < self.size.h) and (0 <= iz < self.size.d)

    @property
    def is2D(self) -> bool:
        return self.size.d == 1

    @property
    def is3D(self) -> bool:
        return self.size.d > 1

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

        # Assert that the data is a 2D or 3D array
        assert isinstance(data, (np.ndarray, cp.ndarray))
        assert data.ndim in [2, 3]
        assert data.shape[0] == gridFrame.size.w
        assert data.shape[1] == gridFrame.size.h
        if data.ndim == 2:
            # Expand to 3D array with depth 1
            data = data[:, :, np.newaxis]
        assert data.shape[2] == gridFrame.size.d

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
    
    def toGPU(self) -> 'gridMap':
        if not self.isGPU:
            return gridMap(self.frame, cp.asarray(self.data))
        return self
    
    def toBool(self, threshold: float) -> 'gridMap':
        return gridMap(self.frame, self.data > threshold)

    def plot(self, layer: int = 0, isPause: bool = False) -> None:
        I = 1 - np.transpose(self.data[:, :, layer])
        plt.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.frame.origin.x*self.frame.r, (self.frame.origin.x + self.frame.size.w)*self.frame.r,
                           self.frame.origin.y*self.frame.r, (self.frame.origin.y + self.frame.size.h)*self.frame.r))
        plt.show(block=isPause)
        plt.pause(0.0001)

    def savePNG(self, layer: int, filename: str) -> None:
        I = 1 - np.transpose(self.data[:, :, layer])
        plt.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.frame.origin.x*self.frame.r, (self.frame.origin.x + self.frame.size.w)*self.frame.r,
                           self.frame.origin.y*self.frame.r, (self.frame.origin.y + self.frame.size.h)*self.frame.r))
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
        x0 = newFrame.origin.x - self.frame.origin.x
        y0 = newFrame.origin.y - self.frame.origin.y
        z0 = newFrame.origin.z - self.frame.origin.z
        x1 = x0 + newFrame.size.w
        y1 = y0 + newFrame.size.h
        z1 = z0 + newFrame.size.d
        return gridMap(newFrame, self.data[x0:x1, y0:y1, z0:z1])

    def reshape(self, newFrame: frame, fill_value: float) -> 'gridMap':
        """
        Reshape the grid map.
        If the new grid is partially outside the old one, the new cells are initialized with the fill value.
        """
        overlap = self.computeOverlap(newFrame)
        if self.isGPU:
            newData = cp.full((newFrame.size.w, newFrame.size.h, newFrame.size.d), fill_value)
        else:
            newData = np.full((newFrame.size.w, newFrame.size.h, newFrame.size.d), fill_value)
        ix_0 = overlap.origin.x - newFrame.origin.x
        iy_0 = overlap.origin.y - newFrame.origin.y
        iz_0 = overlap.origin.z - newFrame.origin.z
        ix_1 = ix_0 + overlap.size.w - 1
        iy_1 = iy_0 + overlap.size.h - 1
        iz_1 = iz_0 + overlap.size.d - 1
        nx_0 = overlap.origin.x - self.frame.origin.x
        ny_0 = overlap.origin.y - self.frame.origin.y
        nz_0 = overlap.origin.z - self.frame.origin.z
        nx_1 = nx_0 + overlap.size.w - 1
        ny_1 = ny_0 + overlap.size.h - 1
        nz_1 = nz_0 + overlap.size.d - 1

        newData[ix_0:ix_1, iy_0:iy_1, iz_0:iz_1] = self.data[nx_0:nx_1, ny_0:ny_1, nz_0:nz_1]
        return gridMap(newFrame, newData)

    def occupancy(self, x: float, y: float, z: float = 0) -> float:
        ix = np.round((x - self.frame.origin.x*self.frame.r)/self.frame.r).astype(int)
        iy = np.round((y - self.frame.origin.y*self.frame.r)/self.frame.r).astype(int)
        iz = np.round((z - self.frame.origin.z*self.frame.r)/self.frame.r).astype(int)
        return self.data[ix][iy][iz]

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
        rotated_corners[:, 0] = (rotated_corners[:, 0] - self.frame.origin.x*self.frame.r)/self.frame.r
        rotated_corners[:, 1] = (rotated_corners[:, 1] - self.frame.origin.y*self.frame.r)/self.frame.r

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

    def plot3D_scatter(self, isPause: bool = False, s_min: float = 120.0, s_max: float = 120.0,
                   alpha_min: float = 0.0, alpha_max: float = 1.0, elev: float = 20, azim: float = -60,
                   value_min: float = 0.0, value_max: float = 1.0) -> None:
        """
        3D scatter representation: place a marker at the center of each voxel cell.
        - Color = grayscale 1 - value (imshow-like)
        - Alpha scales with occupancy (alpha_min..alpha_max)
        - Marker size scales with occupancy (s_min..s_max) to hint density
        - Only plot values in [value_min..value_max] range (default 0..1)

        Drawn per z-slice back-to-front to improve blending.
        """
        data = cp.asnumpy(self.data) if self.isGPU else self.data
        values = np.clip(data.astype(float), 0.0, 1.0)
        inten = 1.0 - values
        alpha = alpha_min + (alpha_max - alpha_min) * values
        sizes = s_min + (s_max - s_min) * values

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        try:
            ax.set_proj_type('ortho')
        except Exception:
            pass
        ax.view_init(elev=elev, azim=azim)

        # Compute centers in index units (can scale to meters if desired)
        xs = self.frame.origin.x + np.arange(self.frame.size.w) + 0.5
        ys = self.frame.origin.y + np.arange(self.frame.size.h) + 0.5
        zs = self.frame.origin.z + np.arange(self.frame.size.d) + 0.5

        # Shapes
        nx, ny, nz = len(xs), len(ys), len(zs)

        # 1D coordinates consistent with arr.ravel(order='C') for shape (nx, ny, nz)
        X = np.repeat(xs, ny * nz)
        Y = np.tile(np.repeat(ys, nz), nx)
        Z = np.tile(zs, nx * ny)

        # Per-point RGBA and sizes
        rgba = np.zeros(inten.shape + (4,), dtype=float)
        rgba[..., 0] = inten
        rgba[..., 1] = inten
        rgba[..., 2] = inten
        rgba[..., 3] = alpha

        # Mask out low and high values
        mask = (values >= value_min) & (values <= value_max)
        mask = mask.ravel()

        ax.scatter(X[mask], Y[mask], Z[mask],
                   s=sizes.ravel()[mask],
                   c=rgba.reshape(-1, 4)[mask],
                   marker='o',
                   depthshade=False)

        # Limits & aspect
        ax.set_xlim(self.frame.origin.x, self.frame.origin.x + self.frame.size.w)
        ax.set_ylim(self.frame.origin.y, self.frame.origin.y + self.frame.size.h)
        ax.set_zlim(self.frame.origin.z, self.frame.origin.z + self.frame.size.d)
        try:
            ax.set_box_aspect((self.frame.size.w, self.frame.size.h, self.frame.size.d))
        except Exception:
            try:
                ax.set_aspect('equal')
            except Exception:
                pass

        tick_interval = max(1, int(round(1 / self.frame.r))) if self.frame.r > 0 else 1
        ax.set_xticks(np.arange(self.frame.origin.x, self.frame.origin.x + self.frame.size.w + 1, tick_interval))
        ax.set_yticks(np.arange(self.frame.origin.y, self.frame.origin.y + self.frame.size.h + 1, tick_interval))
        ax.set_zticks(np.arange(self.frame.origin.z, self.frame.origin.z + self.frame.size.d + 1, tick_interval))

        plt.show(block=isPause)
        plt.pause(0.0001)

    def plot3D_cubes(self, isPause: bool = False, cube_size: float = 1.0,
                   alpha_min: float = 0.0, alpha_max: float = 1.0,
                   face_edges: bool = False, elev: float = 20, azim: float = -60) -> None:
        """
        3D glyph scatter with cubes: draw a smaller cube centered in each cell.
        - cube_size in (0, 1]: side-length relative to cell size (default 0.6)
        - Color = 1 - value (imshow-like grayscale)
        - Alpha scales with occupancy (alpha_min..alpha_max)
        - Optional edges (off by default for speed)

        Renders per z-slice back-to-front for decent transparency blending.
        """
        data = cp.asnumpy(self.data) if self.isGPU else self.data
        values = np.clip(data.astype(float), 0.0, 1.0)
        inten = 1.0 - values
        alpha = alpha_min + (alpha_max - alpha_min) * values

        half = float(cube_size) / 2.0
        # Clamp to (0, 0.5]; 0.5 means cubes span the full cell and touch
        if half <= 0:
            half = 1e-3
        if half > 0.5:
            half = 0.5

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        try:
            ax.set_proj_type('ortho')
        except Exception:
            pass
        ax.view_init(elev=elev, azim=azim)

        def cube_faces(cx, cy, cz, h):
            x0, x1 = cx - h, cx + h
            y0, y1 = cy - h, cy + h
            z0, z1 = cz - h, cz + h
            return [
                [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
                [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
                [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
                [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
                [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
                [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
            ]

        # Centers per axis
        xs = self.frame.origin.x + np.arange(self.frame.size.w) + 0.5
        ys = self.frame.origin.y + np.arange(self.frame.size.h) + 0.5
        zs = self.frame.origin.z + np.arange(self.frame.size.d) + 0.5

        for k in range(self.frame.size.d):
            zc = zs[k]
            polys = []
            face_cols = []
            edge_cols = []
            for i in range(self.frame.size.w):
                xc = xs[i]
                for j in range(self.frame.size.h):
                    yc = ys[j]
                    col = inten[i, j, k]
                    a = alpha[i, j, k]
                    faces = cube_faces(xc, yc, zc, half)
                    rgba = (col, col, col, a)
                    for f in faces:
                        polys.append(f)
                        face_cols.append(rgba)
                        edge_cols.append((0, 0, 0, 0.25) if face_edges else (0, 0, 0, 0))
            if polys:
                coll = Poly3DCollection(polys, facecolors=face_cols, edgecolors=edge_cols)
                coll.set_alpha(None)
                try:
                    coll.set_zsort('none')
                except Exception:
                    pass
                ax.add_collection3d(coll)

        ax.set_xlim(self.frame.origin.x, self.frame.origin.x + self.frame.size.w)
        ax.set_ylim(self.frame.origin.y, self.frame.origin.y + self.frame.size.h)
        ax.set_zlim(self.frame.origin.z, self.frame.origin.z + self.frame.size.d)
        try:
            ax.set_box_aspect((self.frame.size.w, self.frame.size.h, self.frame.size.d))
        except Exception:
            try:
                ax.set_aspect('equal')
            except Exception:
                pass

        tick_interval = max(1, int(round(1 / self.frame.r))) if self.frame.r > 0 else 1
        ax.set_xticks(np.arange(self.frame.origin.x, self.frame.origin.x + self.frame.size.w + 1, tick_interval))
        ax.set_yticks(np.arange(self.frame.origin.y, self.frame.origin.y + self.frame.size.h + 1, tick_interval))
        ax.set_zticks(np.arange(self.frame.origin.z, self.frame.origin.z + self.frame.size.d + 1, tick_interval))

        plt.show(block=isPause)
        plt.pause(0.0001)

    @classmethod
    def loadState(cls, filename: str, data_type: np.dtype = np.float64) -> 'gridMap':
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
            obj.data = obj.data.astype(data_type)
            return obj

def main() -> None:
    width = 10*2
    height = 5*2
    resolution = 0.5
    orig = origin(0, 0, 0)
    frame_size = size(width, height, 2)
    currentFrame = frame(orig, frame_size, resolution)

    data = np.zeros((width, height, 2))+0.01
    data[0][0][0] = 1
    data[19][0][1] = 0.5
    
    grid = gridMap(currentFrame, data)
    grid.drawFilledRectangle(0.0, 2.0, 0.0, 2.0, 1.0, 1.0)
    grid.plot(0, isPause=True)
    grid.plot3D_scatter(isPause=True) # Balls
    grid.plot3D_cubes(isPause=True)

    newFrame = frame(origin(10, 0, 0), size(10, 6, 2), 0.5)
    
    grid.crop(newFrame).plot(isPause=True)
    grid.crop(newFrame).plot3D_scatter(isPause=True)
    grid.crop(newFrame).plot3D_cubes(isPause=True)

    # Test pose transformations
    p1 = pose(position(1.0, 0.0, 0.0), orientation(0.0, 0.0, np.pi/2))
    p2 = pose(position(3.0, 0.0, 0.0), orientation(0.0, 0.0, np.pi/4))
    p3 = p2.transform(p1)

    print(f"Transformed Position: x={p3.position.x}, y={p3.position.y}, z={p3.position.z}")
    print(f"Transformed Orientation: roll={p3.orientation.roll}, pitch={p3.orientation.pitch}, yaw={p3.orientation.yaw}")

    x_t = pose(position(2.5, 2.5, 2.5), orientation(0.0, 0.0, 0.0))
    link_base_sensor = pose(position(0.22, 0.0, -0.15), orientation(0.0, -3.14159/6, 0.0))
    x_t = link_base_sensor.transform(x_t)

    print(f"Transformed Position: x={x_t.position.x}, y={x_t.position.y}, z={x_t.position.z}")
    print(f"Transformed Orientation: roll={x_t.orientation.roll}, pitch={x_t.orientation.pitch}, yaw={x_t.orientation.yaw}")

if __name__ == '__main__':
    main()