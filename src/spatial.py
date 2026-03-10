import numpy as np
from typing import Optional, Tuple

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
    
    def to_R_zyx(self) -> np.ndarray:
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        # Rz(yaw) * Ry(pitch) * Rx(roll)
        return np.array([
            [cy*cp,                cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,                sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,                  cp*sr,             cp*cr           ]
        ])
    
    @staticmethod
    def from_R_zyx(R: np.ndarray) -> 'orientation':
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        eps = 1e-9
        if sy > eps:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)  # ≈ ±pi/2
            yaw = 0.0
        return orientation(roll, pitch, yaw)

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

    def compose(self, right: 'pose') -> 'pose':
        """
        Return self ⊕ right (apply self, then right).
        """
        R_self = self.orientation.to_R_zyx()
        t_self = np.array([self.position.x, self.position.y, self.position.z])

        R_right = right.orientation.to_R_zyx()
        t_right = np.array([right.position.x, right.position.y, right.position.z])

        R_out = R_self @ R_right
        t_out = R_self @ t_right + t_self

        o_out = orientation.from_R_zyx(R_out)
        p_out = position(float(t_out[0]), float(t_out[1]), float(t_out[2]))
        return pose(p_out, o_out)

    def __matmul__(self, right: 'pose') -> 'pose':
        """
        Python @ operator: A @ B == A ⊕ B (apply A, then B)
        """
        return self.compose(right)

class covariance:
    def __init__(self, matrix: np.ndarray):
        assert isinstance(matrix, np.ndarray)
        assert matrix.ndim == 2
        assert matrix.shape == (6, 6)

        mat = np.array(matrix, dtype=float, copy=True)
        assert np.all(np.isfinite(mat)), "Covariance contains non-finite values"

        # Numerical operations can introduce tiny asymmetries; enforce symmetry.
        self._P = 0.5 * (mat + mat.T)

    def as_array(self) -> np.ndarray:
        return self._P

class poseWithCovariance:
    def __init__(self, x_t: pose, cov: covariance):
        assert isinstance(x_t, pose)
        assert isinstance(cov, covariance)
        self.pose = x_t
        self.covariance = cov

    def plot2D(self, ax, **kwargs):
        # Plot ellipse representing 95% confidence interval in x-y plane
        from matplotlib.patches import Ellipse
        cov_xy = self.covariance._P[0:2, 0:2]
        eigvals, eigvecs = np.linalg.eigh(cov_xy)
        width, height = 2 * np.sqrt(eigvals)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0]) * 180 / np.pi
        ellipse = Ellipse(xy=(self.pose.position.x, self.pose.position.y), width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellipse)
        # Force plot to redraw to show the ellipse
        ax.figure.canvas.draw()

    def plot3D(self, ax, **kwargs):
        # Plot ellipsoid representing 95% confidence interval in x-y-z space
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        cov = self.covariance._P[0:3, 0:3]
        eigvals, eigvecs = np.linalg.eigh(cov)
        radii = 2 * np.sqrt(eigvals)
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = eigvecs @ [x[i, j], y[i, j], z[i, j]]
                x[i, j] += self.pose.position.x
                y[i, j] += self.pose.position.y
                z[i, j] += self.pose.position.z
        verts = [list(zip(x.flatten(), y.flatten(), z.flatten()))]
        ax.add_collection3d(Poly3DCollection(verts, **kwargs))
        ax.figure.canvas.draw()

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
        assert self.r == other.r, "Cannot compute overlap of frames with different resolutions"
        overlap_origin_x = max(self.origin.x, other.origin.x)
        overlap_origin_y = max(self.origin.y, other.origin.y)
        overlap_origin_z = max(self.origin.z, other.origin.z)
        overlap_width = min(self.origin.x + self.size.w, other.origin.x + other.size.w) - overlap_origin_x
        overlap_height = min(self.origin.y + self.size.h, other.origin.y + other.size.h) - overlap_origin_y
        overlap_depth = min(self.origin.z + self.size.d, other.origin.z + other.size.d) - overlap_origin_z

        overlap_origin = origin(overlap_origin_x, overlap_origin_y, overlap_origin_z)
        overlap_size = size(overlap_width, overlap_height, overlap_depth)

        return frame(overlap_origin, overlap_size, self.r)
    
    def computeUnion(self, other: 'frame') -> 'frame':
        """
        Compute the union between this frame and another frame.
        """
        assert self.r == other.r, "Cannot compute union of frames with different resolutions"
        union_origin_x = min(self.origin.x, other.origin.x)
        union_origin_y = min(self.origin.y, other.origin.y)
        union_origin_z = min(self.origin.z, other.origin.z)
        union_width = max(self.origin.x + self.size.w, other.origin.x + other.size.w) - union_origin_x
        union_height = max(self.origin.y + self.size.h, other.origin.y + other.size.h) - union_origin_y
        union_depth = max(self.origin.z + self.size.d, other.origin.z + other.size.d) - union_origin_z

        union_origin = origin(union_origin_x, union_origin_y, union_origin_z)
        union_size = size(union_width, union_height, union_depth)

        return frame(union_origin, union_size, self.r)
    
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