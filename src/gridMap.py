import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import cupy as cp
from typing import Union
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spatial import position, orientation, pose, frame, origin, size

class discreteDist:
    """
    Base class for discrete probability distributions.
    """
    def __init__(self, probabilities: np.ndarray, values: np.ndarray = None):
        # Set attributes
        self.probabilities = probabilities
        if values is None:
            self.values = np.arange(len(probabilities))
        else:
            self.values = values

    def expected_value(self) -> float:
        """
        Compute the expected value of the distribution.
        E[S] = sum(s * p(S = s)) for s = 0..N
        """
        return np.sum(self.values * self.probabilities)
    
    def updateLikelihood(self, likelihoods: np.ndarray) -> 'discreteDist':
        """
        Update the distribution with new likelihoods using Bayes' rule.
        p_new(S = s) = p_old(S = s) * p(likelihood | S = s) / normalization
        """
        updated_probs = self.probabilities * likelihoods
        normalization = np.sum(updated_probs)
        if normalization > 0:
            updated_probs /= normalization
        return discreteDist(updated_probs, self.values)
    
    def normalize(self) -> 'discreteDist':
        normalization = np.sum(self.probabilities)
        if normalization > 0:
            normalized_probs = self.probabilities / normalization
        else:
            normalized_probs = self.probabilities
        return discreteDist(normalized_probs, self.values)
    
    def plot(self, fig=None, ax=None) -> None:
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(1, 1, 1)
        ax.clear()
        # Use full-width bars without edges to avoid aliasing gaps in PNGs
        ax.bar(
            self.values,
            self.probabilities,
            width=1.0,
            align='center',
            alpha=1.0,
            edgecolor='none',
            linewidth=0,
            antialiased=False,
        )
        ax.set_title('P(S = s)')
        plt.show(block=False)

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

    def cardinality(self) -> discreteDist:
        """
        Return a vector with the probability of the sum of the occupied cells being equal to each index.
        p(S = s) for s = 0..N where N is the number of cells.
        It is computed as a Poisson Binomial distribution, using a dynamic programming approach.
        """
        if self.isGPU:
            p = self.data.flatten().astype(cp.float64)
            N = int(p.size)
            print(f"Computing cardinality for N={N} cells (GPU, FFT).")

            if N == 0:
                return discreteDist(np.array([1.0], dtype=np.float64))

            polys = []
            one = cp.ones((), dtype=cp.float64)
            for i in range(N):
                pi = p[i]
                polys.append(cp.stack([one - pi, pi]).astype(cp.float64))

            def fft_convolve(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
                total_len = int(a.size + b.size - 1)
                n = 1 << (total_len - 1).bit_length()
                fa = cp.fft.rfft(a, n)
                fb = cp.fft.rfft(b, n)
                fc = fa * fb
                c = cp.fft.irfft(fc, n)
                return c[:total_len]

            while len(polys) > 1:
                new_polys = []
                for i in range(0, len(polys), 2):
                    if i + 1 < len(polys):
                        new_polys.append(fft_convolve(polys[i], polys[i + 1]))
                    else:
                        new_polys.append(polys[i])
                polys = new_polys

            cardinality_gpu = polys[0]
            cardinality_gpu = cp.clip(cardinality_gpu, 0.0, 1.0)
            s = cp.sum(cardinality_gpu)
            if s > 0:
                cardinality_gpu = cardinality_gpu / s

            return discreteDist(cp.asnumpy(cardinality_gpu))
        else:
            p = self.data.flatten()
            N = p.size
            print(f"Computing cardinality for N={N} cells.")
            cardinality = np.zeros(N + 1, dtype=np.float64)
            cardinality[0] = 1.0

            for i in range(N):
                p_i = p[i]
                for s in range(i + 1, 0, -1):
                    cardinality[s] = cardinality[s] * (1 - p_i) + cardinality[s - 1] * p_i
                cardinality[0] = cardinality[0] * (1 - p_i)

        return discreteDist(cardinality)
    
    def logOddShift(self, shift: float) -> 'gridMap':
        """
        Apply a log-odds shift to the occupancy probabilities.
        New probability p' = 1 - 1 / (1 + exp(logit(p) + shift))
        where logit(p) = log(p / (1 - p))
        """
        eps = 1e-12
        if self.isGPU:
            p = cp.clip(self.data, eps, 1.0 - eps)
            logit = cp.log(p / (1 - p))
            logit_shifted = logit + shift
            p_new = 1 - 1 / (1 + cp.exp(logit_shifted))
            return gridMap(self.frame, p_new)
        else:
            p = np.clip(self.data, eps, 1.0 - eps)
            logit = np.log(p / (1 - p))
            logit_shifted = logit + shift
            p_new = 1 - 1 / (1 + np.exp(logit_shifted))
            return gridMap(self.frame, p_new)
        
    def rebalance(self, target_cardinality: discreteDist) -> 'gridMap':
        """
        Rebalance the grid map to match a target cardinality distribution.
        Uses an iterative approach to adjust the value of the shift applied to the log-odds.
        """
        shift_low = -10.0
        shift_high = 10.0
        tolerance = 1e-3
        max_iterations = 20

        expected_target = target_cardinality.expected_value()

        for iteration in range(max_iterations):
            shift_mid = (shift_low + shift_high) / 2.0
            gm_shifted = self.logOddShift(shift_mid)
            cardinality_shifted = gm_shifted.cardinality()
            expected_shifted = cardinality_shifted.expected_value()

            if abs(expected_shifted - expected_target) < tolerance:
                return gm_shifted

            if expected_shifted < expected_target:
                shift_low = shift_mid
            else:
                shift_high = shift_mid

        return self.logOddShift(shift_mid)

    @classmethod
    def loadState(cls, filename: str, data_type: np.dtype = np.float64) -> 'gridMap':
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
            obj.data = obj.data.astype(data_type)
            return obj
        
    @classmethod
    def loadFromPNG(cls, filename: str, gridOrigin: origin, resolution: float) -> 'gridMap':
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image from {filename}")
        img = cv2.flip(img, 0)
        img_normalized = img.astype(np.float32) / 255.0
        # Transpose to have shape (width, height)
        img_normalized = np.transpose(img_normalized)
        width, height = img_normalized.shape
        print(f"Loaded PNG '{filename}' with size: width={width}, height={height}")
        grid_frame = frame(gridOrigin, size(width, height), resolution)
        data = 1.0 - img_normalized
        data_3d = data[:, :, np.newaxis]
        return cls(grid_frame, data_3d)

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
    cardinality = grid.cardinality()

    newFrame = frame(origin(10, 0, 0), size(10, 6, 2), 0.5)
    
    grid.crop(newFrame).plot(isPause=True)
    grid.crop(newFrame).plot3D_scatter(isPause=True)
    grid.crop(newFrame).plot3D_cubes(isPause=True)

    # Test pose transformations
    p1 = pose(position(1.0, 0.0, 0.0), orientation(0.0, 0.0, np.pi/2))
    p2 = pose(position(3.0, 0.0, 0.0), orientation(0.0, 0.0, np.pi/4))
    p3 = p2.compose(p1)

    print(f"Transformed Position: x={p3.position.x}, y={p3.position.y}, z={p3.position.z}")
    print(f"Transformed Orientation: roll={p3.orientation.roll}, pitch={p3.orientation.pitch}, yaw={p3.orientation.yaw}")

    x_t = pose(position(2.5, 2.5, 2.5), orientation(0.0, 0.0, 0.0))
    link_base_sensor = pose(position(0.22, 0.0, -0.15), orientation(0.0, -3.14159/6, 0.0))
    x_t = link_base_sensor.compose(x_t)

    print(f"Transformed Position: x={x_t.position.x}, y={x_t.position.y}, z={x_t.position.z}")
    print(f"Transformed Orientation: roll={x_t.orientation.roll}, pitch={x_t.orientation.pitch}, yaw={x_t.orientation.yaw}")

    # Testing cardinality
    cframe = frame(origin(0, 0, 0), size(2, 2, 1), 1.0)
    cdata = np.array([[[0.0], [0.5]],
                      [[0.0], [1.0]]])
    cgrid = gridMap(cframe, cdata)
    ccardinality = cgrid.cardinality()
    print(f"Cardinality: {ccardinality}")
    target_cardinality = discreteDist(np.array([0.0, 0.0, 1.0, 0.0]))
    rebalanced_grid = cgrid.rebalance(target_cardinality)
    print(f"Rebalanced Grid Data:\n{rebalanced_grid.data}")

if __name__ == '__main__':
    main()