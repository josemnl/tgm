from __future__ import annotations

import numpy as np
from typing import Any
from .cupy_compat import require_cupy
from .gridMap import gridMap
from .spatial import frame, origin, size, pose, position, orientation
from .lidarScan import lidarScan, lidarScan3D
import time
from .utilities import read3DLidarCSV

class sensorModel:
    def __init__ (self, smFrame: frame, sensorRange, invModel ,occPrior):
        assert isinstance(smFrame, frame)
        self.frame = smFrame
        self.sensorRange = sensorRange
        self.invModel = invModel
        self.occPrior = occPrior
        self.data = np.ones((self.frame.size.w, self.frame.size.h)) * self.occPrior

    def updateBasedOnPose(self, x_t: pose):
        x_t = np.array([x_t.position.x, x_t.position.y, x_t.orientation.yaw])
        self.frame.origin = origin(int((x_t[0] / self.frame.r) - (self.frame.size.w/2)), int((x_t[1] / self.frame.r) - (self.frame.size.h/2)), 0)

    def generateGridMap(self, z_t, x_t: pose, z_t_ground=None, rayTraceGround = True):
        x_t = np.array([x_t.position.x, x_t.position.y, x_t.orientation.yaw])
        timeStart = time.time()
        assert isinstance(z_t, lidarScan)
        assert isinstance(z_t_ground, lidarScan) or z_t_ground is None
        ang, dist = z_t.angles, z_t.ranges

        # Update measurement orientation with agent's pose
        ang = ang + x_t[2]
        timePose = time.time()

        # Remove measurements further than sensor range
        mask = dist < self.sensorRange * self.frame.r
        ang = ang[mask]
        dist = dist[mask]
        timeClip = time.time()

        # Compute detection points on global frame
        ox = x_t[0] + np.cos(ang) * dist
        oy = x_t[1] + np.sin(ang) * dist
        timeGlobal = time.time()

        # Compute ground points on global frame
        if z_t_ground is not None:
            ang_ground, dist_ground = z_t_ground.angles, z_t_ground.ranges
            np.add(ang_ground, x_t[2], out=ang_ground)
            mask = dist_ground < self.sensorRange * self.frame.r
            ang_ground = ang_ground[mask]
            dist_ground = dist_ground[mask]
            ox_ground = x_t[0] + np.cos(ang_ground) * dist_ground
            oy_ground = x_t[1] + np.sin(ang_ground) * dist_ground
        timeGround = time.time()

        # Compute matrix index for ego pose
        ix_t = ((x_t[0:2] / self.frame.r) - (self.frame.origin.x, self.frame.origin.y)).astype(int)

        # Initialize matrix with prior
        self.data.fill(self.occPrior)
        timeInit = time.time()

        # Compute matrix indices for detections
        ix = np.round((ox / self.frame.r) - self.frame.origin.x).astype(int)
        iy = np.round((oy / self.frame.r) - self.frame.origin.y).astype(int)

        # Filter out-of-bounds detections
        valid = (ix >= 0) & (ix < self.data.shape[0]) & (iy >= 0) & (iy < self.data.shape[1])
        ix = ix[valid]
        iy = iy[valid]

        # Mark free cells along the rays
        for i in range(ix.size):
            self.insertRay((ix_t[0], ix_t[1]), (ix[i], iy[i]), self.invModel[0])
        timeFree = time.time()

        # If ground points are provided and rayTraceGround is false, mark free cells
        if z_t_ground is not None and not rayTraceGround:
            # Compute the matrix indices for ground points
            ix_ground = np.round((ox_ground / self.frame.r) - self.frame.origin.x).astype(int)
            iy_ground = np.round((oy_ground / self.frame.r) - self.frame.origin.y).astype(int)
            # Filter out-of-bounds ground points
            valid = (ix_ground >= 0) & (ix_ground < self.data.shape[0]) & (iy_ground >= 0) & (iy_ground < self.data.shape[1])
            ix_ground = ix_ground[valid]
            iy_ground = iy_ground[valid]
            # Mark free cells on the ground
            self.data[ix_ground, iy_ground] = self.invModel[0]
        timeGroundFree = time.time()

        # Mark cells in between detections as unknown
        for i in range(ix.size):
            self.insertRay((ix[i], iy[i]), (ix[i-1], iy[i-1]), self.occPrior)
        timeUnknown = time.time()
        
        # Mark occupied cells
        self.data[ix, iy] = self.invModel[1]
        timeOccupied = time.time()

        # If ground points are provided and rayTraceGround is true, mark free cells along the rays
        if z_t_ground is not None and rayTraceGround:
            # Compute the matrix indices for ground points
            ix_ground = np.round((ox_ground / self.frame.r) - self.frame.origin.x).astype(int)
            iy_ground = np.round((oy_ground / self.frame.r) - self.frame.origin.y).astype(int)
            # Filter out-of-bounds ground points
            valid = (ix_ground >= 0) & (ix_ground < self.data.shape[0]) & (iy_ground >= 0) & (iy_ground < self.data.shape[1])
            ix_ground = ix_ground[valid]
            iy_ground = iy_ground[valid]
            # Mark free cells along the rays
            for i in range(ix_ground.size):
                self.insertRay((ix_t[0], ix_t[1]), (ix_ground[i], iy_ground[i]), self.invModel[0], self.invModel[1])

        '''
        print("Times sensor model:")
        print("Pose: " + str(timePose - timeStart))
        print("Clip: " + str(timeClip - timePose))
        print("Global: " + str(timeGlobal - timeClip))
        print("Ground: " + str(timeGround - timeGlobal))
        print("Init: " + str(timeInit - timeGround))
        print("Free: " + str(timeFree - timeInit))
        print("Ground Free: " + str(timeGroundFree - timeFree))
        print("Unknown: " + str(timeUnknown - timeGroundFree))
        print("Occupied: " + str(timeOccupied - timeUnknown))
        print("Total: " + str(timeOccupied - timeStart))
        print("")
        '''

        gridOrigin = origin(self.frame.origin.x, self.frame.origin.y, 0)
        gridSize = size(self.frame.size.w, self.frame.size.h, 1)

        gridFrame = frame(gridOrigin, gridSize, self.frame.r)

        return gridMap(gridFrame, self.data)

    def insertRay(self,start,end,value, valueCondition = None):
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        is_steep = abs(dy) > abs(dx)

        if is_steep:
            if dy == 0:
                y_coords = np.array([y1])
                step = 1 if x2 > x1 else -1
                x_coords = np.arange(x1, x2 + step, step)
            else:
                step = 1 if y2 > y1 else -1
                y_coords = np.arange(y1, y2 + step, step)
                # Calculate x_coords using integer division to avoid floating points
                x_coords = np.floor(x1 + (dx * (y_coords - y1) / dy)).astype(int)
        else:
            if dx == 0:
                x_coords = np.array([x1])
                step = 1 if y2 > y1 else -1
                y_coords = np.arange(y1, y2 + step, step)
            else:
                step = 1 if x2 > x1 else -1
                x_coords = np.arange(x1, x2 + step, step)
                # Calculate y_coords using integer division to avoid floating points
                y_coords = np.floor(y1 + (dy * (x_coords - x1) / dx)).astype(int)

        if valueCondition is None:
            self.data[x_coords, y_coords] = value
        else:
            # Check if all cells in the ray are different from the valueCondition
            if np.all(self.data[x_coords, y_coords] != valueCondition):
                self.data[x_coords, y_coords] = value

class sensorModelGPU:
    def __init__(self, smFrame: frame, sensorRange, invModel, occPrior):
        require_cupy("sensorModelGPU")
        import cupy as cp
        self.cp = cp
        assert isinstance(smFrame, frame)
        self.frame = smFrame
        self.sensorRange = sensorRange
        self.invModel = invModel
        self.occPrior = occPrior
        self.data = cp.ones((self.frame.size.w, self.frame.size.h), dtype=cp.float32) * cp.float32(self.occPrior)

        # Each thread carves one 2D ray. mode=0 writes unconditionally, mode=1 writes only
        # when no cell on the ray matches value_condition.
        self._carve_rays_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void carve_rays(const int N,
                        const int* __restrict__ sx,
                        const int* __restrict__ sy,
                        const int* __restrict__ ex,
                        const int* __restrict__ ey,
                        float* __restrict__ grid,
                        const int W,
                        const int H,
                        const float value,
                        const int mode,
                        const float value_condition)
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= N) return;

            int x1 = sx[i], y1 = sy[i];
            int x2 = ex[i], y2 = ey[i];
            int dx = x2 - x1;
            int dy = y2 - y1;
            int adx = dx >= 0 ? dx : -dx;
            int ady = dy >= 0 ? dy : -dy;

            bool blocked = false;

            if (ady > adx) {
                if (dy == 0) {
                    if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) {
                        int idx = x1 * H + y1;
                        if (mode == 1 && grid[idx] == value_condition) {
                            blocked = true;
                        }
                    }
                    if (!blocked && x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) {
                        int idx = x1 * H + y1;
                        grid[idx] = value;
                    }
                    return;
                }

                int step = (y2 > y1) ? 1 : -1;

                if (mode == 1) {
                    for (int y = y1;; y += step) {
                        double rx = ((double)dx * (double)(y - y1)) / (double)dy;
                        int x = (int)floor((double)x1 + rx);
                        if (x >= 0 && x < W && y >= 0 && y < H) {
                            int idx = x * H + y;
                            if (grid[idx] == value_condition) {
                                blocked = true;
                                break;
                            }
                        }
                        if (y == y2) break;
                    }
                }

                if (!blocked) {
                    for (int y = y1;; y += step) {
                        double rx = ((double)dx * (double)(y - y1)) / (double)dy;
                        int x = (int)floor((double)x1 + rx);
                        if (x >= 0 && x < W && y >= 0 && y < H) {
                            int idx = x * H + y;
                            grid[idx] = value;
                        }
                        if (y == y2) break;
                    }
                }
            } else {
                if (dx == 0) {
                    if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) {
                        int idx = x1 * H + y1;
                        if (mode == 1 && grid[idx] == value_condition) {
                            blocked = true;
                        }
                    }
                    if (!blocked && x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) {
                        int idx = x1 * H + y1;
                        grid[idx] = value;
                    }
                    return;
                }

                int step = (x2 > x1) ? 1 : -1;

                if (mode == 1) {
                    for (int x = x1;; x += step) {
                        double ry = ((double)dy * (double)(x - x1)) / (double)dx;
                        int y = (int)floor((double)y1 + ry);
                        if (x >= 0 && x < W && y >= 0 && y < H) {
                            int idx = x * H + y;
                            if (grid[idx] == value_condition) {
                                blocked = true;
                                break;
                            }
                        }
                        if (x == x2) break;
                    }
                }

                if (!blocked) {
                    for (int x = x1;; x += step) {
                        double ry = ((double)dy * (double)(x - x1)) / (double)dx;
                        int y = (int)floor((double)y1 + ry);
                        if (x >= 0 && x < W && y >= 0 && y < H) {
                            int idx = x * H + y;
                            grid[idx] = value;
                        }
                        if (x == x2) break;
                    }
                }
            }
        }
        ''', 'carve_rays')

    def updateBasedOnPose(self, x_t: pose):
        x_t_np = np.array([x_t.position.x, x_t.position.y, x_t.orientation.yaw])
        self.frame.origin = origin(
            int((x_t_np[0] / self.frame.r) - (self.frame.size.w / 2)),
            int((x_t_np[1] / self.frame.r) - (self.frame.size.h / 2)),
            0,
        )

    def _carve_rays(self,
                    sx: Any,
                    sy: Any,
                    ex: Any,
                    ey: Any,
                    value: float,
                    valueCondition: float | None = None):
        cp = self.cp
        n = int(ex.size)
        if n == 0:
            return
        threads = 256
        blocks = (n + threads - 1) // threads
        mode = cp.int32(1 if valueCondition is not None else 0)
        cond = cp.float32(valueCondition if valueCondition is not None else 0.0)
        self._carve_rays_kernel(
            (blocks,),
            (threads,),
            (
                cp.int32(n),
                sx.astype(cp.int32, copy=False),
                sy.astype(cp.int32, copy=False),
                ex.astype(cp.int32, copy=False),
                ey.astype(cp.int32, copy=False),
                self.data.ravel(),
                cp.int32(self.frame.size.w),
                cp.int32(self.frame.size.h),
                cp.float32(value),
                mode,
                cond,
            ),
        )

    def generateGridMap(self, z_t, x_t: pose, z_t_ground=None, rayTraceGround=True):
        cp = self.cp
        x_t_np = np.array([x_t.position.x, x_t.position.y, x_t.orientation.yaw])
        assert isinstance(z_t, lidarScan)
        assert isinstance(z_t_ground, lidarScan) or z_t_ground is None

        ang = cp.asarray(z_t.angles, dtype=cp.float32) + cp.float32(x_t_np[2])
        dist = cp.asarray(z_t.ranges, dtype=cp.float32)

        mask = dist < cp.float32(self.sensorRange * self.frame.r)
        ang = ang[mask]
        dist = dist[mask]

        ox = cp.float32(x_t_np[0]) + cp.cos(ang) * dist
        oy = cp.float32(x_t_np[1]) + cp.sin(ang) * dist

        if z_t_ground is not None:
            ang_ground = cp.asarray(z_t_ground.angles, dtype=cp.float32) + cp.float32(x_t_np[2])
            dist_ground = cp.asarray(z_t_ground.ranges, dtype=cp.float32)
            mask_ground = dist_ground < cp.float32(self.sensorRange * self.frame.r)
            ang_ground = ang_ground[mask_ground]
            dist_ground = dist_ground[mask_ground]
            ox_ground = cp.float32(x_t_np[0]) + cp.cos(ang_ground) * dist_ground
            oy_ground = cp.float32(x_t_np[1]) + cp.sin(ang_ground) * dist_ground

        ix_t = ((x_t_np[0:2] / self.frame.r) - (self.frame.origin.x, self.frame.origin.y)).astype(int)

        self.data.fill(cp.float32(self.occPrior))

        ix = cp.rint((ox / self.frame.r) - self.frame.origin.x).astype(cp.int32)
        iy = cp.rint((oy / self.frame.r) - self.frame.origin.y).astype(cp.int32)

        valid = (ix >= 0) & (ix < self.data.shape[0]) & (iy >= 0) & (iy < self.data.shape[1])
        ix = ix[valid]
        iy = iy[valid]

        if ix.size > 0:
            sx = cp.full(ix.shape, cp.int32(ix_t[0]), dtype=cp.int32)
            sy = cp.full(iy.shape, cp.int32(ix_t[1]), dtype=cp.int32)
            self._carve_rays(sx, sy, ix, iy, self.invModel[0])

        if z_t_ground is not None and not rayTraceGround:
            ix_ground = cp.rint((ox_ground / self.frame.r) - self.frame.origin.x).astype(cp.int32)
            iy_ground = cp.rint((oy_ground / self.frame.r) - self.frame.origin.y).astype(cp.int32)
            valid_ground = (
                (ix_ground >= 0) & (ix_ground < self.data.shape[0]) &
                (iy_ground >= 0) & (iy_ground < self.data.shape[1])
            )
            ix_ground = ix_ground[valid_ground]
            iy_ground = iy_ground[valid_ground]
            self.data[ix_ground, iy_ground] = cp.float32(self.invModel[0])

        if ix.size > 0:
            prev_ix = cp.roll(ix, 1)
            prev_iy = cp.roll(iy, 1)
            self._carve_rays(ix, iy, prev_ix, prev_iy, self.occPrior)

        self.data[ix, iy] = cp.float32(self.invModel[1])

        if z_t_ground is not None and rayTraceGround:
            ix_ground = cp.rint((ox_ground / self.frame.r) - self.frame.origin.x).astype(cp.int32)
            iy_ground = cp.rint((oy_ground / self.frame.r) - self.frame.origin.y).astype(cp.int32)
            valid_ground = (
                (ix_ground >= 0) & (ix_ground < self.data.shape[0]) &
                (iy_ground >= 0) & (iy_ground < self.data.shape[1])
            )
            ix_ground = ix_ground[valid_ground]
            iy_ground = iy_ground[valid_ground]

            if ix_ground.size > 0:
                sx_ground = cp.full(ix_ground.shape, cp.int32(ix_t[0]), dtype=cp.int32)
                sy_ground = cp.full(iy_ground.shape, cp.int32(ix_t[1]), dtype=cp.int32)
                self._carve_rays(
                    sx_ground,
                    sy_ground,
                    ix_ground,
                    iy_ground,
                    self.invModel[0],
                    valueCondition=self.invModel[1],
                )

        gridOrigin = origin(self.frame.origin.x, self.frame.origin.y, 0)
        gridSize = size(self.frame.size.w, self.frame.size.h, 1)
        gridFrame = frame(gridOrigin, gridSize, self.frame.r)
        return gridMap(gridFrame, self.data)

class sensorModel3D:
    def __init__(self, smFrame: frame, invModel, occPrior: float):
        assert isinstance(smFrame, frame)
        assert smFrame.size.d > 0
        self.frame = smFrame
        self.invModel = invModel                # [free_val, occ_val]
        self.occPrior = occPrior
        self.data = np.ones(
            (self.frame.size.w, self.frame.size.h, self.frame.size.d), dtype=float
        ) * self.occPrior

    def updateBasedOnPose(self, x_t: pose):
        ox = int((x_t.position.x / self.frame.r) - (self.frame.size.w / 2))
        oy = int((x_t.position.y / self.frame.r) - (self.frame.size.h / 2))
        oz = int((x_t.position.z / self.frame.r) - (self.frame.size.d / 2))
        self.frame.origin = origin(ox, oy, oz)

    def generateGridMap(self, z_t: lidarScan3D, x_t: pose) -> gridMap:
        assert isinstance(z_t, lidarScan3D)

        # Reset grid to prior
        self.data.fill(self.occPrior)

        # Transform 3D points from sensor to world using transform method
        z_t_world = z_t.transform(x_t)

        # Start voxel (robot cell)
        sx, sy, sz = self.frame.world_to_idx(x_t.position.x, x_t.position.y, x_t.position.z)
        if not self.frame.in_bounds(sx, sy, sz):
            # Error: sensor origin out of bounds
            raise ValueError("Sensor origin out of bounds of the grid map.")
        # Transform 3D points from world to grid indices using world_to_idx
        grid_pts = np.array([self.frame.world_to_idx(p[0], p[1], p[2]) for p in z_t_world.points3D])

        # Filter out-of-bounds points using in_bounds
        in_bounds_mask = np.array([self.frame.in_bounds(p[0], p[1], p[2]) for p in grid_pts])
        grid_pts = grid_pts[in_bounds_mask]

        # Mark free cells along the rays
        for p in grid_pts:
            ex, ey, ez = p
            self.insertRay3D((sx, sy, sz), (ex, ey, ez), self.invModel[0])

        # Mark occupied endpoints
        self.data[grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2]] = self.invModel[1]

        return gridMap(self.frame, self.data)

    def insertRay3D(self, start: tuple[int, int, int], end: tuple[int, int, int],
                    value: float, valueCondition: float | None = None):
        x1, y1, z1 = start
        x2, y2, z2 = end
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        adx, ady, adz = abs(dx), abs(dy), abs(dz)

        # Choose dominant axis and compute coordinates by proportional mapping with floor
        if adx >= ady and adx >= adz:
            # March in x
            if dx == 0:
                x_coords = np.array([x1], dtype=int)
            else:
                step = 1 if dx > 0 else -1
                x_coords = np.arange(x1, x2 + step, step, dtype=int)
            if dx != 0:
                y_coords = np.floor(y1 + (dy * (x_coords - x1) / dx)).astype(int)
                z_coords = np.floor(z1 + (dz * (x_coords - x1) / dx)).astype(int)
            else:
                y_coords = np.array([y1], dtype=int).repeat(x_coords.size)
                z_coords = np.array([z1], dtype=int).repeat(x_coords.size)
        elif ady >= adx and ady >= adz:
            # March in y
            if dy == 0:
                y_coords = np.array([y1], dtype=int)
            else:
                step = 1 if dy > 0 else -1
                y_coords = np.arange(y1, y2 + step, step, dtype=int)
            if dy != 0:
                x_coords = np.floor(x1 + (dx * (y_coords - y1) / dy)).astype(int)
                z_coords = np.floor(z1 + (dz * (y_coords - y1) / dy)).astype(int)
            else:
                x_coords = np.array([x1], dtype=int).repeat(y_coords.size)
                z_coords = np.array([z1], dtype=int).repeat(y_coords.size)
        else:
            # March in z
            if dz == 0:
                z_coords = np.array([z1], dtype=int)
            else:
                step = 1 if dz > 0 else -1
                z_coords = np.arange(z1, z2 + step, step, dtype=int)
            if dz != 0:
                x_coords = np.floor(x1 + (dx * (z_coords - z1) / dz)).astype(int)
                y_coords = np.floor(y1 + (dy * (z_coords - z1) / dz)).astype(int)
            else:
                x_coords = np.array([x1], dtype=int).repeat(z_coords.size)
                y_coords = np.array([y1], dtype=int).repeat(z_coords.size)

        # Bounds mask
        inb = (
            (0 <= x_coords) & (x_coords < self.frame.size.w) &
            (0 <= y_coords) & (y_coords < self.frame.size.h) &
            (0 <= z_coords) & (z_coords < self.frame.size.d)
        )
        xi, yi, zi = x_coords[inb], y_coords[inb], z_coords[inb]

        if valueCondition is None:
            self.data[xi, yi, zi] = value
        else:
            if np.all(self.data[xi, yi, zi] != valueCondition):
                self.data[xi, yi, zi] = value

class sensorModel3DGPU:
    def __init__(self, smFrame: frame, invModel, occPrior: float,
                 diffusion_radius: int = 0, cone_free: bool = False):
        require_cupy("sensorModel3DGPU")
        import cupy as cp
        self.cp = cp
        assert isinstance(smFrame, frame)
        assert smFrame.size.d > 0
        self.frame = smFrame
        self.invModel = invModel          # [free_val, occ_val]
        self.occPrior = occPrior
        self.diffusion_radius = int(diffusion_radius)
        self.cone_free = bool(cone_free)
        self.data = cp.ones(
            (self.frame.size.w, self.frame.size.h, self.frame.size.d), dtype=cp.float32
        ) * self.occPrior

        # Raw CUDA kernel: each thread carves one ray using dominant-axis parametric stepping
        self._carve_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void carve(int N,
                   const int sx, const int sy, const int sz,
                   const int* __restrict__ ex,
                   const int* __restrict__ ey,
                   const int* __restrict__ ez,
                   float* __restrict__ grid,
                   const int W, const int H, const int D,
                   const float free_val)
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= N) return;
            int x0 = sx, y0 = sy, z0 = sz;
            int x1 = ex[i], y1 = ey[i], z1 = ez[i];

            int dx = x1 - x0;
            int dy = y1 - y0;
            int dz = z1 - z0;
            int adx = dx >= 0 ? dx : -dx;
            int ady = dy >= 0 ? dy : -dy;
            int adz = dz >= 0 ? dz : -dz;

            if (adx >= ady && adx >= adz) {
                int step = (dx > 0) ? 1 : (dx < 0 ? -1 : 0);
                if (step == 0) {
                    if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H && z0 >= 0 && z0 < D) {
                        int idx = ((x0 * H) + y0) * D + z0;
                        grid[idx] = free_val;
                    }
                } else {
                    for (int x = x0;; x += step) {
                        double ry = ((double)dy * (double)(x - x0)) / (double)dx;
                        double rz = ((double)dz * (double)(x - x0)) / (double)dx;
                        int y = (int)floor((double)y0 + ry);
                        int z = (int)floor((double)z0 + rz);
                        if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
                            int idx = ((x * H) + y) * D + z;
                            grid[idx] = free_val;
                        }
                        if (x == x1) break;
                    }
                }
            } else if (ady >= adx && ady >= adz) {
                int step = (dy > 0) ? 1 : (dy < 0 ? -1 : 0);
                if (step == 0) {
                    if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H && z0 >= 0 && z0 < D) {
                        int idx = ((x0 * H) + y0) * D + z0;
                        grid[idx] = free_val;
                    }
                } else {
                    for (int y = y0;; y += step) {
                        double rx = ((double)dx * (double)(y - y0)) / (double)dy;
                        double rz = ((double)dz * (double)(y - y0)) / (double)dy;
                        int x = (int)floor((double)x0 + rx);
                        int z = (int)floor((double)z0 + rz);
                        if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
                            int idx = ((x * H) + y) * D + z;
                            grid[idx] = free_val;
                        }
                        if (y == y1) break;
                    }
                }
            } else {
                int step = (dz > 0) ? 1 : (dz < 0 ? -1 : 0);
                if (step == 0) {
                    if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H && z0 >= 0 && z0 < D) {
                        int idx = ((x0 * H) + y0) * D + z0;
                        grid[idx] = free_val;
                    }
                } else {
                    for (int z = z0;; z += step) {
                        double rx = ((double)dx * (double)(z - z0)) / (double)dz;
                        double ry = ((double)dy * (double)(z - z0)) / (double)dz;
                        int x = (int)floor((double)x0 + rx);
                        int y = (int)floor((double)y0 + ry);
                        if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
                            int idx = ((x * H) + y) * D + z;
                            grid[idx] = free_val;
                        }
                        if (z == z1) break;
                    }
                }
            }
        }
        ''', 'carve')

        # Cone carve with exponential decay toward prior; combine via min()
        self._carve_cone_decay_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void carve_cone_decay(int N,
                              const int sx, const int sy, const int sz,
                              const int* __restrict__ ex,
                              const int* __restrict__ ey,
                              const int* __restrict__ ez,
                              float* __restrict__ grid,
                              const int W, const int H, const int D,
                              const float free_val,
                              const float prior,
                              const int max_r,
                              const float decay)
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= N) return;

            int x0 = sx, y0 = sy, z0 = sz;
            int x1 = ex[i], y1 = ey[i], z1 = ez[i];

            int dx = x1 - x0;
            int dy = y1 - y0;
            int dz = z1 - z0;
            int adx = dx >= 0 ? dx : -dx;
            int ady = dy >= 0 ? dy : -dy;
            int adz = dz >= 0 ? dz : -dz;

            int steps = adx;
            if (ady > steps) steps = ady;
            if (adz > steps) steps = adz;
            if (steps == 0) {
                if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H && z0 >= 0 && z0 < D) {
                    int idx = ((x0 * H) + y0) * D + z0;
                    grid[idx] = fminf(grid[idx], free_val);
                }
                return;
            }

            float inv_decay = (decay > 1e-6f) ? (1.0f / decay) : 1.0f;

            if (adx >= ady && adx >= adz) {
                int step = (dx > 0) ? 1 : -1;
                int k = 0;
                for (int x = x0;; x += step, ++k) {
                    double ry = ((double)dy * (double)(x - x0)) / (double)dx;
                    double rz = ((double)dz * (double)(x - x0)) / (double)dx;
                    int y = (int)floor((double)y0 + ry);
                    int z = (int)floor((double)z0 + rz);

                    int r = (int)floor(((double)max_r * (double)k) / (double)steps);

                    for (int ox = -r; ox <= r; ++ox) {
                        int xx = x + ox;
                        if (xx < 0 || xx >= W) continue;
                        for (int oy = -r; oy <= r; ++oy) {
                            int yy = y + oy;
                            if (yy < 0 || yy >= H) continue;
                            for (int oz = -r; oz <= r; ++oz) {
                                int zz = z + oz;
                                if (zz < 0 || zz >= D) continue;

                                float dist = sqrtf((float)(ox*ox + oy*oy + oz*oz));
                                float w = expf(-dist * inv_decay);
                                float val = prior + (free_val - prior) * w;

                                int idx = ((xx * H) + yy) * D + zz;
                                grid[idx] = fminf(grid[idx], val);
                            }
                        }
                    }
                    if (x == x1) break;
                }
            } else if (ady >= adx && ady >= adz) {
                int step = (dy > 0) ? 1 : -1;
                int k = 0;
                for (int y = y0;; y += step, ++k) {
                    double rx = ((double)dx * (double)(y - y0)) / (double)dy;
                    double rz = ((double)dz * (double)(y - y0)) / (double)dy;
                    int x = (int)floor((double)x0 + rx);
                    int z = (int)floor((double)z0 + rz);

                    int r = (int)floor(((double)max_r * (double)k) / (double)steps);

                    for (int ox = -r; ox <= r; ++ox) {
                        int xx = x + ox;
                        if (xx < 0 || xx >= W) continue;
                        for (int oy = -r; oy <= r; ++oy) {
                            int yy = y + oy;
                            if (yy < 0 || yy >= H) continue;
                            for (int oz = -r; oz <= r; ++oz) {
                                int zz = z + oz;
                                if (zz < 0 || zz >= D) continue;

                                float dist = sqrtf((float)(ox*ox + oy*oy + oz*oz));
                                float w = expf(-dist * inv_decay);
                                float val = prior + (free_val - prior) * w;

                                int idx = ((xx * H) + yy) * D + zz;
                                grid[idx] = fminf(grid[idx], val);
                            }
                        }
                    }
                    if (y == y1) break;
                }
            } else {
                int step = (dz > 0) ? 1 : -1;
                int k = 0;
                for (int z = z0;; z += step, ++k) {
                    double rx = ((double)dx * (double)(z - z0)) / (double)dz;
                    double ry = ((double)dy * (double)(z - z0)) / (double)dz;
                    int x = (int)floor((double)x0 + rx);
                    int y = (int)floor((double)y0 + ry);

                    int r = (int)floor(((double)max_r * (double)k) / (double)steps);

                    for (int ox = -r; ox <= r; ++ox) {
                        int xx = x + ox;
                        if (xx < 0 || xx >= W) continue;
                        for (int oy = -r; oy <= r; ++oy) {
                            int yy = y + oy;
                            if (yy < 0 || yy >= H) continue;
                            for (int oz = -r; oz <= r; ++oz) {
                                int zz = z + oz;
                                if (zz < 0 || zz >= D) continue;

                                float dist = sqrtf((float)(ox*ox + oy*oy + oz*oz));
                                float w = expf(-dist * inv_decay);
                                float val = prior + (free_val - prior) * w;

                                int idx = ((xx * H) + yy) * D + zz;
                                grid[idx] = fminf(grid[idx], val);
                            }
                        }
                    }
                    if (z == z1) break;
                }
            }
        }
        ''', 'carve_cone_decay')

        # Occupied diffusion with exponential decay toward prior; combine via max()
        self._dilate_decay_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void dilate_decay(int N,
                          const int* __restrict__ ex,
                          const int* __restrict__ ey,
                          const int* __restrict__ ez,
                          float* __restrict__ grid,
                          const int W, const int H, const int D,
                          const float occ_val,
                          const float prior,
                          const int r,
                          const float decay)
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= N) return;

            int x1 = ex[i], y1 = ey[i], z1 = ez[i];
            float inv_decay = (decay > 1e-6f) ? (1.0f / decay) : 1.0f;

            for (int ox = -r; ox <= r; ++ox) {
                int x = x1 + ox;
                if (x < 0 || x >= W) continue;
                for (int oy = -r; oy <= r; ++oy) {
                    int y = y1 + oy;
                    if (y < 0 || y >= H) continue;
                    for (int oz = -r; oz <= r; ++oz) {
                        int z = z1 + oz;
                        if (z < 0 || z >= D) continue;

                        float dist = sqrtf((float)(ox*ox + oy*oy + oz*oz));
                        float w = expf(-dist * inv_decay);
                        float val = prior + (occ_val - prior) * w;

                        int idx = ((x * H) + y) * D + z;
                        grid[idx] = fmaxf(grid[idx], val);
                    }
                }
            }
        }
        ''', 'dilate_decay')

    def updateBasedOnPose(self, x_t: pose):
        ox = int((x_t.position.x / self.frame.r) - (self.frame.size.w / 2))
        oy = int((x_t.position.y / self.frame.r) - (self.frame.size.h / 2))
        oz = int((x_t.position.z / self.frame.r) - (self.frame.size.d / 2))
        self.frame.origin = origin(ox, oy, oz)

    def generateGridMap(self, z_t: lidarScan3D, x_t: pose, isMinimizeFrame: bool = True) -> gridMap:
        cp = self.cp
        # Reset grid to prior
        self.data.fill(self.occPrior)

        # Transform points to world (CuPy)
        pts_world = z_t.transform(x_t).points3D
        if not isinstance(pts_world, cp.ndarray):
            pts_world = cp.asarray(pts_world)

        # Convert to voxel indices (match CPU rounding)
        rx = (pts_world[:, 0] / self.frame.r) - self.frame.origin.x
        ry = (pts_world[:, 1] / self.frame.r) - self.frame.origin.y
        rz = (pts_world[:, 2] / self.frame.r) - self.frame.origin.z

        ix = cp.rint(rx).astype(cp.int32)
        iy = cp.rint(ry).astype(cp.int32)
        iz = cp.rint(rz).astype(cp.int32)

        inb = (ix >= 0) & (ix < self.frame.size.w) & \
              (iy >= 0) & (iy < self.frame.size.h) & \
              (iz >= 0) & (iz < self.frame.size.d)

        ix = ix[inb]
        iy = iy[inb]
        iz = iz[inb]

        if ix.size == 0:
            return gridMap(self.frame, self.data)

        # Sensor origin in grid
        sx, sy, sz = self.frame.world_to_idx(x_t.position.x, x_t.position.y, x_t.position.z)
        if not self.frame.in_bounds(sx, sy, sz):
            raise ValueError("Sensor origin out of bounds.")

        if isMinimizeFrame:
            min_ix = cp.minimum(ix.min(), cp.int32(sx))
            min_iy = cp.minimum(iy.min(), cp.int32(sy))
            min_iz = cp.minimum(iz.min(), cp.int32(sz))
            max_ix = cp.maximum(ix.max(), cp.int32(sx))
            max_iy = cp.maximum(iy.max(), cp.int32(sy))
            max_iz = cp.maximum(iz.max(), cp.int32(sz))

        # Carve rays (cone or line)
        N = ix.size
        threads = 256
        blocks = (N + threads - 1) // threads

        if self.cone_free and self.diffusion_radius > 0:
            self._carve_cone_decay_kernel((blocks,), (threads,),
                                          (N,
                                           cp.int32(sx), cp.int32(sy), cp.int32(sz),
                                           ix, iy, iz,
                                           self.data.ravel(),
                                           cp.int32(self.frame.size.w),
                                           cp.int32(self.frame.size.h),
                                           cp.int32(self.frame.size.d),
                                           cp.float32(self.invModel[0]),
                                           cp.float32(self.occPrior),
                                           cp.int32(self.diffusion_radius),
                                           cp.float32(self.diffusion_radius)))
        else:
            self._carve_kernel((blocks,), (threads,),
                               (N,
                                cp.int32(sx), cp.int32(sy), cp.int32(sz),
                                ix, iy, iz,
                                self.data.ravel(),
                                cp.int32(self.frame.size.w),
                                cp.int32(self.frame.size.h),
                                cp.int32(self.frame.size.d),
                                cp.float32(self.invModel[0])))

        if self.diffusion_radius > 0:
            self._dilate_decay_kernel((blocks,), (threads,),
                                      (N, ix, iy, iz,
                                       self.data.ravel(),
                                       cp.int32(self.frame.size.w),
                                       cp.int32(self.frame.size.h),
                                       cp.int32(self.frame.size.d),
                                       cp.float32(self.invModel[1]),
                                       cp.float32(self.occPrior),
                                       cp.int32(self.diffusion_radius),
                                       cp.float32(self.diffusion_radius)))
        else:
            self.data[ix, iy, iz] = cp.float32(self.invModel[1])

        if isMinimizeFrame:
            min_ix = int(cp.asnumpy(min_ix))
            min_iy = int(cp.asnumpy(min_iy))
            min_iz = int(cp.asnumpy(min_iz))
            max_ix = int(cp.asnumpy(max_ix))
            max_iy = int(cp.asnumpy(max_iy))
            max_iz = int(cp.asnumpy(max_iz))

            crop_origin = origin(self.frame.origin.x + min_ix,
                                self.frame.origin.y + min_iy,
                                self.frame.origin.z + min_iz)
            crop_size = size(max_ix - min_ix + 1,
                            max_iy - min_iy + 1,
                            max_iz - min_iz + 1)
            crop_frame = frame(crop_origin, crop_size, self.frame.r)

            return gridMap(self.frame, self.data).reshape(crop_frame, self.occPrior)
        else:
            return gridMap(self.frame, self.data)

def main():
    import cupy as cp
    runs = 100

    def _print_bench_stats(name: str, samples: list[float]):
        arr = np.array(samples, dtype=float)
        print(f'{name} first run: {arr[0]:.6f}s')
        print(f'{name} mean all {runs}: {arr.mean():.6f}s')
        print(f'{name} mean runs 2..{runs}: {arr[1:].mean():.6f}s')

    # -----------------------------
    # 2D benchmark and comparison
    # -----------------------------
    smOrigin = origin(0, 0, 0)
    width = 300
    height = 100
    resolution = 0.5
    sensorRange = 50
    invModel = [0.1, 0.9]
    occPrior = 0.5
    smSize = size(width, height)

    sM_2d = sensorModel(frame(smOrigin, smSize, resolution), sensorRange, invModel, occPrior)
    sM_gpu_2d = sensorModelGPU(frame(smOrigin, smSize, resolution), sensorRange, invModel, occPrior)

    with open("./logs/sim_corridor/z_100.csv") as data:
        z_t = lidarScan(*np.array([line.split(",") for line in data]).astype(float).T)

    with open("./logs/sim_corridor/x_100.csv") as data:
        x_t_raw = np.array([line.split(",") for line in data]).astype(float)[0]
    x_t_2d = pose(position(x_t_raw[0], x_t_raw[1], 0.0), orientation(0.0, 0.0, x_t_raw[2]))

    cpu_times_2d = []
    gpu_times_2d = []
    gm_cpu_2d = None
    gm_gpu_2d = None

    for _ in range(runs):
        start = time.time()
        gm_cpu_2d = sM_2d.generateGridMap(z_t, x_t_2d)
        cpu_times_2d.append(time.time() - start)

        cp.cuda.Stream.null.synchronize()
        start = time.time()
        gm_gpu_2d = sM_gpu_2d.generateGridMap(z_t, x_t_2d)
        cp.cuda.Stream.null.synchronize()
        gpu_times_2d.append(time.time() - start)

    print('\n2D benchmark ({runs} runs):'.format(runs=runs))
    _print_bench_stats('2D CPU', cpu_times_2d)
    _print_bench_stats('2D GPU', gpu_times_2d)

    cpu_grid_2d = gm_cpu_2d.data[:, :, 0]
    gpu_grid_2d = cp.asnumpy(gm_gpu_2d.data[:, :, 0])
    diff_2d = np.abs(cpu_grid_2d - gpu_grid_2d)
    max_diff_2d = float(np.max(diff_2d))
    differing_2d = np.where(diff_2d > 0)
    differing_2d_tol = np.where(diff_2d > 1e-6)
    print('2D max abs difference:', max_diff_2d)
    print('2D exact differing cells:', differing_2d[0].size)
    print('2D differing cells (>1e-6):', differing_2d_tol[0].size)
    print('2D allclose (atol=1e-6):', bool(np.allclose(cpu_grid_2d, gpu_grid_2d, atol=1e-6)))

    # -----------------------------
    # 3D benchmark and comparison
    # -----------------------------
    smOrigin = origin(0, 0, 0)
    smSize = size(100, 100, 25)
    resolution = 0.5
    invModel = [0.1, 0.9]
    occPrior = 0.5

    sM_3d = sensorModel3D(frame(smOrigin, smSize, resolution), invModel, occPrior)
    sM_gpu_3d = sensorModel3DGPU(frame(smOrigin, smSize, resolution), invModel, occPrior)

    z_t_3D = read3DLidarCSV("./logs/2024-02-13-10-35-56/z_1.csv")
    z_t_3D.voxelGridFilter(resolution)
    print('Number of points after voxel grid filter:', z_t_3D.points3D.shape[0])

    x_t_3d = pose(position(25, 25, 2.0), orientation(0.0, 0.0, 0.0))

    cpu_times_3d = []
    gpu_times_3d = []
    gm_cpu_3d = None
    gm_gpu_3d = None

    for _ in range(runs):
        start = time.time()
        gm_cpu_3d = sM_3d.generateGridMap(z_t_3D, x_t_3d)
        cpu_times_3d.append(time.time() - start)

        cp.cuda.Stream.null.synchronize()
        start = time.time()
        gm_gpu_3d = sM_gpu_3d.generateGridMap(z_t_3D, x_t_3d, isMinimizeFrame=False)
        cp.cuda.Stream.null.synchronize()
        gpu_times_3d.append(time.time() - start)

    print('\n3D benchmark ({runs} runs):'.format(runs=runs))
    _print_bench_stats('3D CPU', cpu_times_3d)
    _print_bench_stats('3D GPU', gpu_times_3d)

    cpu_grid_3d = gm_cpu_3d.data
    gpu_grid_3d = cp.asnumpy(gm_gpu_3d.data)
    diff_3d = np.abs(cpu_grid_3d - gpu_grid_3d)
    print('3D max abs difference:', float(np.max(diff_3d)))
    differing_3d = np.where(diff_3d > 1e-6)
    print('3D differing voxels (>1e-6):', differing_3d[0].size)

if __name__ == '__main__':
    main()