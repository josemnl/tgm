import numpy as np
import cupy as cp
from gridMap import gridMap
from spatial import frame, origin, size, pose, position, orientation
from lidarScan import lidarScan, lidarScan3D
import time
from utilities import read3DLidarCSV

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
    def __init__(self, smFrame: frame, invModel, occPrior: float):
        assert isinstance(smFrame, frame)
        assert smFrame.size.d > 0
        self.frame = smFrame
        self.invModel = invModel          # [free_val, occ_val]
        self.occPrior = occPrior
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
                    // Single voxel ray
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

    def updateBasedOnPose(self, x_t: pose):
        ox = int((x_t.position.x / self.frame.r) - (self.frame.size.w / 2))
        oy = int((x_t.position.y / self.frame.r) - (self.frame.size.h / 2))
        oz = int((x_t.position.z / self.frame.r) - (self.frame.size.d / 2))
        self.frame.origin = origin(ox, oy, oz)

    def generateGridMap(self, z_t: lidarScan3D, x_t: pose, isMinimizeFrame: bool = True) -> gridMap:
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

        # Match CPU indexing (frame.world_to_idx uses np.round), so use cp.rint here
        ix = cp.rint(rx).astype(cp.int32)
        iy = cp.rint(ry).astype(cp.int32)
        iz = cp.rint(rz).astype(cp.int32)

        inb = (ix >= 0) & (ix < self.frame.size.w) & \
              (iy >= 0) & (iy < self.frame.size.h) & \
              (iz >= 0) & (iz < self.frame.size.d)

        ix = ix[inb]
        iy = iy[inb]
        iz = iz[inb]

        # If nothing is in bounds, return full map
        if ix.size == 0:
            return gridMap(self.frame, self.data)

        # Sensor origin in grid
        sx, sy, sz = self.frame.world_to_idx(x_t.position.x, x_t.position.y, x_t.position.z)
        if not self.frame.in_bounds(sx, sy, sz):
            raise ValueError("Sensor origin out of bounds.")

        # Bounding box of all affected voxels (endpoints + origin)
        if isMinimizeFrame:
            min_ix = cp.minimum(ix.min(), cp.int32(sx))
            min_iy = cp.minimum(iy.min(), cp.int32(sy))
            min_iz = cp.minimum(iz.min(), cp.int32(sz))
            max_ix = cp.maximum(ix.max(), cp.int32(sx))
            max_iy = cp.maximum(iy.max(), cp.int32(sy))
            max_iz = cp.maximum(iz.max(), cp.int32(sz))

        # Carve rays
        N = ix.size
        threads = 256
        blocks = (N + threads - 1) // threads
        self._carve_kernel((blocks,), (threads,),
                           (N,
                            cp.int32(sx), cp.int32(sy), cp.int32(sz),
                            ix, iy, iz,
                            self.data.ravel(),
                            cp.int32(self.frame.size.w),
                            cp.int32(self.frame.size.h),
                            cp.int32(self.frame.size.d),
                            cp.float32(self.invModel[0])))

        # Mark endpoints as occupied
        self.data[ix, iy, iz] = cp.float32(self.invModel[1])

        if isMinimizeFrame:
            # Crop via reshape
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
    # Test 2D sensor model
    smOrigin = origin(0, 0, 0)
    width = 300
    height = 100
    resolution = 0.5
    sensorRange = 50
    invModel = [0.1, 0.9]
    occPrior = 0.5
    smSize = size(width, height)
    sM = sensorModel(frame(smOrigin, smSize, resolution), sensorRange, invModel, occPrior)

    with open("./logs/sim_corridor/z_100.csv") as data:
        z_t = lidarScan(*np.array([line.split(",") for line in data]).astype(float).T)
    
    with open("./logs/sim_corridor/x_100.csv") as data:
        x_t = np.array([line.split(",") for line in data]).astype(float)[0]
    x_t = pose(position(x_t[0], x_t[1], 0.0), orientation(0.0, 0.0, x_t[2]))

    start = time.time()
    gm = sM.generateGridMap(z_t, x_t)
    print(time.time() - start)
    #gm.plot()

    # Test 3D sensor model
    smOrigin = origin(0, 0, 0)
    smSize = size(100, 100, 25)
    resolution = 0.5
    sensorRange = 50
    invModel = [0.1, 0.9]
    occPrior = 0.5
    sM = sensorModel3D(frame(smOrigin, smSize, resolution), invModel, occPrior)

    z_t_3D = read3DLidarCSV("./logs/2024-02-13-10-35-56/z_1.csv")

    z_t_3D.voxelGridFilter(resolution)

    # print number of points
    print("Number of points after voxel grid filter:", z_t_3D.points3D.shape[0])

    z_t_3D.plot()

    x_t = pose(position(25, 25, 2.0), orientation(0.0, 0.0, 0.0))
    start = time.time()
    gm = sM.generateGridMap(z_t_3D, x_t)
    print('Time taken (CPU):', time.time() - start)
    gm.plot3D_scatter(isPause=True, value_min=0.6, value_max=1.0)

    # Test 3D sensor model on GPU
    sM_gpu = sensorModel3DGPU(frame(smOrigin, smSize, resolution), invModel, occPrior)

    start = time.time()
    gm_gpu = sM_gpu.generateGridMap(z_t_3D, x_t)
    print('Time taken (GPU):', time.time() - start)
    gm_gpu.plot3D_scatter(isPause=True, value_min=0.6, value_max=1.0)

    # Compute the difference between CPU and GPU results
    diff = np.abs(gm.data - cp.asnumpy(gm_gpu.data))
    print('Max difference between CPU and GPU results:', np.max(diff))
    # Diagnostic counts
    cpu_grid = gm.data
    gpu_grid = cp.asnumpy(gm_gpu.data)
    differing = np.where(np.abs(cpu_grid - gpu_grid) > 1e-6)
    print('Total differing voxels:', differing[0].size)
    if differing[0].size > 0:
        # Show distribution of CPU values where they differ
        unique_cpu, counts_cpu = np.unique(cpu_grid[differing], return_counts=True)
        unique_gpu, counts_gpu = np.unique(gpu_grid[differing], return_counts=True)
        print('CPU differing value counts:', dict(zip(unique_cpu, counts_cpu)))
        print('GPU differing value counts:', dict(zip(unique_gpu, counts_gpu)))

if __name__ == '__main__':
    main()