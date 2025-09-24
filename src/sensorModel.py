from typing import Optional
import numpy as np
from gridMap import gridMap, frame, origin, size, pose, position, orientation
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
    def __init__(self, smFrame: frame, sensorRange, invModel, occPrior: float):
        assert isinstance(smFrame, frame)
        assert smFrame.size.d > 0
        self.frame = smFrame
        self.sensorRange = sensorRange          # in cells (same semantics as 2D)
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

        # Transform 3D points from sensor to world using RPY
        R = self._rpy_to_R(
            x_t.orientation.roll, x_t.orientation.pitch, x_t.orientation.yaw
        )
        T = np.array([x_t.position.x, x_t.position.y, x_t.position.z])
        world_pts = (R @ z_t.points3D.T).T + T

        # Range clip: sensorRange is in cells; convert to meters
        max_range_m = self.sensorRange * self.frame.r
        ranges = np.linalg.norm(world_pts - T, axis=1)
        mask = ranges < max_range_m
        world_pts = world_pts[mask]

        # Start voxel (robot cell)
        sx, sy, sz = self._world_to_idx(T[0], T[1], T[2])
        if not self._in_bounds(sx, sy, sz):
            # Error: sensor origin out of bounds
            raise ValueError("Sensor origin out of bounds of the grid map.")

        # Transform 3D points from world to grid indices using _world_to_idx
        grid_pts = np.array([self._world_to_idx(p[0], p[1], p[2]) for p in world_pts])

        # Filter out-of-bounds points using _in_bounds
        in_bounds_mask = np.array([self._in_bounds(p[0], p[1], p[2]) for p in grid_pts])
        grid_pts = grid_pts[in_bounds_mask]

        # Mark free cells along the rays
        for p in grid_pts:
            ex, ey, ez = p
            # Vectorized, dominant-axis integer-index line (like your 2D insertRay)
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

        # Choose dominant axis
        if adx >= ady and adx >= adz:
            # March in x (vectorized)
            if dx == 0:
                x_coords = np.array([x1], dtype=int)
            else:
                step = 1 if dx > 0 else -1
                x_coords = np.arange(x1, x2 + step, step, dtype=int)
            # y,z by integer division (floor) to avoid floating drift
            if dx != 0:
                y_coords = np.floor(y1 + (dy * (x_coords - x1) / dx)).astype(int)
                z_coords = np.floor(z1 + (dz * (x_coords - x1) / dx)).astype(int)
            else:
                y_coords = np.array([y1], dtype=int).repeat(len(x_coords))
                z_coords = np.array([z1], dtype=int).repeat(len(x_coords))
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
                x_coords = np.array([x1], dtype=int).repeat(len(y_coords))
                z_coords = np.array([z1], dtype=int).repeat(len(y_coords))
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
                x_coords = np.array([x1], dtype=int).repeat(len(z_coords))
                y_coords = np.array([y1], dtype=int).repeat(len(z_coords))

        # Bounds mask (avoid IndexError)
        inb = (
            (0 <= x_coords) & (x_coords < self.frame.size.w) &
            (0 <= y_coords) & (y_coords < self.frame.size.h) &
            (0 <= z_coords) & (z_coords < self.frame.size.d)
        )

        xi, yi, zi = x_coords[inb], y_coords[inb], z_coords[inb]

        if valueCondition is None:
            self.data[xi, yi, zi] = value
        else:
            # Only write if none of the traversed cells equal valueCondition (as in 2D)
            if np.all(self.data[xi, yi, zi] != valueCondition):
                self.data[xi, yi, zi] = value

    def _world_to_idx(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        ix = int(np.round((x / self.frame.r) - self.frame.origin.x))
        iy = int(np.round((y / self.frame.r) - self.frame.origin.y))
        iz = int(np.round((z / self.frame.r) - self.frame.origin.z))
        return ix, iy, iz

    def _in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        return (0 <= ix < self.frame.size.w) and (0 <= iy < self.frame.size.h) and (0 <= iz < self.frame.size.d)

    @staticmethod
    def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

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
    sM = sensorModel3D(frame(smOrigin, smSize, resolution), sensorRange, invModel, occPrior)

    z_t_3D = read3DLidarCSV("./logs/2024-02-13-10-35-56/z_1.csv")

    z_t_3D.voxelGridFilter(resolution)

    z_t_3D.plot()

    x_t = pose(position(25, 25, 2.0), orientation(0.0, 0.0, 0.0))
    start = time.time()
    gm = sM.generateGridMap(z_t_3D, x_t)
    print(time.time() - start)
    gm.plot3D_scatter(isPause=True, value_min=0.6, value_max=1.0)

if __name__ == '__main__':
    main()