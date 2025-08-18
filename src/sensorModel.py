import numpy as np
from gridMap import gridMap, frame
from lidarScan import lidarScan
import time

class sensorModel:
    def __init__ (self, origin, width, height, resolution, sensorRange, invModel ,occPrior):
        # Units are converted to meters for the origin, width and height; and to cells/meter for the resolution
        self.origin = int(origin[0]*resolution), int(origin[1]*resolution)
        self.width = int(width*resolution)
        self.height = int(height*resolution)
        self.resolution = int(1/resolution)
        self.sensorRange = int(sensorRange*resolution)
        self.invModel = invModel
        self.occPrior = occPrior
        self.data = np.ones((self.width*self.resolution, self.height*self.resolution)) * self.occPrior

    def updateBasedOnPose(self, x_t):
        self.origin = ((x_t[0:2] - np.array([self.width/2, self.height/2])) * self.resolution).round(0) / self.resolution

    def generateGridMap(self, z_t, x_t, z_t_ground=None, rayTraceGround = True):
        timeStart = time.time()
        assert isinstance(z_t, lidarScan)
        assert isinstance(z_t_ground, lidarScan) or z_t_ground is None
        ang, dist = z_t.angles, z_t.ranges

        # Update measurement orientation with agent's pose
        ang = ang + x_t[2]
        timePose = time.time()

        # Remove measurements further than sensor range
        mask = dist < self.sensorRange
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
            mask = dist_ground < self.sensorRange
            ang_ground = ang_ground[mask]
            dist_ground = dist_ground[mask]
            ox_ground = x_t[0] + np.cos(ang_ground) * dist_ground
            oy_ground = x_t[1] + np.sin(ang_ground) * dist_ground
        timeGround = time.time()

        # Compute matrix index for ego pose
        ix_t = ((x_t[0:2]-self.origin) * self.resolution).astype(int)

        # Initialize matrix with prior
        self.data.fill(self.occPrior)
        timeInit = time.time()

        # Compute matrix indices for detections
        ix = np.round((ox - self.origin[0]) * self.resolution).astype(int)
        iy = np.round((oy - self.origin[1]) * self.resolution).astype(int)

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
            ix_ground = np.round((ox_ground - self.origin[0]) * self.resolution).astype(int)
            iy_ground = np.round((oy_ground - self.origin[1]) * self.resolution).astype(int)
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
            ix_ground = np.round((ox_ground - self.origin[0]) * self.resolution).astype(int)
            iy_ground = np.round((oy_ground - self.origin[1]) * self.resolution).astype(int)
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

        gridFrame = frame(int(self.origin[0]*self.resolution), int(self.origin[1]*self.resolution), int(self.width*self.resolution), int(self.height*self.resolution), 1/self.resolution)

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

def main():
    origin = [0,0]
    width = 300
    height = 100
    resolution = 0.5
    sensorRange = 50
    invModel = [0.1, 0.9]
    occPrior = 0.5
    sM = sensorModel(origin, width, height, resolution, sensorRange, invModel ,occPrior)

    with open("./logs/sim_corridor/z_100.csv") as data:
        z_t = lidarScan(*np.array([line.split(",") for line in data]).astype(float).T)
    
    with open("./logs/sim_corridor/x_100.csv") as data:
        x_t = np.array([line.split(",") for line in data]).astype(float)[0]

    start = time.time()
    gm = sM.generateGridMap(z_t, x_t)
    print(time.time() - start)
    gm.plot()
    

if __name__ == '__main__':
    main()