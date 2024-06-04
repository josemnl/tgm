import numpy as np
from gridMap import gridMap
from lidarScan import lidarScan
import time
from collections import deque

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

    def generateGridMap(self, z_t, x_t, z_t_ground=None):
        timeStart = time.time()
        assert isinstance(z_t, lidarScan)
        assert isinstance(z_t_ground, lidarScan) or z_t_ground is None
        ang, dist = z_t.angles, z_t.ranges

        # Update measurement orientation with agent's pose
        np.add(ang, x_t[2], out=ang)
        timePose = time.time()

        # Limit measurement distance to sensor range
        np.clip(dist, a_min=None, a_max=self.sensorRange, out=dist)
        timeClip = time.time()

        # Compute detection points on global frame
        ox = x_t[0] + np.cos(ang) * dist
        oy = x_t[1] + np.sin(ang) * dist
        timeGlobal = time.time()

        # Compute ground points on global frame
        if z_t_ground is not None:
            ang_ground, dist_ground = z_t_ground.angles, z_t_ground.ranges
            np.add(ang, x_t[2], out=ang)
            np.clip(dist_ground, a_min=None, a_max=self.sensorRange, out=dist_ground)
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
        for i in range(ox.size):
            self.insetRay((ix_t[0], ix_t[1]), (ix[i], iy[i]), self.invModel[0])
        timeFree = time.time()

        # If ground points are provided, mark them as free
        if z_t_ground is not None:
            # Compute the matrix indices for ground points
            ix_ground = np.round((ox_ground - self.origin[0]) * self.resolution).astype(int)
            iy_ground = np.round((oy_ground - self.origin[1]) * self.resolution).astype(int)
            # Mark free cells on the ground
            self.data[ix_ground, iy_ground] = self.invModel[0]
        timeGroundFree = time.time()

        # Mark cells in between detections as unknown
        for i in range(ox.size):
            self.insetRay((ix[i], iy[i]), (ix[i-1], iy[i-1]), self.occPrior)
        timeUnknown = time.time()

        # Remove points exactly at the sensor limit
        ix = ix[dist < self.sensorRange]
        iy = iy[dist < self.sensorRange]
        
        # Mark occupied cells
        self.data[ix, iy] = self.invModel[1]
        timeOccupied = time.time()

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

        return gridMap(int(self.origin[0]*self.resolution), int(self.origin[1]*self.resolution), int(self.width*self.resolution), int(self.height*self.resolution), 1/self.resolution, self.data)

    def insetRay(self,start,end,value):
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        is_steep = abs(dy) > abs(dx)

        if is_steep:
            if dy == 0:
                y_coords = np.array([y1])
                x_coords = np.linspace(x1, x2, abs(x2 - x1) + 1).astype(int)
            else:
                y_coords = np.linspace(y1, y2, abs(y2 - y1) + 1).astype(int)
                x_coords = np.round(x1 + dx/dy * (y_coords - y1)).astype(int)
        else:
            if dx == 0:
                x_coords = np.array([x1])
                y_coords = np.linspace(y1, y2, abs(y2 - y1) + 1).astype(int)
            else:
                x_coords = np.linspace(x1, x2, abs(x2 - x1) + 1).astype(int)
                y_coords = np.round(y1 + dy/dx * (x_coords - x1)).astype(int)

        self.data[x_coords, y_coords] = value

def bresenham(start, end):
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = deque()
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points = deque(reversed(points))
    points = np.array(points)
    return points

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