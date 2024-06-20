import numpy as np
from tgm.gridMap import gridMap
from tgm.lidarScan import lidarScan
import time

class sensorModel:
    def __init__ (self, origin, width, height, resolution, sensorRange, invModel ,occPrior):
        self.origin = origin
        self.width = width
        self.height = height
        self.resolution = resolution
        self.sensorRange = sensorRange
        self.invModel = invModel
        self.occPrior = occPrior

    def updateBasedOnPose(self, x_t):
        self.origin = ((x_t[0:2] - np.array([self.width/2, self.height/2])) * self.resolution).round(0) / self.resolution

    def generateGridMap(self, z_t, x_t):
        assert isinstance(z_t, lidarScan)
        ang, dist = z_t.angles, z_t.ranges
        # Update measurement orientation with agent's pose
        ang = ang + x_t[2]
        # Limit measurement distance to sensor range
        dist[dist>self.sensorRange] = self.sensorRange
        # Compute detection points on global frame
        ox = x_t[0] + np.cos(ang) * dist
        oy = x_t[1] + np.sin(ang) * dist
        # Compute matrix index for ego pose
        ix_t = ((x_t[0:2]-self.origin) * self.resolution).astype(int)
        # Initialize matrix with prior
        data = np.ones((self.width*self.resolution, self.height*self.resolution)) * self.occPrior
        # Mark occupied cells
        for (x, y, d) in zip(ox, oy, dist):
            # Compute the matrix index for detection points
            ix = int(round((x - self.origin[0]) * self.resolution))
            iy = int(round((y - self.origin[1]) * self.resolution))
            # If the detection is within the range, mark it as occupied
            if d<self.sensorRange:
                try:
                    data[ix][iy] = self.invModel[1]
                except:
                    pass
        # Mark free cells
        for (x, y, d) in zip(ox, oy, dist):
            # Compute the matrix index for detection points
            ix = int(round((x - self.origin[0]) * self.resolution))
            iy = int(round((y - self.origin[1]) * self.resolution))
            # Mark as free the cells along the ray
            points = bresenham((ix_t[0], ix_t[1]), (ix, iy))
            for point in points:
                try:
                    if data[point[0]][point[1]] != self.invModel[1]:
                        data[point[0]][point[1]] = self.invModel[0]
                    else:
                        break
                except:
                    pass
        for i in range(ox.size):
            ix1 = int(round((ox[i] - self.origin[0]) * self.resolution))
            iy1 = int(round((oy[i] - self.origin[1]) * self.resolution))
            ix2 = int(round((ox[i-1] - self.origin[0]) * self.resolution))
            iy2 = int(round((oy[i-1] - self.origin[1]) * self.resolution))
            points = bresenham((ix1, iy1), (ix2, iy2))
            for point in points:
                try:
                    if data[point[0]][point[1]] == self.invModel[0]:
                        data[point[0]][point[1]] = self.occPrior
                except:
                    pass
        return gridMap(self.origin, self.width, self.height, self.resolution, data)

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
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points

def main():
    origin = [0,0]
    width = 150
    height = 50
    resolution = 2
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