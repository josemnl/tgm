import numpy as np
import matplotlib.pyplot as plt
from utilities import read3DLidarCSV

def plot_lidar_from_bag(logPath, i):
    z_t_3D = read3DLidarCSV(logPath + "z_" + str(i) + ".csv")
    groundThreshold = -1.5
    skyThreshold = 1
    minDistance = 2
    maxDistance = 35
    z_t = z_t_3D.removeSky(skyThreshold).removeGround(groundThreshold).convertTo2D().removeClosePoints(minDistance).removeFarPoints(maxDistance)
    

    # Plot ego vehicle
    x = 0
    y = 0
    theta = 0
    length = 4.953
    width = 1.923
    x1 = x + length/2 * np.cos(theta) + width/2 * np.cos(theta + np.pi/2)
    y1 = y + length/2 * np.sin(theta) + width/2 * np.sin(theta + np.pi/2)
    x2 = x + length/2 * np.cos(theta) - width/2 * np.cos(theta + np.pi/2)
    y2 = y + length/2 * np.sin(theta) - width/2 * np.sin(theta + np.pi/2)
    x3 = x - length/2 * np.cos(theta) - width/2 * np.cos(theta + np.pi/2)
    y3 = y - length/2 * np.sin(theta) - width/2 * np.sin(theta + np.pi/2)
    x4 = x - length/2 * np.cos(theta) + width/2 * np.cos(theta + np.pi/2)
    y4 = y - length/2 * np.sin(theta) + width/2 * np.sin(theta + np.pi/2)
    plt.fill([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], color='white', edgecolor='black')
    # Plot the heading as a triangle
    x1 = x + length/2 * np.cos(theta)
    y1 = y + length/2 * np.sin(theta)
    x2 = x + (length/2-width) * np.cos(theta) + width/2 * np.sin(theta)
    y2 = y + (length/2-width) * np.sin(theta) - width/2 * np.cos(theta)
    x3 = x + (length/2-width) * np.cos(theta) - width/2 * np.sin(theta)
    y3 = y + (length/2-width) * np.sin(theta) + width/2 * np.cos(theta)
    plt.fill([x1, x2, x3, x1], [y1, y2, y3, y1], color='white', edgecolor='black')
    z_t.plot()

    # Save plot as svg with the name of the log file
    plt.savefig('lidar_' + str(i) + '.svg')

if __name__ == "__main__":
    logID = '2024-03-15-11-25-54'
    i = 480 + 350
    logID = '2024-03-01-15-10-32'
    i = 480 + 600
    logPath = './logs/' + logID + '/'
    
    plot_lidar_from_bag(logPath, i)