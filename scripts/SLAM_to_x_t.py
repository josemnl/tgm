import numpy as np
import os

LOG = "25"

INPUT_DIRECTORY = "D:/RAYGROUND-results/SnowyKitti-" + LOG + "-DROR-k-5-rho-0.07"
OUTPUT_DIRECTORY = "D:/snowyKITTI/dataset/sequences/" + LOG + "/snow_pose"

def SLAM_to_x_t():
    # Load the SLAM output, called x_t_SLAM.csv. The formas is [x, y, theta] at each row for each time step
    x_t_SLAM = np.loadtxt(INPUT_DIRECTORY + '/x_t_SLAM.csv', delimiter=',')

    # Check if the output directory exists, if not, create it
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # Save each x_t as an independent file, called "x_" + str(i).zfill(6) + ".csv"
    for i in range(x_t_SLAM.shape[0]):
        np.savetxt(OUTPUT_DIRECTORY + '/x_' + str(i).zfill(6) + '.csv', x_t_SLAM[i].reshape(1, -1), delimiter=',')

if __name__ == "__main__":
    SLAM_to_x_t()