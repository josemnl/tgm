import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave

def plotSLAMfigure(baselineFile, TGMFile, saveImg=False):
    baseline = np.loadtxt(baselineFile, delimiter=',')
    TGM = np.loadtxt(TGMFile, delimiter=',')
    fig = plt.figure()
    plt.plot(baseline[:,0], label='Baseline')
    plt.plot(TGM[:,0], label='TGM')
    plt.xlabel('Time step')
    plt.ylabel('X position (m)')
    plt.legend()
    if saveImg:
        fig.savefig('SLAM_2' + '.svg', format='svg', dpi=1200)
    plt.show()
    #plt.pause(0.01)

if __name__ == '__main__':
    baselineFile = './results/SLAM-2024-03-15-11-25-54-Baseline/x_t_SLAM.csv'
    TGMFile = './results/SLAM-2024-03-15-11-25-54-TGM/x_t_SLAM.csv'
    plotSLAMfigure(baselineFile, TGMFile, True)