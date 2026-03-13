import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave

def plotSLAMfigure(baseline1File, baseline2File, TGMFile, GroundTruthFile, saveImg=False):
    baseline1 = np.loadtxt(baseline1File, delimiter=',')
    baseline2 = np.loadtxt(baseline2File, delimiter=',')
    TGM = np.loadtxt(TGMFile, delimiter=',')
    GroundTruth = np.loadtxt(GroundTruthFile, delimiter=',')
    fig = plt.figure()
    # Plot vertical line at t=48s and t=55s
    plt.axvline(x=48, color='dimgray', linestyle='dotted')
    plt.axvline(x=55, color='dimgray', linestyle='dotted')
    x_values = np.arange(len(TGM[:,0])) / 10
    plt.plot(x_values, baseline1[:,0], 'tab:blue', label='OGM')#, linestyle=(0, (4, 3, 1, 3)))
    plt.plot(x_values, baseline2[:,0], 'tab:red', label='c-OGM')#, linestyle=(0, (3, 4, 1, 4, 1, 4)))
    plt.plot(x_values, TGM[:,0], 'tab:orange', label='TGM')
    plt.plot(x_values, GroundTruth[:,0], 'tab:green', label='Ground Truth', linestyle=(0, (5, 5)))
    plt.xlabel('Time (s)')
    plt.ylabel('X position (m)')
    plt.legend()
    plt.gca().set_aspect(0.3)
    if saveImg:
        fig.savefig('SLAM_3' + '.svg', format='svg', dpi=1200)
    plt.show()
    #plt.pause(0.01)

if __name__ == '__main__':
    baseline1File = './results/Exp2-2024-03-15-11-25-54-Baseline/x_t_SLAM.csv'
    baseline2File = './results/Exp2-2024-03-15-11-25-54-BaselineSat/x_t_SLAM.csv'
    TGMFile = './results/Exp2-2024-03-15-11-25-54-TGM/x_t_SLAM.csv'
    GroundTruthFile = './results/Exp2-2024-03-15-11-25-54-GT-TGM/x_t_SLAM.csv'
    plotSLAMfigure(baseline1File, baseline2File, TGMFile, GroundTruthFile, True)