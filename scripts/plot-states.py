import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from mpl_toolkits.axisartist.axislines import SubplotZero

def plotStates(res, lim=1, saveImg=False):
    I = np.zeros((int(res*lim), int(res*lim), 3))
    fig = plt.figure()
    staticMap = np.zeros((int(res*lim), int(res*lim)))
    dynamicMap = np.zeros((int(res*lim), int(res*lim)))

    for i in range(res):
        for j in range(res):
            if i + j <= res:
                staticMap[i, j] = (i+1)/res
                dynamicMap[i, j] = (j+1)/res

    I[:,:,0] = 1 - np.transpose(1.0*staticMap + 0.0*dynamicMap)
    I[:,:,1] = 1 - np.transpose(0.5*staticMap + 0.5*dynamicMap)
    I[:,:,2] = 1 - np.transpose(0.0*staticMap + 1.0*dynamicMap)
    #ax = fig.add_subplot(1, 1, 1)
    ax = SubplotZero(fig, 111)

    for direction in ["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")

        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    fig.add_subplot(ax)
    ax.imshow(I, vmin=0, vmax=1, origin ="lower",
                extent=(0, lim,
                        0, lim))
    trianglex = [0, 1, 1, 0] 
    triangley = [1, 1, 0, 1]
    #plt.plot([0, 1], [1, 0], 'w-', lw=2)
    ax.set_yticks(np.arange(0,1.1,1))
    ax.set_xticks(np.arange(0,1.1,1))
    plt.fill(trianglex, triangley, 'w')
    plt.xlabel("$p(m^s_{t,i})$")
    plt.ylabel("$p(m^d_{t,i})$")
    if saveImg:
        #plt.savefig(imgName + '.png')
        imsave('states' + '.png', I, origin ="lower")
    plt.show()
    #plt.pause(0.01)

def plotStatesGrey(res, lim=1, saveImg=False):
    I = np.zeros((int(res*lim), int(res*lim), 3))
    fig = plt.figure()
    staticMap = np.zeros((int(res*lim), int(res*lim)))
    dynamicMap = np.zeros((int(res*lim), int(res*lim)))

    for i in range(res):
        for j in range(res):
            staticMap[i, j] = (i+1)/res
            dynamicMap[i, j] = (i+1)/res

    I[:,:,0] = 1 - np.transpose(1.0*staticMap + 0.0*dynamicMap)
    I[:,:,1] = 1 - np.transpose(0.5*staticMap + 0.5*dynamicMap)
    I[:,:,2] = 1 - np.transpose(0.0*staticMap + 1.0*dynamicMap)
    #ax = fig.add_subplot(1, 1, 1)
    ax = SubplotZero(fig, 111)

    for direction in ["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")

        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    fig.add_subplot(ax)
    ax.imshow(I, vmin=0, vmax=1, origin ="lower",
                extent=(0, lim,
                        0, lim))
    plt.xlabel("$p(m^s_{t,i})$")
    plt.ylabel("$p(m^d_{t,i})$")
    if saveImg:
        #plt.savefig(imgName + '.png')
        imsave('states' + '.png', I, origin ="lower")
    plt.show()
    #plt.pause(0.01)



if __name__ == '__main__':
    res = 1000
    lim = 1.05
    plotStates(res, lim)
    plotStatesGrey(res, lim)