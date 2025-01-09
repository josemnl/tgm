import os
import matplotlib.pyplot as plt
import numpy as np
from utilities import loadConfigAsDict

#DATASET_ROOT = './snowyKITTI/dataset/sequences/'
RESULTS_ROOT = './results/'
META_RESULTS_FOLDER = './results/snowMetaResults/'

VALID_LOGS = [0, 2, 3, 5, 7, 8, 9, 11, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25]

TEST_LOGS = [22]

def snowPlotResults():
    configPath = './config/'
    # Load default config file
    defConfFile = 'config'
    conf = loadConfigAsDict(configPath, defConfFile)

    # Load specific config file
    snowConfig = 'snowyKitti'
    specificConf = loadConfigAsDict(configPath, snowConfig)

    # Update default config file with specific config file
    conf.__dict__.update(specificConf.__dict__)

    # For each log in VALID_LOGS
    for log in VALID_LOGS:
        # For each filter
        for filter in ['ROR', 'SOR', 'DROR']:
            # Define the logID
            if filter == 'ROR':
                logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(conf.ROR_k) + '-r-' + str(conf.ROR_r)
            elif filter == 'SOR':
                logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(conf.SOR_k) + '-s-' + str(conf.SOR_s)
            elif filter == 'DROR':
                logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(conf.DROR_k) + '-rho-' + str(conf.DROR_rho)

            # Define the folder
            folder = RESULTS_ROOT + logID + '/'
            
            # Load files
            with open(folder + 'nWrongSnowGrids_original.csv') as f:
                nWrongSnowGrids_original = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'nWrongSnowGrids_baseline.csv') as f:
                nWrongSnowGrids_baseline = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'nWrongSnowGrids_TGM.csv') as f:
                nWrongSnowGrids_TGM = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'IoU.csv') as f:
                IoU = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'Intersection.csv') as f:
                Intersection = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'Union.csv') as f:
                Union = np.array([line.split(",") for line in f]).astype(float)

            # Plot the nWrongSnowGrids
            plt.plot(nWrongSnowGrids_original, label='Original')
            plt.plot(nWrongSnowGrids_baseline, label='Baseline')
            plt.plot(nWrongSnowGrids_TGM, label='TGM')
            plt.xlabel('Frame')
            plt.ylabel('nWrongSnowGrids')
            plt.title('nWrongSnowGrids for ' + logID)
            plt.legend()
            plt.savefig(META_RESULTS_FOLDER + logID + '_nWrongSnowGrids.png')

            # Clear the plot
            plt.clf()
            
            # Plot the IoU
            plt.plot(IoU)
            plt.xlabel('Frame')
            plt.ylabel('IoU')
            plt.title('IoU for ' + logID)
            plt.savefig(META_RESULTS_FOLDER + logID + '_IoU.png')

            # Clear the plot
            plt.clf()


            # Print mean IoU
            print('Mean IoU for ' + logID + ' with filter ' + filter + ': ' + str(np.mean(IoU)))

if __name__ == '__main__':
    snowPlotResults()