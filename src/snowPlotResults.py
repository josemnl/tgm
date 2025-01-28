import os
import matplotlib.pyplot as plt
import numpy as np
from utilities import loadConfigAsDict

#DATASET_ROOT = './snowyKITTI/dataset/sequences/'
RESULTS_ROOT = 'D:/Results/'
META_RESULTS_FOLDER = './results/snowMetaResults/'

VALID_LOGS = [0, 2, 3, 5, 7, 8, 9, 11, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25]

ROR_VALUES = [0.25, 0.2, 0.18, 0.15]
SOR_VALUES = [0.1, 0.15, 0.25, 0.5]
DROR_VALUES = [0.05, 0.07, 0.08, 0.1]

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
            with open(folder + 'nWrongSnowGrids_tgm.csv') as f:
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

def filterValuesPlot():
    configPath = './config/'
    # Load default config file
    defConfFile = 'config'
    conf = loadConfigAsDict(configPath, defConfFile)

    # Load specific config file
    snowConfig = 'snowyKitti'
    specificConf = loadConfigAsDict(configPath, snowConfig)

    # Update default config file with specific config file
    conf.__dict__.update(specificConf.__dict__)

    for filter in ['ROR', 'SOR', 'DROR']:
        if filter == 'ROR':
            filter_values = ROR_VALUES
        elif filter == 'SOR':
            filter_values = SOR_VALUES
        elif filter == 'DROR':
            filter_values = DROR_VALUES

        AccIoU = [[] for _ in filter_values]

        print(len(AccIoU))
        
        for i, value in enumerate(filter_values):
            for log in VALID_LOGS:
                # Define the logID
                if filter == 'ROR':
                    logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(conf.ROR_k) + '-r-' + str(value)
                elif filter == 'SOR':
                    logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(conf.SOR_k) + '-s-' + str(value)
                elif filter == 'DROR':
                    logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(conf.DROR_k) + '-rho-' + str(value)

                # Define the folder
                folder = RESULTS_ROOT + logID + '/'
                
                # Load files
                with open(folder + 'IoU.csv') as f:
                    IoU = np.array([line.split(",") for line in f]).astype(float)

                for j in IoU:
                    AccIoU[i].append(j)
                
                print(len(AccIoU[i]))

        # Count how many NaN values there are in each value
        nans = [np.count_nonzero(np.isnan(AccIoU[i])) for i in range(len(AccIoU))]

        # Remove NaN values
        AccIoU = [[x for x in AccIoU[i] if not np.isnan(x)] for i in range(len(AccIoU))]

        print('nans: ', nans)
        
        # Compute the mean IoU for each value
        meanIoU = [np.mean(AccIoU[i]) for i in range(len(AccIoU))]

        print('AccIoU: ', AccIoU)

        # Plot the mean IoU
        plt.plot(filter_values, meanIoU, label=filter, marker='o')

        plt.xlabel('Frame')
        plt.ylabel('IoU')
        plt.title('IoU for ' + filter)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    #snowPlotResults()
    filterValuesPlot()