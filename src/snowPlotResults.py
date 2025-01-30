import os
import matplotlib.pyplot as plt
import numpy as np
from utilities import loadConfigAsDict

#DATASET_ROOT = './snowyKITTI/dataset/sequences/'
RESULTS_ROOT = 'D:/Results/'
META_RESULTS_FOLDER = './results/snowMetaResults/'

VALID_LOGS = [11]

ROR_VALUES = [0.25, 0.2, 0.18, 0.15]
SOR_VALUES = [0.1, 0.15, 0.25, 0.5]
DROR_VALUES = [0.07]

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
        for filter in ['DROR']:
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
            with open(folder + 'Precision.csv') as f:
                Precision = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'Recall.csv') as f:
                Recall = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'F1.csv') as f:
                F1 = np.array([line.split(",") for line in f]).astype(float)

            # Load new files
            with open(folder + 'IoU_b.csv') as f:
                IoU_b = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'Precision_b.csv') as f:
                Precision_b = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'Recall_b.csv') as f:
                Recall_b = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'F1_b.csv') as f:
                F1_b = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'IoU_t.csv') as f:
                IoU_t = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'Precision_t.csv') as f:
                Precision_t = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'Recall_t.csv') as f:
                Recall_t = np.array([line.split(",") for line in f]).astype(float)
            with open(folder + 'F1_t.csv') as f:
                F1_t = np.array([line.split(",") for line in f]).astype(float)

            # Plot the nWrongSnowGrids
            plt.plot(nWrongSnowGrids_original, label='Unfiltered')
            plt.plot(nWrongSnowGrids_baseline, label=str(filter))
            plt.plot(nWrongSnowGrids_TGM, label=str(filter) + ' + TGM')
            plt.legend()
            plt.ylim(0, max(max(nWrongSnowGrids_original), max(nWrongSnowGrids_baseline), max(nWrongSnowGrids_TGM)))
            plt.xlim(0, len(nWrongSnowGrids_original))
            plt.gcf().set_size_inches(20, 5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_nWrongSnowGrids.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_nWrongSnowGrids.svg', format='svg', dpi=1200)

            # Clear the plot
            plt.clf()
            
            # Plot the IoU
            plt.plot(IoU)
            plt.ylim(0, 1)
            plt.xlim(0, len(IoU))
            plt.gcf().set_size_inches(20, 5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_IoU.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_IoU.svg', format='svg', dpi=1200)

            # Clear the plot
            plt.clf()

            # Plot precision, recall and F1
            plt.plot(Precision, label='Precision')
            plt.plot(Recall, label='Recall')
            plt.plot(F1, label='F1')
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, len(Precision))
            plt.gcf().set_size_inches(20, 5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_metrics.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_metrics.svg', format='svg', dpi=1200)

            # clear the plot
            plt.clf()

            # Plot f1 and IoU
            plt.plot(F1, label='F1')
            plt.plot(IoU, label='IoU')
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, len(F1))
            plt.gcf().set_size_inches(20, 5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_f1_iou.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_f1_iou.svg', format='svg', dpi=1200)

            # Clear the plot
            plt.clf()

            # NEW METRICS
            # IoU
            plt.plot(IoU_b, label='IoU_b')
            plt.plot(IoU_t, label='IoU_t')
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, len(IoU_b))
            plt.gcf().set_size_inches(20, 5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_new_metrics_iou.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_new_metrics_iou.svg', format='svg', dpi=1200)

            # Clear the plot
            plt.clf()

            # Precision
            plt.plot(Precision_b, label='Precision_b')
            plt.plot(Precision_t, label='Precision_t')
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, len(Precision_b))
            plt.gcf().set_size_inches(20, 5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_new_metrics_precision.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_new_metrics_precision.svg', format='svg', dpi=1200)

            # Clear the plot
            plt.clf()

            # Recall
            plt.plot(Recall_b, label='Recall_b')
            plt.plot(Recall_t, label='Recall_t')
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, len(Recall_b))
            plt.gcf().set_size_inches(20, 5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_new_metrics_recall.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_new_metrics_recall.svg', format='svg', dpi=1200)

            # Clear the plot
            plt.clf()

            # F1
            plt.plot(F1_b, label='F1_b')
            plt.plot(F1_t, label='F1_t')
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, len(F1_b))
            plt.gcf().set_size_inches(20, 5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_new_metrics_f1.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_new_metrics_f1.svg', format='svg', dpi=1200)

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
    snowPlotResults()
    #filterValuesPlot()