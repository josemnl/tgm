import os
import matplotlib.pyplot as plt
import numpy as np
from utilities import loadConfigAsDict

RESULTS_ROOT = './results/'
META_RESULTS_FOLDER = './meta results/'

VALID_LOGS = [0, 2, 3, 5, 7, 8, 9, 11, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25]

DETAILED_LOG = [11]

ROR_VALUES = [0.14, 0.16, 0.18, 0.20, 0.22, 0.24]
SOR_VALUES = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]
DROR_VALUES = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

TEST_LOGS = [22]

def detailedPlot():
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
    for log in DETAILED_LOG:
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
            with open(folder + 'IoU_t_b.csv') as f:
                IoU_t_b = np.array([line.split(",") for line in f]).astype(float)

            # NEW METRICS
            # IoU baseline/unfiltered vs IoU TGM/unfiltered
            plt.plot(IoU_b, label='IoU ' + filter + ' / unfiltered')
            plt.plot(IoU_t, label='IoU ' + filter + ' + TGM / unfiltered')
            plt.legend(loc='lower left', bbox_to_anchor=(0.025, 0.0))
            plt.ylim(0, 1)
            plt.xlim(0, len(IoU_b))
            plt.gcf().set_size_inches(20, 2.5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_iou.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_iou.svg', format='svg', dpi=1200)

            # Clear the plot
            plt.clf()

            # IoU TGM/baseline
            plt.plot(IoU_t_b, label='IoU TGM / ' + filter, color='black')
            plt.legend(loc='lower left', bbox_to_anchor=(0.025, 0.0))
            plt.ylim(0, 1)
            plt.xlim(0, len(IoU_t_b))
            plt.gcf().set_size_inches(20, 2.5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_iou_tgm_baseline.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_iou_tgm_baseline.svg', format='svg', dpi=1200)

            # Clear the plot
            plt.clf()

            # Precision
            plt.plot(Precision_b, label='Precision_b')
            plt.plot(Precision_t, label='Precision_t')
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, len(Precision_b))
            plt.gcf().set_size_inches(20, 2.5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_precision.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_precision.svg', format='svg', dpi=1200)

            # Clear the plot
            plt.clf()

            # Recall
            plt.plot(Recall_b, label='Recall_b')
            plt.plot(Recall_t, label='Recall_t')
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, len(Recall_b))
            plt.gcf().set_size_inches(20, 2.5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_recall.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_recall.svg', format='svg', dpi=1200)

            # Clear the plot
            plt.clf()

            # F1
            plt.plot(F1_b, label='F1_b')
            plt.plot(F1_t, label='F1_t')
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, len(F1_b))
            plt.gcf().set_size_inches(20, 2.5)
            plt.savefig(META_RESULTS_FOLDER + logID + '_f1.png')
            plt.savefig(META_RESULTS_FOLDER + logID + '_f1.svg', format='svg', dpi=1200)

            # Clear the plot
            plt.clf()

def sensitivityPlot():
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
            filter_variable_txt = 'r'
        elif filter == 'SOR':
            filter_values = SOR_VALUES
            filter_variable_txt = 's'
        elif filter == 'DROR':
            filter_values = DROR_VALUES
            filter_variable_txt = r'$\gamma$'

        AccIoU_t_b = [[] for _ in filter_values]
        AccIoU_t = [[] for _ in filter_values]
        AccIoU_b = [[] for _ in filter_values]
        AccF1_t_b = [[] for _ in filter_values]
        AccF1_t = [[] for _ in filter_values]
        AccF1_b = [[] for _ in filter_values]
        AccRecall_t_b = [[] for _ in filter_values]
        AccRecall_t = [[] for _ in filter_values]
        AccRecall_b = [[] for _ in filter_values]
        AccPrecision_t_b = [[] for _ in filter_values]
        AccPrecision_t = [[] for _ in filter_values]
        AccPrecision_b = [[] for _ in filter_values]
        
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
                with open(folder + 'IoU_t_b.csv') as f:
                    IoU_t_b = np.array([line.split(",") for line in f]).astype(float)
                with open(folder + 'IoU_t.csv') as f:
                    IoU_t = np.array([line.split(",") for line in f]).astype(float)
                with open(folder + 'IoU_b.csv') as f:
                    IoU_b = np.array([line.split(",") for line in f]).astype(float)
                with open(folder + 'f1_t_b.csv') as f:
                    F1_t_b = np.array([line.split(",") for line in f]).astype(float)
                with open(folder + 'f1_t.csv') as f:
                    F1_t = np.array([line.split(",") for line in f]).astype(float)
                with open(folder + 'f1_b.csv') as f:
                    F1_b = np.array([line.split(",") for line in f]).astype(float)
                with open(folder + 'recall_t_b.csv') as f:
                    Recall_t_b = np.array([line.split(",") for line in f]).astype(float)
                with open(folder + 'recall_t.csv') as f:
                    Recall_t = np.array([line.split(",") for line in f]).astype(float)
                with open(folder + 'recall_b.csv') as f:
                    Recall_b = np.array([line.split(",") for line in f]).astype(float)
                with open(folder + 'precision_t_b.csv') as f:
                    Precision_t_b = np.array([line.split(",") for line in f]).astype(float)
                with open(folder + 'precision_t.csv') as f:
                    Precision_t = np.array([line.split(",") for line in f]).astype(float)
                with open(folder + 'precision_b.csv') as f:
                    Precision_b = np.array([line.split(",") for line in f]).astype(float)

                # Append each value to the accumulators
                AccIoU_t_b[i].extend(IoU_t_b)
                AccIoU_t[i].extend(IoU_t)
                AccIoU_b[i].extend(IoU_b)
                AccF1_t_b[i].extend(F1_t_b)
                AccF1_t[i].extend(F1_t)
                AccF1_b[i].extend(F1_b)
                AccRecall_t_b[i].extend(Recall_t_b)
                AccRecall_t[i].extend(Recall_t)
                AccRecall_b[i].extend(Recall_b)
                AccPrecision_t_b[i].extend(Precision_t_b)
                AccPrecision_t[i].extend(Precision_t)
                AccPrecision_b[i].extend(Precision_b)

        # Remove NaN values
        AccIoU_t_b = [[x for x in AccIoU_t_b[i] if not np.isnan(x)] for i in range(len(AccIoU_t_b))]
        AccIoU_t = [[x for x in AccIoU_t[i] if not np.isnan(x)] for i in range(len(AccIoU_t))]
        AccIoU_b = [[x for x in AccIoU_b[i] if not np.isnan(x)] for i in range(len(AccIoU_b))]
        AccF1_t_b = [[x for x in AccF1_t_b[i] if not np.isnan(x)] for i in range(len(AccF1_t_b))]
        AccF1_t = [[x for x in AccF1_t[i] if not np.isnan(x)] for i in range(len(AccF1_t))]
        AccF1_b = [[x for x in AccF1_b[i] if not np.isnan(x)] for i in range(len(AccF1_b))]
        AccRecall_t_b = [[x for x in AccRecall_t_b[i] if not np.isnan(x)] for i in range(len(AccRecall_t_b))]
        AccRecall_t = [[x for x in AccRecall_t[i] if not np.isnan(x)] for i in range(len(AccRecall_t))]
        AccRecall_b = [[x for x in AccRecall_b[i] if not np.isnan(x)] for i in range(len(AccRecall_b))]
        AccPrecision_t_b = [[x for x in AccPrecision_t_b[i] if not np.isnan(x)] for i in range(len(AccPrecision_t_b))]
        AccPrecision_t = [[x for x in AccPrecision_t[i] if not np.isnan(x)] for i in range(len(AccPrecision_t))]
        AccPrecision_b = [[x for x in AccPrecision_b[i] if not np.isnan(x)] for i in range(len(AccPrecision_b))]
        
        # Compute the mean IoU for each value
        meanIoU_t_b = [np.mean(AccIoU_t_b[i]) for i in range(len(AccIoU_t_b))]
        meanIoU_t = [np.mean(AccIoU_t[i]) for i in range(len(AccIoU_t))]
        meanIoU_b = [np.mean(AccIoU_b[i]) for i in range(len(AccIoU_b))]
        meanF1_t_b = [np.mean(AccF1_t_b[i]) for i in range(len(AccF1_t_b))]
        meanF1_t = [np.mean(AccF1_t[i]) for i in range(len(AccF1_t))]
        meanF1_b = [np.mean(AccF1_b[i]) for i in range(len(AccF1_b))]
        meanRecall_t_b = [np.mean(AccRecall_t_b[i]) for i in range(len(AccRecall_t_b))]
        meanRecall_t = [np.mean(AccRecall_t[i]) for i in range(len(AccRecall_t))]
        meanRecall_b = [np.mean(AccRecall_b[i]) for i in range(len(AccRecall_b))]
        meanPrecision_t_b = [np.mean(AccPrecision_t_b[i]) for i in range(len(AccPrecision_t_b))]
        meanPrecision_t = [np.mean(AccPrecision_t[i]) for i in range(len(AccPrecision_t))]
        meanPrecision_b = [np.mean(AccPrecision_b[i]) for i in range(len(AccPrecision_b))]

        ##############################################################################################################
        # MEAN IOU PLOT
        ##############################################################################################################

        # Plot the mean IoU
        plt.plot(filter_values, meanIoU_t, label=filter + ' + TGM / unfiltered', marker='o', color='red')
        plt.plot(filter_values, meanIoU_b, label=filter + ' / unfiltered', marker='o', color='blue')
        plt.plot(filter_values, meanIoU_t_b, label='TGM / ' + filter, marker='o', color='black')

        # Set the legend
        plt.legend()
        plt.ylim(0, 1)

        plt.xlabel(filter_variable_txt)
        plt.ylabel('IoU')

        # Set the size of the plot
        plt.gcf().set_size_inches(10, 2.5)

        # Set xticks
        plt.xticks(filter_values)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(META_RESULTS_FOLDER + 'Sensitivity' + filter + '_IoU.png')
        plt.savefig(META_RESULTS_FOLDER + 'Sensitivity' + filter + '_IoU.svg', format='svg', dpi=1200)

        # Clear the plot
        plt.clf()

        ##############################################################################################################
        # MEAN F1 PLOT
        ##############################################################################################################

        # Plot the mean F1
        plt.plot(filter_values, meanF1_t, label=filter + ' + TGM / unfiltered', marker='o', color='red')
        plt.plot(filter_values, meanF1_b, label=filter + ' / unfiltered', marker='o', color='blue')
        plt.plot(filter_values, meanF1_t_b, label='TGM / ' + filter, marker='o', color='black')

        # Set the legend in the bottom left
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0))
        plt.ylim(0, 1)

        plt.xlabel(filter_variable_txt)
        plt.ylabel('F1')

        # Set the size of the plot
        plt.gcf().set_size_inches(10, 2.5)

        # Set xticks
        plt.xticks(filter_values)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(META_RESULTS_FOLDER + 'Sensitivity' + filter + '_F1.png')
        plt.savefig(META_RESULTS_FOLDER + 'Sensitivity' + filter + '_F1.svg', format='svg', dpi=1200)

        # Clear the plot
        plt.clf()

        ##############################################################################################################
        # PRECISION RECALL PLOT
        ##############################################################################################################

        # Plot precision vs recall
        plt.plot(meanRecall_t, meanPrecision_t, label=filter + ' + TGM / unfiltered', marker='s', color='red')
        plt.plot(meanRecall_b, meanPrecision_b, label=filter + ' / unfiltered', marker='o', color='blue')

        # Draw lines connecting each recall-precision point from the TGM filter to the baseline filter
        #for i in range(len(filter_values)):
        #    plt.plot([meanRecall_t[i], meanRecall_b[i]], [meanPrecision_t[i], meanPrecision_b[i]], color='black', linestyle='--')

        # Write the value of the filter at each point
        for i, txt in enumerate(filter_values):
            plt.annotate('  ' + filter_variable_txt + ' = ' + str(txt) + '   ', (meanRecall_t[i], meanPrecision_t[i]), color='red', ha='left', va='bottom')
            plt.annotate('  ' + filter_variable_txt + ' = ' + str(txt) + '   ', (meanRecall_b[i], meanPrecision_b[i]), color='blue', ha='right', va='top')

        # Set the legend in the bottom left
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0))
        #plt.ylim(min(min(meanPrecision_t), min(meanPrecision_b)), max(max(meanPrecision_t), max(meanPrecision_b)))
        #plt.xlim(min(min(meanRecall_t), min(meanRecall_b)), max(max(meanRecall_t), max(meanRecall_b)))

        plt.xlabel('Recall')
        plt.ylabel('Precision')

        # Set the size of the plot
        plt.gcf().set_size_inches(7, 3.5)

        # Set ticks to not have more than 2 decimal places
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        # Add grid lines
        plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.minorticks_on()

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(META_RESULTS_FOLDER + 'Sensitivity' + filter + '_PrecisionRecall.png')
        plt.savefig(META_RESULTS_FOLDER + 'Sensitivity' + filter + '_PrecisionRecall.svg', format='svg', dpi=1200)

        # Clear the plot
        plt.clf()

def sumaryTable():
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
    for filter in ['ROR', 'SOR', 'DROR']:
        AccIoU_b = []
        AccPrecision_b = []
        AccRecall_b = []
        AccF1_b = []
        AccIoU_t = []
        AccPrecision_t = []
        AccRecall_t = []
        AccF1_t = []

        # For each filter
        for log in VALID_LOGS:
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

            # Append each value to the accumulators
            AccIoU_b.extend(IoU_b)
            AccPrecision_b.extend(Precision_b)
            AccRecall_b.extend(Recall_b)
            AccF1_b.extend(F1_b)
            AccIoU_t.extend(IoU_t)
            AccPrecision_t.extend(Precision_t)
            AccRecall_t.extend(Recall_t)
            AccF1_t.extend(F1_t)

        # Compute the mean values
        meanIoU_b = np.mean(AccIoU_b)
        meanPrecision_b = np.mean(AccPrecision_b)
        meanRecall_b = np.mean(AccRecall_b)
        meanF1_b = np.mean(AccF1_b)
        meanIoU_t = np.mean(AccIoU_t)
        meanPrecision_t = np.mean(AccPrecision_t)
        meanRecall_t = np.mean(AccRecall_t)
        meanF1_t = np.mean(AccF1_t)

        # Print the values
        print('Filter: ' + filter)
        print('IoU_b: ' + str(meanIoU_b))
        print('Precision_b: ' + str(meanPrecision_b))
        print('Recall_b: ' + str(meanRecall_b))
        print('F1_b: ' + str(meanF1_b))
        print('IoU_t: ' + str(meanIoU_t))
        print('Precision_t: ' + str(meanPrecision_t))
        print('Recall_t: ' + str(meanRecall_t))
        print('F1_t: ' + str(meanF1_t))
        print('')


if __name__ == '__main__':
    detailedPlot()
    sensitivityPlot()
    sumaryTable()