from run import run
from utilities import loadConfigAsDict
import os

#DATASET_ROOT = './snowyKITTI/dataset/sequences/'
DATASET_ROOT = 'D:/snowyKITTI/dataset/sequences/'

VALID_LOGS = [0, 2, 3, 5, 7, 8, 9, 11, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25]

def snowRunLoop():
    configPath = './config/'
    # Load default config file
    defConfFile = 'config'
    conf = loadConfigAsDict(configPath, defConfFile)

    # Load specific config file
    snowConfig = 'snowyKitti'
    specificConf = loadConfigAsDict(configPath, snowConfig)

    # Update default config file with specific config file
    conf.__dict__.update(specificConf.__dict__)

    for log in VALID_LOGS:
        # Update the paths for the lidar and the labels
        conf.lidarPath = DATASET_ROOT + str(log).zfill(2) + '/snow_velodyne/'
        conf.labelPath = DATASET_ROOT + str(log).zfill(2) + '/snow_labels/'

        # Update the simulation horizon
        files = [f for f in os.listdir(conf.lidarPath) if f.endswith('.bin')]
        conf.simHorizon = len(files)

        # Update the filter
        for filter in ['ROR', 'SOR', 'DROR']:
            if filter == 'ROR':
                conf.isROR = True
                conf.isSOR = False
                conf.isDROR = False
                logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(conf.ROR_k) + '-r-' + str(conf.ROR_r)
            elif filter == 'SOR':
                conf.isROR = False
                conf.isSOR = True
                conf.isDROR = False
                logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(conf.SOR_k) + '-s-' + str(conf.SOR_s)
            elif filter == 'DROR':
                conf.isROR = False
                conf.isSOR = False
                conf.isDROR = True
                logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(conf.DROR_k) + '-rho-' + str(conf.DROR_rho)

            print('Running simulation ' + str(log) + ' with filter ' + filter)

            # Run the simulation
            run(logID, conf)

if __name__ == '__main__':
    snowRunLoop()