from run import run
from utilities import loadConfigAsDict
import os

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

    for i in range(10):
        # Update the paths for the lidar and the labels
        conf.lidarPath = './snowyKITTI/dataset/sequences/' + str(i).zfill(2) + '/snow_velodyne/'
        conf.labelPath = './snowyKITTI/dataset/sequences/' + str(i).zfill(2) + '/snow_labels/'

        # Update the simulation horizon
        files = [f for f in os.listdir(conf.lidarPath) if f.endswith('.bin')]
        conf.simHorizon = len(files)

        # Update the filter
        for filter in ['ROR', 'SOR', 'DROR']:
            if filter == 'ROR':
                conf.isROR = True
            elif filter == 'SOR':
                conf.isSOR = True
            elif filter == 'DROR':
                conf.isDROR = True

            # Update the logID
            logID = 'SnowyKitti-' + str(i).zfill(2) + '-' + filter

            # Run the simulation
            run(logID, conf)
        


if __name__ == '__main__':
    snowRunLoop()