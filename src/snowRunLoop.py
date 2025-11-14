from run import run
from utilities import loadConfigAsDict
import os
import copy
import concurrent.futures

# Check if CuPy is available
try:
    import cupy as cp
    _cupy_available = True
except ImportError:
    _cupy_available = False

DATASET_ROOT = './snowyKITTI/dataset/sequences/'
POSES_ROOT = './snowyKITTI_poses/'

VALID_LOGS = [0, 2, 3, 5, 7, 8, 9, 11, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25]

FILTERS = ['ROR', 'SOR', 'DROR']

MAX_WORKERS = 12

IS_PARALLEL = False

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
    
    # Check for GPU configuration mismatch
    if hasattr(conf, 'isGPU') and conf.isGPU and not _cupy_available:
        print("⚠ WARNING: Config has isGPU=True but CuPy is not installed.")
        print("  → Code will run in CPU mode. Install CuPy for GPU acceleration.")
        print()

    tasks = []

    for log in VALID_LOGS:
        # Update the paths for the lidar and the labels
        conf.lidarPath = DATASET_ROOT + str(log).zfill(2) + '/snow_velodyne/'
        conf.labelPath = DATASET_ROOT + str(log).zfill(2) + '/snow_labels/'
        conf.posePath = POSES_ROOT + str(log).zfill(2) + '/snow_pose/'

        # Update the simulation horizon
        files = [f for f in os.listdir(conf.lidarPath) if f.endswith('.bin')]
        conf.simHorizon = len(files)

        # Update the filter
        for filter in FILTERS:
            newConf = copy.deepcopy(conf)
            if filter == 'ROR':
                newConf.isROR = True
                newConf.isSOR = False
                newConf.isDROR = False
                logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(newConf.ROR_k) + '-r-' + str(newConf.ROR_r)
            elif filter == 'SOR':
                newConf.isROR = False
                newConf.isSOR = True
                newConf.isDROR = False
                logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(newConf.SOR_k) + '-s-' + str(newConf.SOR_s)
            elif filter == 'DROR':
                newConf.isROR = False
                newConf.isSOR = False
                newConf.isDROR = True
                logID = 'SnowyKitti-' + str(log).zfill(2) + '-' + filter + '-k-' + str(newConf.DROR_k) + '-rho-' + str(newConf.DROR_rho)

            print('Adding task: ' + logID)

            tasks.append((logID, newConf))

    if IS_PARALLEL:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(run, task[0], task[1]) for task in tasks]
            concurrent.futures.wait(futures)
    else:
        for task in tasks:
            run(task[0], task[1])

if __name__ == '__main__':
    snowRunLoop()