import os

from tgm.utilities import read3DLidarCSV, read3DLidarBIN, read3DLabledLidarBIN, loadConfigAsDict


def _load_scan(conf, index):
    if conf.lidarFormat == 'CSV':
        path = os.path.join(conf.lidarPath, f"z_{index}.csv")
        return read3DLidarCSV(path)
    if conf.lidarFormat == 'BIN':
        file_stem = f"{index:06d}.bin"
        lidar_file = os.path.join(conf.lidarPath, file_stem)
        if conf.isLabeled:
            label_file = os.path.join(conf.labelPath, f"{index:06d}.label")
            return read3DLabledLidarBIN(lidar_file, label_file)
        return read3DLidarBIN(lidar_file)
    raise ValueError(f"Invalid lidar format: {conf.lidarFormat}")


def main():
    config_path = './config/'
    def_conf_file = 'config'
    log_id = 'Exp2-TGM-GPU'

    conf = loadConfigAsDict(config_path, def_conf_file)
    specific_conf = loadConfigAsDict(config_path, log_id)
    conf.__dict__.update(specific_conf.__dict__)

    if not conf.is3D:
        raise RuntimeError('test_ground expects a 3D lidar log (conf.is3D=False)')


    start = int(conf.initialTimeStep)
    if int(conf.simHorizon) == 0:
        indices = list(range(start, start + 5))
    else:
        horizon = int(conf.simHorizon)
        steps = 5
        if horizon <= steps:
            indices = list(range(start, start + horizon))
        else:
            stride = max(horizon // (steps - 1), 1)
            indices = [start + k * stride for k in range(steps)]
            max_index = start + horizon - 1
            indices = [min(idx, max_index) for idx in indices]

    for i in indices:
        print(f"Processing {i:06d}")
        scan = _load_scan(conf, i)

        # Match the pre-filtering used in the main pipeline.
        scan.removeClosePoints(conf.minDistance)
        scan.removeFarPoints(conf.maxDistance)
        scan.removeSky(conf.skyThreshold)

        ground, _ = scan.BEV_GroundSeg(grid_res=conf.resolution)
        ground.plot_o3d(title="BEV", compare_scan=scan)


if __name__ == '__main__':
    main()
