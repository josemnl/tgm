"""
Export ROS 2 bag (db3) messages.
- PointCloud2 will be exported to .bin (TODO)
- PoseStamped will be exported to .csv (TODO)

Current action: list topics and print header.frame_id for Pose messages
using rosbags.highlevel.AnyReader, which provides a built-in deserialize().
"""

from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, stores

import os
import struct
from typing import Tuple

import numpy as np
import cv2

from utilities import read3DLidarBIN

# Path to the bag folder (the directory that contains metadata.yaml)
BAG_DIR = Path('./logs/underwater/rosbag2_2025_10_13-14_53_47/')
EXP_DIR = Path('./logs/underwater/rosbag2_2025_10_13-14_53_47/export/')

POSE_TOPIC = '/mocap/saabmarine/pose'
LIDAR_TOPIC = '/sonar/point_cloud'
CAMERA_TOPIC = '/saabmarine/camera/image_raw'


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def decode_pose(msg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode geometry_msgs/msg/PoseStamped to position and orientation arrays.
    Returns:
        position: np.ndarray of shape (3,) [x, y, z]
        orientation: np.ndarray of shape (4,) quaternion [x, y, z, w]
        header: np.ndarray of shape (2,) [sec, nanosec]
    """
    pos = msg.pose.position
    ori = msg.pose.orientation
    position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
    orientation = np.array([ori.x, ori.y, ori.z, ori.w], dtype=np.float32)
    header = np.array([msg.header.stamp.sec, msg.header.stamp.nanosec], dtype=np.int32)
    return position, orientation, header

def decode_pointcloud2_xyz(msg) -> np.ndarray:
    """Decode sensor_msgs/msg/PointCloud2 to Nx3 float32 XYZ array.

    - Supports little/big endian based on msg.is_bigendian
    - Extracts fields named 'x', 'y', 'z' (float32/float64)
    - Skips points where any of x,y,z is NaN
    """
    # Locate field offsets and datatypes
    x_field = next((f for f in msg.fields if f.name == 'x'), None)
    y_field = next((f for f in msg.fields if f.name == 'y'), None)
    z_field = next((f for f in msg.fields if f.name == 'z'), None)
    if x_field is None or y_field is None or z_field is None:
        return np.empty((0, 3), dtype=np.float32)

    # Determine struct formats
    # PointField datatypes: 7=float32, 8=float64
    def fmt_for(field) -> Tuple[str, int]:
        if field.datatype == 7:  # FLOAT32
            return ('f', 4)
        if field.datatype == 8:  # FLOAT64
            return ('d', 8)
        # Fallback: try to interpret as float32
        return ('f', 4)

    x_fmt, x_size = fmt_for(x_field)
    y_fmt, y_size = fmt_for(y_field)
    z_fmt, z_size = fmt_for(z_field)

    # Endianness
    endian = '>' if msg.is_bigendian else '<'
    s_x = struct.Struct(endian + x_fmt)
    s_y = struct.Struct(endian + y_fmt)
    s_z = struct.Struct(endian + z_fmt)

    # Iterate points
    data = memoryview(msg.data)
    step = msg.point_step
    print('Step value:', step)
    n_points = (len(msg.data) // step)
    out = np.empty((n_points, 3), dtype=np.float32)
    write_idx = 0
    for i in range(n_points):
        base = i * step
        try:
            x = s_x.unpack_from(data, base + x_field.offset)[0]
            y = s_y.unpack_from(data, base + y_field.offset)[0]
            z = s_z.unpack_from(data, base + z_field.offset)[0]
        except Exception:
            continue
        # Filter NaNs
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        out[write_idx] = (x, y, z)
        write_idx += 1

    return out[:write_idx]


def image_from_ros2(msg) -> np.ndarray:
    """Convert sensor_msgs/msg/Image to a numpy array (H,W,C) in BGR for OpenCV.
    Supports common encodings: 'rgb8', 'bgr8', 'mono8', 'mono16', 'rgba8', 'bgra8'.
    """
    enc = msg.encoding.lower() if hasattr(msg, 'encoding') and msg.encoding else 'bgr8'
    h, w = msg.height, msg.width
    step = msg.step
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    # Determine number of channels and dtype
    if enc in ('bgr8', 'rgb8'):
        ch = 3
        img = buf.reshape((h, step))[:, : w * ch].reshape((h, w, ch))
        if enc == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    if enc in ('bgra8', 'rgba8'):
        ch = 4
        img = buf.reshape((h, step))[:, : w * ch].reshape((h, w, ch))
        if enc == 'rgba8':
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    if enc in ('mono8', '8uc1'):
        img = buf.reshape((h, step))[:, : w].reshape((h, w))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc in ('mono16', '16uc1'):
        buf16 = np.frombuffer(msg.data, dtype=np.uint16)
        img16 = buf16.reshape((h, step // 2))[:, : w].reshape((h, w))
        # Normalize to 8-bit for viewing/saving
        img8 = cv2.convertScaleAbs(img16, alpha=255.0 / max(1, img16.max()))
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    # Fallback: try to guess 3 channels
    ch = 3
    img = buf.reshape((h, step))[:, : w * ch].reshape((h, w, ch))
    return img

# Open bag with high-level reader and deserialize via reader.deserialize()
typestore = get_typestore(stores.Stores.LATEST)

with AnyReader([BAG_DIR], default_typestore=typestore) as reader:
    # List topics and message types
    for topic, conn in reader.topics.items():
        print('Listing topics:')
        print(topic, conn.msgtype)

    # Iterate over pose, camera and lidar messages and export
    ensure_dir(EXP_DIR)

    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == POSE_TOPIC:
            msg = reader.deserialize(rawdata, connection.msgtype)
            try:
                position, orientation, header = decode_pose(msg)
                pose_path = EXP_DIR / f"pose_{timestamp}.csv"
                with open(pose_path, 'w') as f:
                    f.write(f"{position[0]},{position[1]},{position[2]},{orientation[0]},{orientation[1]},{orientation[2]},{orientation[3]}\n")
                print(f"Saved pose: {pose_path}")
            except Exception as e:
                print(f"[WARN] Failed to decode/save pose @ {timestamp}: {e}")

        if connection.topic == CAMERA_TOPIC:
            msg = reader.deserialize(rawdata, connection.msgtype)
            try:
                img = image_from_ros2(msg)
                img_path = EXP_DIR / f"camera_{timestamp}.png"
                cv2.imwrite(str(img_path), img)
                print(f"Saved camera frame: {img_path}")
            except Exception as e:
                print(f"[WARN] Failed to save image @ {timestamp}: {e}")

        if connection.topic == LIDAR_TOPIC:
            msg = reader.deserialize(rawdata, connection.msgtype)
            try:
                pts = decode_pointcloud2_xyz(msg)
                if pts.size == 0:
                    continue
                bin_path = EXP_DIR / f"lidar_{timestamp}.bin"
                # Save as float32 (N,3)
                pts.astype(np.float32).tofile(str(bin_path))
                print(f"Saved lidar points: {bin_path} ({len(pts)} pts)")

                # Optional: read back and basic sanity check using existing utility
                scan3D = read3DLidarBIN(str(bin_path), n_fields=3)
                print(f"Read-back points: {len(scan3D.points3D)}")
                scan3D.plot()
            except Exception as e:
                print(f"[WARN] Failed to decode/save lidar @ {timestamp}: {e}")