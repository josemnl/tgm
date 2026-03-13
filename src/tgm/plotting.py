from typing import TYPE_CHECKING
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if TYPE_CHECKING:
    from .TGM import TGM
    from .gridMap import gridMap
    from .gridMap import discreteDist

'''
This file contains the plotting functions for the TGM repository.
All functions get the object they plot and the axes to plot on as input, plus some optional parameters.
'''

def discreteDist_plot(dist: 'discreteDist', ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    ax.clear()
    # Use full-width bars without edges to avoid aliasing gaps in PNGs
    ax.bar(
        dist.values,
        dist.probabilities,
        width=1.0,
        align='center',
        alpha=1.0,
        edgecolor='none',
        linewidth=0,
        antialiased=False,
    )
    ax.set_title('P(S = s)')
    plt.show(block=False)

def gridMap_plot2D(gM: 'gridMap', ax = None, frame = None, isPause: bool = False):
    assert gM.is2D
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    if frame is None:
        frame = gM.frame
    overlap = gM.frame.computeOverlap(frame)

    # Keep data on GPU if available
    xp = cp if gM.isGPU else np

    # Crop the map to the overlapping region and transfer to CPU if needed
    croppedMap = gM.crop(overlap).data

    I = xp.zeros((overlap.size.h, overlap.size.w))
    I[:,:] = 1 - np.transpose(croppedMap)

    if gM.isGPU:
        I = cp.asnumpy(I)
    
    plt.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                extent=(gM.frame.origin.x*gM.frame.r, (gM.frame.origin.x + gM.frame.size.w)*gM.frame.r,
                        gM.frame.origin.y*gM.frame.r, (gM.frame.origin.y + gM.frame.size.h)*gM.frame.r))
    plt.show(block=isPause)
    plt.pause(0.0001)

def gridMap_plot3D(gM: 'gridMap', ax = None, frame = None, isPause: bool = False, value_min: float = 0.0, value_max: float = 1.0):
    """
    3D scatter representation: place a marker at the center of each voxel cell.
    - Color = grayscale 1 - value (imshow-like)
    - Alpha scales with occupancy (alpha_min..alpha_max)
    - Marker size scales with occupancy (s_min..s_max) to hint density
    - Only plot values in [value_min..value_max] range (default 0..1)

    Drawn per z-slice back-to-front to improve blending.
    """
    assert gM.is3D
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if frame is None:
        frame = gM.frame
    overlap = gM.frame.computeOverlap(frame)
    
    # Crop the map to the overlapping region (keep on GPU if available)
    croppedMap = gM.crop(overlap).data

    # Keep data on GPU if available, use xp abstraction for all operations
    if gM.isGPU:
        xp = cp
    else:
        xp = np
        
    values = xp.clip(croppedMap.astype(float), 0.0, 1.0)
    inten = 1.0 - values
    alpha = values
    sizes = xp.full(values.shape, 120.0)

    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass

    # Compute centers in index units (can scale to meters if desired)
    xs = overlap.origin.x + xp.arange(overlap.size.w) + 0.5
    ys = overlap.origin.y + xp.arange(overlap.size.h) + 0.5
    zs = overlap.origin.z + xp.arange(overlap.size.d) + 0.5

    # Shapes
    nx, ny, nz = len(xs), len(ys), len(zs)

    # 1D coordinates consistent with arr.ravel(order='C') for shape (nx, ny, nz)
    X = xp.repeat(xs, ny * nz)
    Y = xp.tile(xp.repeat(ys, nz), nx)
    Z = xp.tile(zs, nx * ny)

    # Per-point RGBA and sizes
    rgba = xp.zeros(inten.shape + (4,), dtype=float)
    rgba[..., 0] = inten
    rgba[..., 1] = inten
    rgba[..., 2] = inten
    rgba[..., 3] = alpha

    # Mask out low and high values (on GPU if available)
    mask = (values >= value_min) & (values <= value_max)
    mask = mask.ravel()

    # Transfer to CPU only after masking (only the filtered results)
    if gM.isGPU:
        X = cp.asnumpy(X[mask])
        Y = cp.asnumpy(Y[mask])
        Z = cp.asnumpy(Z[mask])
        sizes_masked = cp.asnumpy(sizes.ravel()[mask])
        rgba_masked = cp.asnumpy(rgba.reshape(-1, 4)[mask])
    else:
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        sizes_masked = sizes.ravel()[mask]
        rgba_masked = rgba.reshape(-1, 4)[mask]

    ax.scatter(X, Y, Z,
                s=sizes_masked,
                c=rgba_masked,
                marker='o',
                depthshade=False)

    # Limits & aspect
    ax.set_xlim(overlap.origin.x, overlap.origin.x + overlap.size.w)
    ax.set_ylim(overlap.origin.y, overlap.origin.y + overlap.size.h)
    ax.set_zlim(overlap.origin.z, overlap.origin.z + overlap.size.d)
    try:
        ax.set_box_aspect((overlap.size.w, overlap.size.h, overlap.size.d))
    except Exception:
        try:
            ax.set_aspect('equal')
        except Exception:
            pass

    tick_interval = max(1, int(round(1 / overlap.r))) if overlap.r > 0 else 1
    ax.set_xticks(np.arange(overlap.origin.x, overlap.origin.x + overlap.size.w + 1, tick_interval))
    ax.set_yticks(np.arange(overlap.origin.y, overlap.origin.y + overlap.size.h + 1, tick_interval))
    ax.set_zticks(np.arange(overlap.origin.z, overlap.origin.z + overlap.size.d + 1, tick_interval))

    plt.show(block=isPause)
    plt.pause(0.0001)

def gridMap_plot3D_cubes(gM, isPause: bool = False, cube_size: float = 1.0,
                   alpha_min: float = 0.0, alpha_max: float = 1.0,
                   face_edges: bool = False, elev: float = 20, azim: float = -60) -> None:
    """
    3D glyph scatter with cubes: draw a smaller cube centered in each cell.
    - cube_size in (0, 1]: side-length relative to cell size (default 0.6)
    - Color = 1 - value (imshow-like grayscale)
    - Alpha scales with occupancy (alpha_min..alpha_max)
    - Optional edges (off by default for speed)

    Renders per z-slice back-to-front for decent transparency blending.
    """
    data = cp.asnumpy(gM.data) if gM.isGPU else gM.data
    values = np.clip(data.astype(float), 0.0, 1.0)
    inten = 1.0 - values
    alpha = alpha_min + (alpha_max - alpha_min) * values

    half = float(cube_size) / 2.0
    # Clamp to (0, 0.5]; 0.5 means cubes span the full cell and touch
    if half <= 0:
        half = 1e-3
    if half > 0.5:
        half = 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)

    def cube_faces(cx, cy, cz, h):
        x0, x1 = cx - h, cx + h
        y0, y1 = cy - h, cy + h
        z0, z1 = cz - h, cz + h
        return [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]

    # Centers per axis
    xs = gM.frame.origin.x + np.arange(gM.frame.size.w) + 0.5
    ys = gM.frame.origin.y + np.arange(gM.frame.size.h) + 0.5
    zs = gM.frame.origin.z + np.arange(gM.frame.size.d) + 0.5

    for k in range(gM.frame.size.d):
        zc = zs[k]
        polys = []
        face_cols = []
        edge_cols = []
        for i in range(gM.frame.size.w):
            xc = xs[i]
            for j in range(gM.frame.size.h):
                yc = ys[j]
                col = inten[i, j, k]
                a = alpha[i, j, k]
                faces = cube_faces(xc, yc, zc, half)
                rgba = (col, col, col, a)
                for f in faces:
                    polys.append(f)
                    face_cols.append(rgba)
                    edge_cols.append((0, 0, 0, 0.25) if face_edges else (0, 0, 0, 0))
        if polys:
            coll = Poly3DCollection(polys, facecolors=face_cols, edgecolors=edge_cols)
            coll.set_alpha(None)
            try:
                coll.set_zsort('none')
            except Exception:
                pass
            ax.add_collection3d(coll)

    ax.set_xlim(gM.frame.origin.x, gM.frame.origin.x + gM.frame.size.w)
    ax.set_ylim(gM.frame.origin.y, gM.frame.origin.y + gM.frame.size.h)
    ax.set_zlim(gM.frame.origin.z, gM.frame.origin.z + gM.frame.size.d)
    try:
        ax.set_box_aspect((gM.frame.size.w, gM.frame.size.h, gM.frame.size.d))
    except Exception:
        try:
            ax.set_aspect('equal')
        except Exception:
            pass

    tick_interval = max(1, int(round(1 / gM.frame.r))) if gM.frame.r > 0 else 1
    ax.set_xticks(np.arange(gM.frame.origin.x, gM.frame.origin.x + gM.frame.size.w + 1, tick_interval))
    ax.set_yticks(np.arange(gM.frame.origin.y, gM.frame.origin.y + gM.frame.size.h + 1, tick_interval))
    ax.set_zticks(np.arange(gM.frame.origin.z, gM.frame.origin.z + gM.frame.size.d + 1, tick_interval))

    plt.show(block=isPause)
    plt.pause(0.0001)

def tgm_plot2D(tgm: 'TGM', ax = None, frame = None, saveMap=False, savePNG=False, saveSvg=False, imgName='', style='combined', egoStyle='rectangle'):
    assert tgm.is2D
    assert style in ['combined', 'static', 'dynamic', 'weather']
    assert egoStyle in ['none', 'dot', 'rectangle']
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    if frame is None:
        frame = tgm.frame
    overlap = tgm.frame.computeOverlap(frame)

    # Keep data on GPU if available
    xp = cp if tgm.GPU else np

    # Crop the maps to the overlapping region
    staticMap = tgm.staticMap.crop(overlap).data
    dynamicMap = tgm.dynamicMap.crop(overlap).data
    weatherMap = tgm.weatherMap.crop(overlap).data

    # Plot the map according to the style
    if style == 'combined':
        I = xp.zeros((overlap.size.h, overlap.size.w, 3))
        I[:,:,0] = 1 - xp.transpose(1.0*staticMap + 0.0*dynamicMap + 2.0*weatherMap/xp.square(1-weatherMap))
        I[:,:,1] = 1 - xp.transpose(0.5*staticMap + 0.5*dynamicMap + 0.0*weatherMap/xp.square(1-weatherMap))
        I[:,:,2] = 1 - xp.transpose(0.0*staticMap + 1.0*dynamicMap + 2.0*weatherMap/xp.square(1-weatherMap))
        # Make sure the values are between 0 and 1
        I = xp.clip(I, 0, 1)
    elif style == 'static':
        I = 1 - xp.transpose(staticMap)
    elif style == 'dynamic':
        I = 1 - xp.transpose(dynamicMap)
    elif style == 'weather':
        I = 1 - xp.transpose(weatherMap)

    # Transfer the image to CPU if using GPU
    if tgm.GPU:
        I = cp.asnumpy(I)

    # Plot the map
    ax.clear()
    ax.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
            extent=(overlap.origin.x*tgm.frame.r, (overlap.origin.x + overlap.size.w)*tgm.frame.r,
                    overlap.origin.y*tgm.frame.r, (overlap.origin.y + overlap.size.h)*tgm.frame.r))

    # Plot the ego pose
    if tgm.x_t is not None:
        if egoStyle == 'dot':
            ax.plot(tgm.x_t.position.x, tgm.x_t.position.y, 'ro')
        elif egoStyle == 'rectangle':
            x = tgm.x_t.position.x
            y = tgm.x_t.position.y
            theta = tgm.x_t.orientation.yaw
            car_length = 4.953
            car_width = 1.923
            x1 = x + car_length/2 * np.cos(theta) + car_width/2 * np.cos(theta + np.pi/2)
            y1 = y + car_length/2 * np.sin(theta) + car_width/2 * np.sin(theta + np.pi/2)
            x2 = x + car_length/2 * np.cos(theta) - car_width/2 * np.cos(theta + np.pi/2)
            y2 = y + car_length/2 * np.sin(theta) - car_width/2 * np.sin(theta + np.pi/2)
            x3 = x - car_length/2 * np.cos(theta) - car_width/2 * np.cos(theta + np.pi/2)
            y3 = y - car_length/2 * np.sin(theta) - car_width/2 * np.sin(theta + np.pi/2)
            x4 = x - car_length/2 * np.cos(theta) + car_width/2 * np.cos(theta + np.pi/2)
            y4 = y - car_length/2 * np.sin(theta) + car_width/2 * np.sin(theta + np.pi/2)
            rectangle = plt.Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], closed=True, facecolor='white', edgecolor='black')
            ax.add_patch(rectangle)
            # Plot the heading as a triangle
            x1 = x + car_length/2 * np.cos(theta)
            y1 = y + car_length/2 * np.sin(theta)
            x2 = x + (car_length/2-car_width) * np.cos(theta) + car_width/2 * np.sin(theta)
            y2 = y + (car_length/2-car_width) * np.sin(theta) - car_width/2 * np.cos(theta)
            x3 = x + (car_length/2-car_width) * np.cos(theta) - car_width/2 * np.sin(theta)
            y3 = y + (car_length/2-car_width) * np.sin(theta) + car_width/2 * np.cos(theta)
            triangle = plt.Polygon([[x1, y1], [x2, y2], [x3, y3]], closed=True, facecolor='white', edgecolor='black')
            ax.add_patch(triangle)

    if saveMap:
        imsave(imgName + '_map.png', I, origin ="lower", cmap='gray')
    if savePNG:
        ax.figure.savefig(imgName + '.png', format='png')
    if saveSvg:
        # Exporting raster images at 1200 DPI every frame is very slow and can freeze the UI.
        ax.figure.savefig(imgName + '.svg', format='svg', dpi=1200)
    
    # Pause to show the image
    ax.figure.canvas.draw_idle()
    plt.pause(0.01)

def tgm_plot3D(tgm: 'TGM', ax = None, frame = None, value_min=0.0, value_max=1.0):
    """
    3D plot of the TGM using scatter plot.
    Very similar to the one in gridMap.py, but plotting the 3 layers each using
    a different RGB layer for the color, similarly to the 2D case.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if frame is None:
        frame = tgm.frame
    overlap = tgm.frame.computeOverlap(frame)

    # Crop the maps to the overlapping region
    staticMap = tgm.staticMap.crop(overlap).data
    dynamicMap = tgm.dynamicMap.crop(overlap).data
    weatherMap = tgm.weatherMap.crop(overlap).data

    # Keep data on GPU if using cupy, otherwise use numpy (optimized like plot3D_open3d)
    if tgm.GPU:
        xp = cp
    else:
        xp = np

    # Create a meshgrid for the coordinates (on GPU if available)
    x = xp.arange(overlap.origin.x, overlap.origin.x + overlap.size.w) * tgm.frame.r
    y = xp.arange(overlap.origin.y, overlap.origin.y + overlap.size.h) * tgm.frame.r
    z = xp.arange(overlap.origin.z, overlap.origin.z + overlap.size.d) * tgm.frame.r
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')

    # Flatten the arrays for plotting (still on GPU if applicable)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    staticMap = staticMap.flatten()
    dynamicMap = dynamicMap.flatten()
    weatherMap = weatherMap.flatten()

    # Create a color array based on the probabilities (on GPU if applicable)
    colors = xp.zeros((len(X), 3))
    # Same color coding as in the 2D case
    colors[:, 0] = 1 - (staticMap + 0.0*dynamicMap + 2.0*weatherMap/xp.square(1-weatherMap))  # Red channel
    colors[:, 1] = 1 - (0.5*staticMap + 0.5*dynamicMap + 0.0*weatherMap/xp.square(1-weatherMap))  # Green channel
    colors[:, 2] = 1 - (0.0*staticMap + 1.0*dynamicMap + 2.0*weatherMap/xp.square(1-weatherMap))  # Blue channel

    # Normalize colors to be between 0 and 1 (on GPU if applicable)
    colors = xp.clip(colors, 0, 1)

    # Mask out low and high values (on GPU if applicable)
    mask = (staticMap + dynamicMap + weatherMap > value_min) & (staticMap + dynamicMap + weatherMap < value_max)
    
    # Apply mask and transfer to CPU only once at the end (if using GPU)
    if tgm.GPU:
        X = cp.asnumpy(X[mask])
        Y = cp.asnumpy(Y[mask])
        Z = cp.asnumpy(Z[mask])
        colors = cp.asnumpy(colors[mask])
    else:
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        colors = colors[mask]

    # Before plotting, enforce equal data scale across X/Y/Z using the true extents (in meters)
    # Compute extents from the overlap frame (not from masked points)
    x_min = overlap.origin.x * tgm.frame.r
    x_max = (overlap.origin.x + overlap.size.w) * tgm.frame.r
    y_min = overlap.origin.y * tgm.frame.r
    y_max = (overlap.origin.y + overlap.size.h) * tgm.frame.r
    z_min = overlap.origin.z * tgm.frame.r
    z_max = (overlap.origin.z + overlap.size.d) * tgm.frame.r

    rx = max(x_max - x_min, 0.0)
    ry = max(y_max - y_min, 0.0)
    rz = max(z_max - z_min, 0.0)

    # Set explicit limits first (so autoscale doesn't change box-aspect)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Try to preserve the real aspect ratios (so short Z is not stretched)
    ax.set_box_aspect((rx, ry, rz))

    # Scatter plot
    ax.scatter(X, Y, Z, c=colors, marker='o', s=1)

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # Plot the ego pose
    if tgm.x_t is not None:
        ax.scatter(tgm.x_t.position.x, tgm.x_t.position.y, tgm.x_t.position.z, c='red', marker='o', s=50)

    # Plot field of view of the sensor (90 degrees horizontal, 40 degrees vertical and 15 m range)
    # The plot displays the sides of the pyramid
    if tgm.x_t is not None:
        sensor_x = tgm.x_t.position.x
        sensor_y = tgm.x_t.position.y
        sensor_z = tgm.x_t.position.z
        sensor_roll = tgm.x_t.orientation.roll
        sensor_pitch = tgm.x_t.orientation.pitch
        sensor_yaw = tgm.x_t.orientation.yaw

        fov_range = 3.0
        fov_hfov = np.deg2rad(45.0)  # Half horizontal FOV
        fov_vfov = np.deg2rad(20.0)  # Half vertical FOV

        # Rotation matrix from sensor to world frame
        R_yaw = np.array([[np.cos(sensor_yaw), -np.sin(sensor_yaw), 0],
                            [np.sin(sensor_yaw), np.cos(sensor_yaw), 0],
                            [0, 0, 1]])
        R_pitch = np.array([[np.cos(sensor_pitch), 0, np.sin(sensor_pitch)],
                            [0, 1, 0],
                            [-np.sin(sensor_pitch), 0, np.cos(sensor_pitch)]])

        R_roll = np.array([[1, 0, 0],
                            [0, np.cos(sensor_roll), -np.sin(sensor_roll)],
                            [0, np.sin(sensor_roll), np.cos(sensor_roll)]])
        R = R_yaw @ R_pitch @ R_roll

        # Define the 4 corner rays in the sensor frame (yaw/pitch offsets)
        # Order: top-left, top-right, bottom-right, bottom-left
        angles = [(-fov_hfov,  fov_vfov),
                    ( fov_hfov,  fov_vfov),
                    ( fov_hfov, -fov_vfov),
                    (-fov_hfov, -fov_vfov)]
        corners_sensor = []
        for yaw_off, pitch_off in angles:
            cpp = np.cos(pitch_off)
            dir_sensor = np.array([cpp * np.cos(yaw_off), cpp * np.sin(yaw_off), np.sin(pitch_off)])
            corners_sensor.append(fov_range * dir_sensor)
        corners = np.vstack(corners_sensor)

        # Rotate and translate corners to world frame
        world_corners = (R @ corners.T).T + np.array([sensor_x, sensor_y, sensor_z])

        # Plot the 4 sides of the pyramid
        for i in range(4):
            x_vals = [sensor_x, world_corners[i, 0]]
            y_vals = [sensor_y, world_corners[i, 1]]
            z_vals = [sensor_z, world_corners[i, 2]]
            ax.plot(x_vals, y_vals, z_vals, color='blue', linestyle='--', linewidth=1)
        
        # Plot the base of the pyramid
        for i in range(4):
            x_vals = [world_corners[i, 0], world_corners[(i+1)%4, 0]]
            y_vals = [world_corners[i, 1], world_corners[(i+1)%4, 1]]
            z_vals = [world_corners[i, 2], world_corners[(i+1)%4, 2]]
            ax.plot(x_vals, y_vals, z_vals, color='blue', linestyle='--', linewidth=1)

    # Plot the coordinate frame of the sensor at the ego position
    if tgm.x_t is not None:
        sensor_x = tgm.x_t.position.x
        sensor_y = tgm.x_t.position.y
        sensor_z = tgm.x_t.position.z
        sensor_roll = tgm.x_t.orientation.roll
        sensor_pitch = tgm.x_t.orientation.pitch
        sensor_yaw = tgm.x_t.orientation.yaw

        # Rotation matrix from sensor to world frame
        R_yaw = np.array([[np.cos(sensor_yaw), -np.sin(sensor_yaw), 0],
                            [np.sin(sensor_yaw), np.cos(sensor_yaw), 0],
                            [0, 0, 1]])
        R_pitch = np.array([[np.cos(sensor_pitch), 0, np.sin(sensor_pitch)],
                            [0, 1, 0],
                            [-np.sin(sensor_pitch), 0, np.cos(sensor_pitch)]])

        R_roll = np.array([[1, 0, 0],
                            [0, np.cos(sensor_roll), -np.sin(sensor_roll)],
                            [0, np.sin(sensor_roll), np.cos(sensor_roll)]])
        R = R_yaw @ R_pitch @ R_roll

        # Define the axes in the sensor frame
        axis_length = 1.0
        axes_sensor = np.array([[axis_length, 0, 0],
                                [0, axis_length, 0],
                                [0, 0, axis_length]])
        axes_world = (R @ axes_sensor.T).T + np.array([sensor_x, sensor_y, sensor_z])

        # Plot the axes
        ax.plot([sensor_x, axes_world[0, 0]], [sensor_y, axes_world[0, 1]], [sensor_z, axes_world[0, 2]], color='red', linewidth=2)   # X-axis
        ax.plot([sensor_x, axes_world[1, 0]], [sensor_y, axes_world[1, 1]], [sensor_z, axes_world[1, 2]], color='green', linewidth=2) # Y-axis
        ax.plot([sensor_x, axes_world[2, 0]], [sensor_y, axes_world[2, 1]], [sensor_z, axes_world[2, 2]], color='blue', linewidth=2)  # Z-axis
        
    plt.show(block=False)

def tgm_plot3D_open3d(tgm: 'TGM', frame = None, isPause=False, value_min=0.0, value_max=1.0):
    """
    3D plot of the TGM using Open3D.
    Very similar to plot3D() but using Open3D for visualization.
    Updates the same window on each call.
    Optimized to minimize GPU-CPU transfers.
    """
    import open3d as o3d

    if frame is None:
        frame = tgm.frame
    
    overlap = tgm.frame.computeOverlap(frame)

    # Crop the maps to the overlapping region
    staticMap = tgm.staticMap.crop(overlap).data
    dynamicMap = tgm.dynamicMap.crop(overlap).data
    weatherMap = tgm.weatherMap.crop(overlap).data

    # Keep data on GPU if using cupy, otherwise use numpy
    if tgm.GPU:
        xp = cp
    else:
        xp = np

    # Create a meshgrid for the coordinates (on GPU if available)
    x = xp.arange(overlap.origin.x, overlap.origin.x + overlap.size.w) * tgm.frame.r
    y = xp.arange(overlap.origin.y, overlap.origin.y + overlap.size.h) * tgm.frame.r
    z = xp.arange(overlap.origin.z, overlap.origin.z + overlap.size.d) * tgm.frame.r
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')

    # Flatten the arrays for plotting (still on GPU)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    staticMap = staticMap.flatten()
    dynamicMap = dynamicMap.flatten()
    weatherMap = weatherMap.flatten()

    # Create a color array based on the probabilities (on GPU)
    colors = xp.zeros((len(X), 3))
    colors[:, 0] = 1 - (staticMap + 0.0*dynamicMap + 2.0*weatherMap/xp.square(1-weatherMap))  # Red channel
    colors[:, 1] = 1 - (0.5*staticMap + 0.5*dynamicMap + 0.0*weatherMap/xp.square(1-weatherMap))  # Green channel
    colors[:, 2] = 1 - (0.0*staticMap + 1.0*dynamicMap + 2.0*weatherMap/xp.square(1-weatherMap))  # Blue channel

    # Normalize colors to be between 0 and 1 (on GPU)
    colors = xp.clip(colors, 0, 1)

    # Mask out low and high values (on GPU)
    mask = (staticMap + dynamicMap + weatherMap > value_min) & (staticMap + dynamicMap + weatherMap < value_max)
    
    # Apply mask and transfer only filtered data to CPU (THIS is the only GPU->CPU transfer)
    if tgm.GPU:
        points = cp.asnumpy(xp.vstack([X[mask], Y[mask], Z[mask]]).T)
        colors = cp.asnumpy(colors[mask])
    else:
        points = xp.vstack([X[mask], Y[mask], Z[mask]]).T
        colors = colors[mask]

    # Compute full map bounds (match plot3D axis limits)
    x_min = overlap.origin.x * tgm.frame.r
    x_max = (overlap.origin.x + overlap.size.w) * tgm.frame.r
    y_min = overlap.origin.y * tgm.frame.r
    y_max = (overlap.origin.y + overlap.size.h) * tgm.frame.r
    z_min = overlap.origin.z * tgm.frame.r
    z_max = (overlap.origin.z + overlap.size.d) * tgm.frame.r
    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

    # Initialize visualizer if needed
    if tgm._o3d_vis is None:
        tgm._o3d_vis = o3d.visualization.Visualizer()
        tgm._o3d_vis.create_window(window_name="TGM Open3D")
        tgm._o3d_pcd = o3d.geometry.PointCloud()
        tgm._o3d_vis.add_geometry(tgm._o3d_pcd)

    # Update point cloud geometry
    tgm._o3d_pcd.points = o3d.utility.Vector3dVector(points)
    tgm._o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
    tgm._o3d_vis.update_geometry(tgm._o3d_pcd)

    # Ensure the camera fits the full map bounds only once (initial pose)
    if not tgm._o3d_view_initialized:
        bounds_points = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max],
        ])
        bounds_pcd = o3d.geometry.PointCloud()
        bounds_pcd.points = o3d.utility.Vector3dVector(bounds_points)
        tgm._o3d_vis.add_geometry(bounds_pcd, reset_bounding_box=True)
        tgm._o3d_vis.reset_view_point(True)
        view_ctl = tgm._o3d_vis.get_view_control()
        if view_ctl is not None:
            view_ctl.set_zoom(0.8)
        tgm._o3d_vis.remove_geometry(bounds_pcd, reset_bounding_box=False)
        tgm._o3d_last_bounds = bounds
        tgm._o3d_view_initialized = True

    # Remove and re-add ego sphere
    if tgm.x_t is not None:
        if tgm._o3d_ego is not None:
            tgm._o3d_vis.remove_geometry(tgm._o3d_ego, reset_bounding_box=False)
        
        tgm._o3d_ego = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        tgm._o3d_ego.translate([tgm.x_t.position.x, tgm.x_t.position.y, tgm.x_t.position.z])
        tgm._o3d_ego.paint_uniform_color([1, 0, 0])
        tgm._o3d_vis.add_geometry(tgm._o3d_ego, reset_bounding_box=False)

    # Remove and re-add coordinate frame with proper rotation
    if tgm.x_t is not None:
        if tgm._o3d_frame is not None:
            tgm._o3d_vis.remove_geometry(tgm._o3d_frame, reset_bounding_box=False)
        
        # Create coordinate frame at origin, then transform it
        tgm._o3d_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        
        # Compute rotation matrix (same as FOV)
        sensor_roll = tgm.x_t.orientation.roll
        sensor_pitch = tgm.x_t.orientation.pitch
        sensor_yaw = tgm.x_t.orientation.yaw
        
        R_yaw = np.array([[np.cos(sensor_yaw), -np.sin(sensor_yaw), 0],
                            [np.sin(sensor_yaw), np.cos(sensor_yaw), 0],
                            [0, 0, 1]])
        R_pitch = np.array([[np.cos(sensor_pitch), 0, np.sin(sensor_pitch)],
                            [0, 1, 0],
                            [-np.sin(sensor_pitch), 0, np.cos(sensor_pitch)]])
        R_roll = np.array([[1, 0, 0],
                            [0, np.cos(sensor_roll), -np.sin(sensor_roll)],
                            [0, np.sin(sensor_roll), np.cos(sensor_roll)]])
        R = R_yaw @ R_pitch @ R_roll
        
        # Create transformation matrix (rotation + translation)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tgm.x_t.position.x, tgm.x_t.position.y, tgm.x_t.position.z]
        
        # Apply transformation
        tgm._o3d_frame.transform(T)
        tgm._o3d_vis.add_geometry(tgm._o3d_frame, reset_bounding_box=False)

    # Remove and re-add FOV pyramid
    if tgm.x_t is not None:
        # Remove old FOV lines
        if tgm._o3d_fov_lines is not None:
            for line_set in tgm._o3d_fov_lines:
                tgm._o3d_vis.remove_geometry(line_set, reset_bounding_box=False)
        tgm._o3d_fov_lines = []

        sensor_x = tgm.x_t.position.x
        sensor_y = tgm.x_t.position.y
        sensor_z = tgm.x_t.position.z
        sensor_roll = tgm.x_t.orientation.roll
        sensor_pitch = tgm.x_t.orientation.pitch
        sensor_yaw = tgm.x_t.orientation.yaw

        fov_range = 3.0
        fov_hfov = np.deg2rad(45.0)  # Half horizontal FOV
        fov_vfov = np.deg2rad(20.0)  # Half vertical FOV

        # Rotation matrix from sensor to world frame
        R_yaw = np.array([[np.cos(sensor_yaw), -np.sin(sensor_yaw), 0],
                            [np.sin(sensor_yaw), np.cos(sensor_yaw), 0],
                            [0, 0, 1]])
        R_pitch = np.array([[np.cos(sensor_pitch), 0, np.sin(sensor_pitch)],
                            [0, 1, 0],
                            [-np.sin(sensor_pitch), 0, np.cos(sensor_pitch)]])
        R_roll = np.array([[1, 0, 0],
                            [0, np.cos(sensor_roll), -np.sin(sensor_roll)],
                            [0, np.sin(sensor_roll), np.cos(sensor_roll)]])
        R = R_yaw @ R_pitch @ R_roll

        # Define the 4 corner rays in the sensor frame
        angles = [(-fov_hfov,  fov_vfov),
                    ( fov_hfov,  fov_vfov),
                    ( fov_hfov, -fov_vfov),
                    (-fov_hfov, -fov_vfov)]
        corners_sensor = []
        for yaw_off, pitch_off in angles:
            cpp = np.cos(pitch_off)
            dir_sensor = np.array([cpp * np.cos(yaw_off), cpp * np.sin(yaw_off), np.sin(pitch_off)])
            corners_sensor.append(fov_range * dir_sensor)
        corners = np.vstack(corners_sensor)

        # Rotate and translate corners to world frame
        world_corners = (R @ corners.T).T + np.array([sensor_x, sensor_y, sensor_z])

        # Create line segments for FOV pyramid
        sensor_pos = np.array([sensor_x, sensor_y, sensor_z])
        
        # Lines from sensor to corners
        for i in range(4):
            points = np.vstack([sensor_pos, world_corners[i]])
            lines = np.array([[0, 1]])
            line_set = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(points),
                o3d.utility.Vector2iVector(lines)
            )
            line_set.paint_uniform_color([0, 0, 1])  # Blue
            tgm._o3d_vis.add_geometry(line_set, reset_bounding_box=False)
            tgm._o3d_fov_lines.append(line_set)

        # Lines connecting the base of the pyramid
        for i in range(4):
            points = np.vstack([world_corners[i], world_corners[(i+1)%4]])
            lines = np.array([[0, 1]])
            line_set = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(points),
                o3d.utility.Vector2iVector(lines)
            )
            line_set.paint_uniform_color([0, 0, 1])  # Blue
            tgm._o3d_vis.add_geometry(line_set, reset_bounding_box=False)
            tgm._o3d_fov_lines.append(line_set)

    # Update renderer
    tgm._o3d_vis.poll_events()
    tgm._o3d_vis.update_renderer()

    if isPause:
        tgm._o3d_vis.run()