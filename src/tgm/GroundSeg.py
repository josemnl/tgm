'''Implementation of Random Markov Field ground segmentation described in:

G. Postica, A. Romanoni, and M. Matteucci. Robust moving objects detection in LiDAR
data exploiting visual cues. In IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS), pages 1093-1098, 2016.

INPUT: point_cloud [num_points, 3] LiDAR data (number of LiDAR points, (x,y,z))
OUTPUT: point_cloud_seg [seg_points, 3] LiDAR points that are not 'ground'.

'''

import numpy as np
import math
import itertools
import copy
import pickle
import os
import time

CACHE_NEIGHBORS = 'neighbors_cache.pkl'
CACHE_CIRCLE = 'circle_cache.pkl'

def save_cache(cache, cache_file):
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)

def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return {}

def ground_seg(point_cloud, res=1./3., s=0.09):

	num_points = point_cloud.shape[0]
	
	# Load the cache
	time_start = time.time()
	neighbors_cache = load_cache(CACHE_NEIGHBORS)
	print("Time to load cache: ", time.time() - time_start)
	is_cache_neighbors = bool(neighbors_cache)
	circle_cache = load_cache(CACHE_CIRCLE)
	is_cache_circle = bool(circle_cache)

	# generate 2-D grid of the LiDAR cloud
	max_index = math.sqrt(2.)*(128/3./2.+1.)

	# a 2D array that contains lists of 3D points in point_cloud that map to 
	# a particular grid cell (according to the place of the 3D point in point_cloud)
	filler = np.frompyfunc(lambda x: list(), 1, 1)
	grid = np.empty((int(2 * math.ceil(max_index/res) + 1), int(2 * math.ceil(max_index/res) + 1)), dtype=object)
	filler(grid, grid)

	# determine the center coordinate of the 2D grid
	center_x = int(math.ceil(max_index/res))
	center_y = int(math.ceil(max_index/res))

	time_start = time.time()
	for i in range(num_points):
		point = point_cloud[i,:]
		x = point[0]
		y = point[1]
		z = point[2]

		if ((math.fabs(x) <= max_index) and (math.fabs(y) <= max_index) and (z <= 3.5)):
		
			grid[int(center_x + round(x/res)), int(center_y + round(y/res))].append(i)
	print("Time to fill grid: ", time.time() - time_start)

	h_G = np.nan*np.empty((grid.shape))
	
	# iterate radially outwards to compute if a point belongs to the ground (1) on mask grid  
	grid_seg = np.zeros(grid.shape)

	# initialize the center coordinate of the 2D grid to ground
	points_z = np.ndarray.tolist(point_cloud[grid[center_x, center_y],2])
	H = max(points_z or [np.nan])
	
	if not math.isnan(H):
		h_G[center_x, center_y] = H
	else:
		# initialize to the z-height of the LiDAR accroding to the KITTI set-up
		h_G[center_x, center_y] = 0.5

	# initialize the coordinates of inner circle
	circle_inner = [[center_x, center_y]]

	# identify all the points that were labeled as not ground
	point_cloud_seg_list = []
	point_cloud_ground_list = []

	time_start = time.time()
	for i in range(1,int(math.ceil(max_index/res))+1):

		# generate indices at the ith inner circle level
		if i in circle_cache:
			circle_curr = circle_cache[i]
		else:
			circle_curr = generate_circle(i, center_x, center_y)
			circle_cache[i] = circle_curr

		for indices in circle_curr:
			x = indices[0]
			y = indices[1]

			# compute h_hat_G: find max h_G of neighbors
			# Use cached neighbors if available
			if (i, x, y) in neighbors_cache:
				neigh_indeces = neighbors_cache[(i, x, y)]
			else:
				neigh_indeces = np.array(get_neighbors(x,y,circle_inner))				# THIS
				neighbors_cache[(i, x, y)] = neigh_indeces
		
			# compute the min and max z coordinates of each grid cell		
			points_z = point_cloud[grid[x, y], 2]
			H = np.nanmax(points_z) if points_z.size > 0 else np.nan
			h = np.nanmin(points_z) if points_z.size > 0 else np.nan

			h_hat_G = np.nanmax(h_G[neigh_indeces])	

			if ((not np.isnan(H)) and (not np.isnan(h)) and \
				(H - h < s) and (H - h_hat_G < s)):
				grid_seg[x,y] = 1
				h_G[x,y] = H
				point_locations = grid[x,y]
				if point_locations:
					point_cloud_ground_list.append(point_cloud[point_locations, :])
			else:

				h_G[x,y] = h_hat_G

				# add to not ground points
				point_locations = grid[x,y]
									
				if point_locations:
					point_cloud_seg_list.append(point_cloud[point_locations, :])
				
		# update the inner circle indices
		circle_inner = circle_curr
	print("Time to segment ground: ", time.time() - time_start)

	# Convert lists to arrays
	point_cloud_ground = np.vstack(point_cloud_ground_list) if point_cloud_ground_list else np.empty((0, 3))
	point_cloud_seg = np.vstack(point_cloud_seg_list) if point_cloud_seg_list else np.empty((0, 3))

	# Save the cache
	if not is_cache_neighbors:
		save_cache(neighbors_cache, CACHE_NEIGHBORS)
		print("Cache saved")
	if not is_cache_circle:
		save_cache(circle_cache, CACHE_CIRCLE)
		print("Cache saved")
	
	return point_cloud_ground, point_cloud_seg

# return the indices of a circle at level i from the center of the grid
def generate_circle(i, center_x, center_y):

	circle_range = range(-1*i,i+1)
	circle = [list(x) for x in itertools.product(circle_range, circle_range)]
	circle = [[item[0]+center_x, item[1]+center_y] for item in circle if ((abs(item[0]) == i) or (abs(item[1]) == i))]		
	
	return circle

# get the inner circle neighbors of a point
def get_neighbors(x,y,circle_inner): 
	neigh_indices = []
	for indices in circle_inner:
		if ((abs(x-indices[0]) < 2) and (abs(y-indices[1]) < 2)):
			neigh_indices.append(indices)

	return neigh_indices