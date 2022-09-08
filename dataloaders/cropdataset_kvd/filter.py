import csv
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sample_matches import load_crops
from dataloaders.cropdataset_kvd.cfg2 import *


def load_matching_crops(path):
	""" Loads pairs of crops from a csv file. """

	print('Loading cached crop matches from {}'.format(path))
	src_crops = []
	dst_crops = []
	with open(path) as file:
		reader = csv.DictReader(file)
		for row in reader:
			src_crops.append((row['src_path'], int(row['src_r0']), int(row['src_r1']), int(row['src_c0']), int(row['src_c1'])))
			dst_crops.append((row['dst_path'], int(row['dst_r0']), int(row['dst_r1']), int(row['dst_c0']), int(row['dst_c1'])))
			pass
		pass

	print('Loaded {} crop matches.'.format(len(src_crops)))
	return src_crops, dst_crops


def save_matching_crops(src_crops, dst_crops, path):
	""" Saves pairs of matched crops to a csv file. """

	with open(path, 'w', newline='') as csvfile:
		fieldnames = ['src_path', 'src_r0', 'src_r1', 'src_c0', 'src_c1', 'dst_path', 'dst_r0', 'dst_r1', 'dst_c0', 'dst_c1']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for src, dst in zip(src_crops, dst_crops):
			writer.writerow({'src_path':src[0], 'src_r0':src[1], 'src_r1':src[2], 'src_c0':src[3], 'src_c1':src[4], 'dst_path':dst[0], 'dst_r0':dst[1], 'dst_r1':dst[2], 'dst_c0':dst[3], 'dst_c1':dst[4]})
			pass
		pass
	pass


def load_and_filter_matching_crops(knn_path, src_crop_path, dst_crop_path, max_dist=1.0):
	""" Loads crop info from source and target datasets and knn matches between crops and filters the matches based on distance. 

	knn_path -- Path to knn matches
	src_crop_path -- Path to csv file with crop info from source dataset.
	dst_crop_path -- Path to csv file with crop info from target dataset.
	max_dist -- maximum distance in feature space between neighbours.

	"""

	# logger.debug(f'Filtering matches from {knn_path}.')
	# logger.debug(f'  Source crops from {src_crop_path}.')
	# logger.debug(f'  Target crops from {dst_crop_path}.')

	data     = np.load(knn_path)
	dst_distances = data['dist'] # may need to add more samples
	dst_indices   = data['ind']

	# logger.debug(f'  Found {dst_distances.shape[0]} source crops with {dst_distances.shape[1]} neighbours each.')

	all_src_crops = load_crops(src_crop_path)
	all_dst_crops = load_crops(dst_crop_path)

	# take only patches with small distance
	# print((dst_distances < max_dist)[0])
	src_ids, knn = np.nonzero(dst_distances < max_dist)

	filtered_src_crops = []
	filtered_dst_crops = []


	for i in tqdm(range(src_ids.shape[0])):
		# print(len(all_dst_crops[1][2]))
		src_id = int(src_ids[i])
		dst_id = int(dst_indices[src_id, int(knn[i])])
		# print(all_src_crops[0][2])

		filtered_src_crops.append((all_src_crops[0][src_id],)+all_src_crops[1][src_id])
		filtered_dst_crops.append((all_dst_crops[0][dst_id],)+all_dst_crops[1][dst_id])
		pass
		
	return filtered_src_crops, filtered_dst_crops


if __name__ == '__main__':

	max_dist = 1.0
	knn_path = match_path
	src_crop_path = fake_path
	dst_crop_path = real_path

	sc, dc = load_and_filter_matching_crops(knn_path, src_crop_path, dst_crop_path, max_dist)
	save_matching_crops(sc, dc, matched_crop_path)
	pass
