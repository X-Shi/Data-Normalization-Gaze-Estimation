import os
import cv2
import numpy as np
import csv
import scipy.io as sio
import h5py
from PIL import Image
import time
import argparse
import math

import configparser
import io

from datetime import datetime

data_dir = '/tudelft.net/staff-umbrella/NightId/eth-xgaze/raw'
output_dir = '/tudelft.net/staff-umbrella/NightId/eth-xgaze/raw/multi-region_3_5/test'

def draw_gaze(image_in, pos, pitchyaw, length=40.0, thickness=1, color=(0, 0, 255)):
	"""Draw gaze angle on given image with a given eye positions."""
	image_out = image_in
	if len(image_out.shape) == 2 or image_out.shape[2] == 1:
		image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
	dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
	dy = -length * np.sin(pitchyaw[0])
	cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
				   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
				   thickness, cv2.LINE_AA, tipLength=0.2)
	return image_out

def vector_to_pitchyaw(vectors):
	n = vectors.shape[0]
	out = np.empty((n, 2))
	vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
	out[:, 0] = np.arcsin(vectors[:, 1])  # theta
	out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
	return out

# normalization function for the face images
# gc (gaze label) removed
def normalizeData_face(img, face_model, landmarks, hr, ht, cam, focal_norm = 960, distance_norm = 600, size = 224):
	## normalized camera parameters
	# focal_norm: focal length of normalized camera
	# distance_norm: normalized distance between eye and camera
	roiSize = (size, size)  # size of cropped eye image

	## compute estimated 3D positions of the landmarks
	ht = ht.reshape((3, 1))
	hR = cv2.Rodrigues(hr)[0]  # rotation matrix
	Fc = np.dot(hR, face_model.T) + ht
	two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
	mouth_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
	face_center = np.mean(np.concatenate((two_eye_center, mouth_center), axis=1), axis=1).reshape((3, 1))

	## ---------- normalize image ----------
	distance = np.linalg.norm(face_center)	# actual distance between eye and original camera

	z_scale = distance_norm / distance
	cam_norm = np.array([
		[focal_norm, 0, roiSize[0] / 2],
		[0, focal_norm, roiSize[1] / 2],
		[0, 0, 1.0],
	])
	S = np.array([	# scaling matrix
		[1.0, 0.0, 0.0],
		[0.0, 1.0, 0.0],
		[0.0, 0.0, z_scale],
	])

	hRx = hR[:, 0]
	forward = (face_center / distance).reshape(3)
	down = np.cross(forward, hRx)
	down /= np.linalg.norm(down)
	right = np.cross(down, forward)
	right /= np.linalg.norm(right)
	R = np.c_[right, down, forward].T  # rotation matrix R

	W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))	# transformation matrix

	img_warped = cv2.warpPerspective(img, W, roiSize)  # image normalization

	## ---------- normalize rotation ----------
	hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
	hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

	# warp the facial landmarks
	num_point, num_axis = landmarks.shape
	det_point = landmarks.reshape([num_point, 1, num_axis])
	det_point_warped = cv2.perspectiveTransform(det_point, W)
	det_point_warped = det_point_warped.reshape(num_point, num_axis)

	return img_warped, hr_norm, det_point_warped, R

# normalization function for the eye images
# gc (gaze label) removed
def normalizeData_eyes(img, face_model, hr, ht, cam, focal_norm = 1800, distance_norm = 343, size = 224):
	## normalized camera parameters
	# focal_norm: focal length of normalized camera
	# distance_norm: normalized distance between eye and camera
	roiSize = (size, size)  # size of cropped eye image

	## compute estimated 3D positions of the landmarks
	ht = ht.reshape((3, 1))
	hR = cv2.Rodrigues(hr)[0]  # rotation matrix
	Fc = np.dot(hR, face_model.T) + ht
	re = 0.5 * (Fc[:, 0] + Fc[:, 1]).reshape((3, 1))  # center of left eye
	le = 0.5 * (Fc[:, 2] + Fc[:, 3]).reshape((3, 1))  # center of right eye

	## normalize each eye
	data = []
	for et in [re, le]:
		## ---------- normalize image ----------
		distance = np.linalg.norm(et)  # actual distance between eye and original camera

		z_scale = distance_norm / distance
		cam_norm = np.array([
			[focal_norm, 0, roiSize[0] / 2],
			[0, focal_norm, roiSize[1] / 2],
			[0, 0, 1.0],
		])
		S = np.array([	# scaling matrix
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, z_scale],
		])

		hRx = hR[:, 0]
		forward = (et / distance).reshape(3)
		down = np.cross(forward, hRx)
		down /= np.linalg.norm(down)
		right = np.cross(down, forward)
		right /= np.linalg.norm(right)
		R = np.c_[right, down, forward].T  # rotation matrix R

		W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))	# transformation matrix

		img_warped = cv2.warpPerspective(img, W, roiSize)  # image normalization

		## ---------- normalize rotation ----------
		hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
		hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

		data.append([img_warped, hr_norm, R])

	return data

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Data Normalization")
	parser.add_argument("-sb", "--subject_begin", type=int, help="which subject to process begining")
	parser.add_argument("-se", "--subject_end", type=int, help="which subject to process at the end")
	args = parser.parse_args()

	## change working dir
	os.chdir(data_dir)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	subject_begin = 0
	subject_end = 1
	if args.subject_begin is not None:
		subject_begin = args.subject_begin
		if args.subject_end is not None:
			subject_end = args.subject_end
		else:
			subject_end = subject_begin + 1
			print('Warning, no subject ender set, will just process the current subject ID: ', subject_begin)
	else:
		if args.subject_end is not None:
			subject_end = args.subject_end
			print('Warning, no subject beginnger set, will start from the first subject ID 0')

	################## Parameters #################################################
	resize_factor = 8
	is_distor = False  # distortion is disable since it cost too much time, and the face is always in the center of image
	report_interval = 60
	is_over_write = True
	patch_size = 224
	###########################################################################

	# load camera matrix
	camera_matrix = []
	camera_distortion = []
	cam_translation = []
	cam_rotation = []

	print('Load the camera parameters')
	for cam_id in range(0, 18):
		file_name = './calibration/cam_calibration/' + 'cam' + str(cam_id).zfill(2) + '.xml'
		fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
		camera_matrix.append(fs.getNode('Camera_Matrix').mat())
		camera_distortion.append(fs.getNode('Distortion_Coefficients').mat()) # here we disable distortion
		cam_translation.append(fs.getNode('cam_translation').mat())
		cam_rotation.append(fs.getNode('cam_rotation').mat())
		fs.release()

	# load face model
	face_model_load = np.loadtxt('./calibration/face_model.txt')
	landmark_use = [20, 23, 26, 29, 15, 19]
	face_model = face_model_load[landmark_use, :]

	for sub_id in range(subject_begin, subject_end):
		start_time = time.time()
		start_time_batch = time.time()

		subject_folder = './data/test/subject' + str(sub_id).zfill(4)
		if not os.path.isdir(subject_folder):  # we keep going
			print('The folder ', subject_folder, ' does not exist')
			continue
		print('Processing ', subject_folder)

		# output file
		hdf_fpath = os.path.join(output_dir, 'subject' + str(sub_id).zfill(4) + '.h5')
		if is_over_write:
			if os.path.exists(hdf_fpath):
				print('Overwrite the file ', subject_folder)
				os.remove(hdf_fpath)
		else:
			if os.path.exists(hdf_fpath):
				print('Skip the file ', subject_folder, ' since it is already exist')
				continue

		output_h5_id = h5py.File(hdf_fpath, 'w')
		print('output file save to ', hdf_fpath)

		output_frame_index = []
		output_cam_index = []

		output_landmarks = []
		output_head_rvec = []
		output_head_tvec = []
		output_face_patch = []
		output_face_gaze = []
		output_face_head_pose = []
		output_face_mat_norm = []

		# load landmarks
		label_path = os.path.join('./data/annotation_test', 'subject' + str(sub_id).zfill(4) + '.csv')
		if not os.path.exists(label_path):
			print('annotation file {} does not exit'.format(label_path))
			exit()

		total_data = 0
		with open(label_path) as f:
			total_data = sum(1 for line in f)
			print('There are in total ', total_data, ' samples')

		save_index = 0
		frame_index = 0
		with open(label_path) as anno_file:
			content = csv.reader(anno_file, delimiter=',')
			cam_id = 0

			for line in content:
				frame_folder = line[0]
				image_name = line[1]
				frame_index = int(frame_folder[5:])
				cam_id = int(image_name[3:5])
				# print('frame_index: ', frame_index)
				# print('cam_id: ', cam_id)

				if frame_index % report_interval==0 and cam_id==0:
					print('process the ', os.path.join(subject_folder, frame_folder))
					precet = frame_index / total_data * 100
					batch_time = time.time() - start_time_batch
					left_time = batch_time * ((total_data - frame_index) / report_interval)
					print('Processed {:01} %, ETA {:02d}:{:02d}:{:02d}'.format(precet, int(left_time // 3600),
																		int(left_time % 3600 // 60),
																		int(left_time % 60)))
					start_time_batch = time.time()

				image_file_name = os.path.join(subject_folder, frame_folder, image_name)
				image = cv2.imread(image_file_name)
				if cam_id in [3, 6, 13]:  # rotate images since some camera is rotated during recording
					(h, w) = image.shape[:2]
					center = (w / 2, h / 2)
					M = cv2.getRotationMatrix2D(center, 180, 1.0)
					image = cv2.warpAffine(image, M, (w, h))

				if is_distor:
					image = cv2.undistort(image, camera_matrix[cam_id], camera_distortion[cam_id])

				landmarks = []

				for num_i in range(0, 68):
					pos_x = float(line[8 + num_i * 2])
					pos_y = float(line[8 + num_i * 2 + 1])
					landmarks.append([pos_x, pos_y])

				landmarks = np.asarray(landmarks)
				landmarks = landmarks.reshape(-1, 2)

				# load annotation
				# gaze_label_3d = np.array([float(line[4]), float(line[5]), float(line[6])]).reshape(3, 1)  # gaze point on the screen coordinate system
				hr = np.array([float(line[2]), float(line[3]), float(line[4])]).reshape(3, 1)
				ht = np.array([float(line[5]), float(line[6]), float(line[7])]).reshape(3, 1)


				# face image
				face_img_normalized, head_norm, landmark_norm, mat_norm_face = \
					normalizeData_face(image, face_model, landmarks, hr, ht, camera_matrix[cam_id], focal_norm = 960, distance_norm = 600, size = patch_size)


				# eye images
				data_normal_eyes = normalizeData_eyes(image, face_model, hr, ht, camera_matrix[cam_id], focal_norm = 1800, distance_norm = 343, size = patch_size)
				[[left_eye_img, hr_norm_left, R_left], [right_eye_img, hr_norm_right, R_right]] = data_normal_eyes

				data_tight_eyes = normalizeData_eyes(image, face_model, hr, ht, camera_matrix[cam_id], focal_norm = 2400, distance_norm = 350, size = patch_size)
				[[left_eye_img_tight, _, _], [right_eye_img_tight, _, _]] = data_tight_eyes

				data_loose_eyes = normalizeData_eyes(image, face_model, hr, ht, camera_matrix[cam_id], focal_norm = 1500, distance_norm = 350, size = patch_size)
				[[left_eye_img_loose, _, _], [right_eye_img_loose, _, _]] = data_loose_eyes



				# create the hdf5 file
				if not output_frame_index:
					output_frame_index = output_h5_id.create_dataset("frame_index", shape=(total_data, 1),
																   dtype=np.int, chunks=(1, 1))
					output_cam_index = output_h5_id.create_dataset("cam_index", shape=(total_data, 1),
																	 dtype=np.int, chunks=(1, 1))
					output_landmarks = output_h5_id.create_dataset("facial_landmarks", shape=(total_data, 68, 2),
																   dtype=np.float, chunks=(1, 68, 2))

					output_face_patch = output_h5_id.create_dataset("face_patch", shape=(total_data, patch_size, patch_size, 3),
																	compression='lzf', dtype=np.uint8,
																	chunks=(1, patch_size, patch_size, 3))

					# create datasets for eye images
					output_left_eye_patch = output_h5_id.create_dataset("left_eye_patch", shape=(total_data, patch_size, patch_size, 3),
																	compression='lzf', dtype=np.uint8,
																	chunks=(1, patch_size, patch_size, 3))
					output_right_eye_patch = output_h5_id.create_dataset("right_eye_patch", shape=(total_data, patch_size, patch_size, 3),
																	compression='lzf', dtype=np.uint8,
																	chunks=(1, patch_size, patch_size, 3))	
					output_left_eye_patch_tight = output_h5_id.create_dataset("left_eye_patch_tight", shape=(total_data, patch_size, patch_size, 3),
																	compression='lzf', dtype=np.uint8,
																	chunks=(1, patch_size, patch_size, 3))
					output_right_eye_patch_tight = output_h5_id.create_dataset("right_eye_patch_tight", shape=(total_data, patch_size, patch_size, 3),
																	compression='lzf', dtype=np.uint8,
																	chunks=(1, patch_size, patch_size, 3))
					output_left_eye_patch_loose = output_h5_id.create_dataset("left_eye_patch_loose", shape=(total_data, patch_size, patch_size, 3),
																	compression='lzf', dtype=np.uint8,
																	chunks=(1, patch_size, patch_size, 3))
					output_right_eye_patch_loose = output_h5_id.create_dataset("right_eye_patch_loose", shape=(total_data, patch_size, patch_size, 3),
																	compression='lzf', dtype=np.uint8,
																	chunks=(1, patch_size, patch_size, 3))	
					
					output_face_mat_norm = output_h5_id.create_dataset("face_mat_norm", shape=(total_data, 3, 3),
																   dtype=np.float, chunks=(1, 3, 3))
					output_face_head_pose = output_h5_id.create_dataset("face_head_pose", shape=(total_data, 2),
																		dtype=np.float, chunks=(1, 2))
					

				output_frame_index[save_index] = frame_index
				output_cam_index[save_index] = cam_id

				output_landmarks[save_index] = landmark_norm
				# output_head_rvec[save_index] = hr.reshape(3)
				# output_head_tvec[save_index] = ht.reshape(3)
				output_face_patch[save_index] = face_img_normalized

				# save eye images
				output_left_eye_patch[save_index] = left_eye_img
				output_right_eye_patch[save_index] = right_eye_img
				output_left_eye_patch_tight[save_index] = left_eye_img_tight
				output_right_eye_patch_tight[save_index] = right_eye_img_tight
				output_left_eye_patch_loose[save_index] = left_eye_img_loose
				output_right_eye_patch_loose[save_index] = right_eye_img_loose

				output_face_mat_norm[save_index] = mat_norm_face

				head = head_norm.reshape(1, 3)
				M = cv2.Rodrigues(head)[0]
				Zv = M[:, 2]
				head_2d = np.array([math.asin(Zv[1]), math.atan2(Zv[0], Zv[2])])
				output_face_head_pose[save_index] = head_2d.reshape(2)

				save_index = save_index + 1

		output_h5_id.close()
		print('close the h5 file')

		print('finish the subject: ', sub_id)
		elapsed_time = time.time() - start_time
		print('///////////////////////////////////')
		print('Running time is {:02d}:{:02d}:{:02d}'.format(int(elapsed_time // 3600), int(elapsed_time % 3600 // 60),
															int(elapsed_time % 60)))