# -*- coding: utf-8 -*-
"""
######################################################################################################################################
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Any publications arising from the use of this software, including but
not limited to academic journal and conference publications, technical
reports and manuals, must cite at least one of the following works:

Revisiting Data Normalization for Appearance-Based Gaze Estimation
Xucong Zhang, Yusuke Sugano, Andreas Bulling
in Proc. International Symposium on Eye Tracking Research and Applications (ETRA), 2018
######################################################################################################################################
"""

import os
import cv2
import numpy as np
import h5py
import scipy.io as sio
import math

# pitch yaw
def GazeTo2d(gaze):
    gaze = gaze.reshape((3,1))
    yaw = np.arctan2(gaze[0], -gaze[2])
    pitch = np.arcsin(gaze[1])
    return np.array([pitch, yaw])

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def normalizeData_face(img, face, hr, ht, cam):
    focal_norm = 960
    distance_norm = 600
    roiSize = (224, 224)
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]
    Fc = np.dot(hR, face) + ht
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3,1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))
    distance = np.linalg.norm(face_center)
    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
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
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)

    return img_warped

def normalizeData(img, face, hr, ht, gc, cam):
    ## normalized camera parameters
    focal_norm = 960 # focal length of normalized camera
    distance_norm = 200 # normalized distance between eye and camera, 600
    roiSize = (224, 224) # size of cropped eye image, 60, 36

    # img_u = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_u = img

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3,1))
    gc = gc.reshape((3,1))
    hR = cv2.Rodrigues(hr)[0] # rotation matrix
    Fc = np.dot(hR, face) + ht # 3D positions of facial landmarks
    re = 0.5*(Fc[:,0] + Fc[:,1]).reshape((3,1)) # center of left eye
    le = 0.5*(Fc[:,2] + Fc[:,3]).reshape((3,1)) # center of right eye
    
    ## normalize each eye
    data = []
    for et in [re, le]:
        ## ---------- normalize image ----------
        distance = np.linalg.norm(et) # actual distance between eye and original camera
        
        z_scale = distance_norm/distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0]/2],
            [0, focal_norm, roiSize[1]/2],
            [0, 0, 1.0],
        ])
        S = np.array([ # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])
        
        hRx = hR[:,0]
        forward = (et/distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T # rotation matrix R
        
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))) # transformation matrix
        
        img_warped = cv2.warpPerspective(img_u, W, roiSize) # image normalization
        # img_warped = cv2.equalizeHist(img_warped)
        
        ## ---------- normalize rotation ----------
        hR_norm = np.dot(R, hR) # rotation matrix in normalized space
        hr_norm = cv2.Rodrigues(hR_norm)[0] # convert rotation matrix to rotation vectors
        
        ## ---------- normalize gaze vector ----------
        gc_normalized = gc - et # gaze vector
        # For modified data normalization, scaling is not applied to gaze direction, so here is only R applied.
        # For original data normalization, here should be:
        # "M = np.dot(S,R)
        # gc_normalized = np.dot(R, gc_normalized)"
        gc_normalized = np.dot(R, gc_normalized)
        gc_normalized = gc_normalized/np.linalg.norm(gc_normalized)
        
        data.append([img_warped, hr_norm, gc_normalized])
        
    return data

if __name__ == '__main__':

    # load the generic face model, which includes 6 facial landmarks: four eye corners and two mouth corners
    face = sio.loadmat('faceModelGeneric.mat')['model']
    num_pts = face.shape[1]
    facePts = face.T.reshape(num_pts, 1, 3)

    subjects = [f'p{index:02}' for index in range(15)] # p00 - p14

    # subjectpath = './' + subjects[0]
    # subject = subjects[0]
    ## load calibration data, these paramters can be obtained by camera calibration functions in OpenCV

    hdf_path = './' + 'MPIIFaceGaze_multiregion.h5'
    output_h5 = h5py.File(hdf_path, 'w')
    for subject in subjects:
        group_subject = output_h5.create_group(subject)
        subjectpath = './' + subject
        cameraCalib = sio.loadmat(subjectpath + '/Calibration/Camera.mat')
        camera_matrix = cameraCalib['cameraMatrix']
        camera_distortion = cameraCalib['distCoeffs']

        annotations_file = open(subjectpath+'/'+subject+'.txt')
        annotations = annotations_file.readlines()
        annotations_file.close()

        annotations_dict = {}
        for line in annotations:
            values = line.strip().split(' ')
            annotations_dict[values[0]] = values
        # print(subject)
        with open('/home/xiangwei/PycharmProjects/technical_report/MPIIFaceGaze/Evaluation Subset/sample list for eye image' + '/' + subject + '.txt') as txtfile:
            # total_num = len(txtfile.readlines())
            # print(total_num)
            output_h5_face = group_subject.create_group('image')
            output_h5_left = group_subject.create_group('left')
            output_h5_right = group_subject.create_group('right')
            output_h5_gaze = group_subject.create_group('gaze')
            output_h5_pose = group_subject.create_group('pose')
            line_num = 0
            # print(line_num)
            # load sample
            for line in txtfile:
                # print(line)
                sample_metadata = line.strip().split(' ')
                sample_path = os.path.join(subjectpath + '/' + sample_metadata[0])
                left_or_right = sample_metadata[-1]
                img_original = cv2.imread(sample_path)
                img = cv2.undistort(img_original, camera_matrix, camera_distortion)

                # load the detected facial landmarks
                landmarks = np.array([[int(annotations_dict[sample_metadata[0]][3]), int(annotations_dict[sample_metadata[0]][4])], [int(annotations_dict[sample_metadata[0]][5]), int(annotations_dict[sample_metadata[0]][6])],
                                        [int(annotations_dict[sample_metadata[0]][7]), int(annotations_dict[sample_metadata[0]][8])], [int(annotations_dict[sample_metadata[0]][9]), int(annotations_dict[sample_metadata[0]][10])],
                                        [int(annotations_dict[sample_metadata[0]][11]), int(annotations_dict[sample_metadata[0]][12])], [int(annotations_dict[sample_metadata[0]][13]), int(annotations_dict[sample_metadata[0]][14])]])

                # print("landmarks shape: ", landmarks.shape)
                # print("num_pts" , num_pts)
                # estimate head pose

                landmarks = landmarks.astype(np.float32)
                landmarks = landmarks.reshape(num_pts, 1, 2)
                hr, ht = estimateHeadPose(landmarks, facePts, camera_matrix, camera_distortion)

                # load 3D gaze target position in camera coordinate system
                gc = np.array([float(annotations_dict[sample_metadata[0]][24]), float(annotations_dict[sample_metadata[0]][25]), float(annotations_dict[sample_metadata[0]][26])])  # 3D gaze taraget position
                # print("gc: ", gc2d )
                # data normalization for left and right eye image
                img_face = normalizeData_face(img, face, hr, ht, camera_matrix)
                dataleft, dataright = normalizeData(img, face, hr, ht, gc, camera_matrix)
                img_left = dataleft[0]
                img_right = dataright[0]

                assert dataleft[2].all() == dataright[2].all()
                gaze2d = GazeTo2d(dataleft[2]).reshape(2)
                # print('original:', gaze2d)
                pose = dataleft[1].reshape((3,1))
                # print('gaze2d:', gaze2d)
                # print('-----------')
                # print(str(line_num).zfill(4))
                if left_or_right == 'left':
                    output_h5_face[str(line_num).zfill(4)] = img_face
                    output_h5_left[str(line_num).zfill(4)] = img_left
                    output_h5_right[str(line_num).zfill(4)] = img_right
                    output_h5_gaze[str(line_num).zfill(4)] = gaze2d
                    output_h5_pose[str(line_num).zfill(4)] = pose
                elif left_or_right == 'right':
                    img_face = cv2.flip(img_face, 1)
                    img_left = cv2.flip(img_left, 1)
                    img_right = cv2.flip(img_right, 1)
                    gaze2d[1] *= -1
                    output_h5_face[str(line_num).zfill(4)] = img_face
                    output_h5_left[str(line_num).zfill(4)] = img_left
                    output_h5_right[str(line_num).zfill(4)] = img_right
                    output_h5_gaze[str(line_num).zfill(4)] = gaze2d
                    output_h5_pose[str(line_num).zfill(4)] = pose
                # output_h5_face_dataset = output_h5_face.create_dataset(str(line_num).zfill(4), data=img_face)
                # output_h5_left_dataset = output_h5_left.create_dataset(str(line_num).zfill(4), data=img_left)
                # output_h5_right_dataset = output_h5_right.create_dataset(str(line_num).zfill(4), data=img_right)
                # output_h5_gaze_dataset = output_h5_gaze.create_dataset(str(line_num).zfill(4), data=gaze2d)
                # output_h5_pose_dataset = output_h5_pose.create_dataset(str(line_num).zfill(4), data=pose)

                # output_h5_face_dataset[:] = img_face
                # output_h5_gaze_dataset[:] = gaze2d
                # output_h5_pose_dataset[:] = pose
                # output_h5_left_dataset[:] = img_left
                # output_h5_right_dataset[:] = img_right

                line_num += 1
    #
                # if line_num==20:
                #     output_h5.close()
                #     exit()
            print(str(subject)+' is done!')
            # output_h5.close()
            # exit()
    output_h5.close()