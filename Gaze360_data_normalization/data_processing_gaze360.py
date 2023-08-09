import numpy as np
import scipy.io as sio
import cv2 
import os
import sys
import math
# sys.path.append("../core/")
# import data_processing_core as dpc
import face_alignment
from skimage import io
import h5py
import argparse

root = path_to_your/dataset/Gaze360/
out_root = path_to_your/dataset/Gaze360/normalized_h5_224/
current_decide = 'cpu' # for face alignment

def draw_gaze(image_in, pitchyaw, length=40.0, thickness=1, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    h = image_in.shape[0]
    w = image_in.shape[1]
    pos = (int(h / 2.0), int(w / 2.0))

    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out

def get_eye_mouth_landmarks(img_path, use_device='cpu'):
    fa = None
    if use_device == 'gpu':
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    else:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

    input = io.imread(img_path)
    preds = fa.get_landmarks(input)

    if preds is None:
        return None # return None if no face is in image
    preds = preds[-1]

    img = cv2.imread(img_path)

    eye_mouth_preds = []
    eye_mouth_preds.append(preds[36])
    eye_mouth_preds.append(preds[39])
    eye_mouth_preds.append(preds[42])
    eye_mouth_preds.append(preds[45])
    eye_mouth_preds.append(preds[48])
    eye_mouth_preds.append(preds[54])
    # eye_mouth_preds.append(preds[31]) # nose
    # eye_mouth_preds.append(preds[35])

    return np.array(eye_mouth_preds)

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec



# we don't take full facial landmarks as input nor output wrapped landmarks
def normalizeData_face(img, face_model, hr, ht, cam, gaze_3d):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

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
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    gaze_3d = gaze_3d.reshape((3, 1))
    gc_normalized = np.dot(R, gaze_3d)
    gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

    return img_warped, gc_normalized, hr_norm

# normalization function for the eye images
def normalizeData(img, face_model, hr, ht, cam):
# def normalizeData(img, face_model, hr, ht, gc, cam):
    ## normalized camera parameters
    focal_norm = 1800  # focal length of normalized camera
    distance_norm = 300  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    # gc = gc.reshape((3, 1))
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
        S = np.array([  # scaling matrix
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

        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

        img_warped = cv2.warpPerspective(img, W, roiSize)  # image normalization

        ## ---------- normalize rotation ----------
        hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
        hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

        # ## ---------- normalize gaze vector ----------
        # gc_normalized = gc - et  # gaze vector
        # gc_normalized = np.dot(R, gc_normalized)
        # gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

        # data.append([img_warped, hr_norm, gc_normalized, R])
        data.append([img_warped])

    return data

# normalization function for the eye images
def normalizeData_tight(img, face_model, hr, ht, cam):
# def normalizeData(img, face_model, hr, ht, gc, cam):
    ## normalized camera parameters
    focal_norm = 2400  # focal length of normalized camera
    distance_norm = 300  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    # gc = gc.reshape((3, 1))
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
        S = np.array([  # scaling matrix
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

        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

        img_warped = cv2.warpPerspective(img, W, roiSize)  # image normalization

        ## ---------- normalize rotation ----------
        hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
        hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

        # ## ---------- normalize gaze vector ----------
        # gc_normalized = gc - et  # gaze vector
        # gc_normalized = np.dot(R, gc_normalized)
        # gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

        # data.append([img_warped, hr_norm, gc_normalized, R])
        data.append([img_warped])

    return data

# normalization function for the eye images
def normalizeData_loose(img, face_model, hr, ht, cam):
# def normalizeData(img, face_model, hr, ht, gc, cam):
    ## normalized camera parameters
    focal_norm = 1200  # focal length of normalized camera
    distance_norm = 300  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    # gc = gc.reshape((3, 1))
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
        S = np.array([  # scaling matrix
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

        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

        img_warped = cv2.warpPerspective(img, W, roiSize)  # image normalization

        ## ---------- normalize rotation ----------
        hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
        hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

        # ## ---------- normalize gaze vector ----------
        # gc_normalized = gc - et  # gaze vector
        # gc_normalized = np.dot(R, gc_normalized)
        # gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

        # data.append([img_warped, hr_norm, gc_normalized, R])
        data.append([img_warped])

    return data


def ImageProcessing_Gaze360(subject_index):

    image_size=224
    # get distortion coeff from ETH-XGaze Cam00
    cam_file_name = 'cam00.xml'
    fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    camera_distortion = fs.getNode('Distortion_Coefficients').mat()

    # get Gaze360 metadata
    msg = sio.loadmat(os.path.join(root, "metadata.mat"))
    
    recordings = msg["recordings"]
    gazes = msg["gaze_dir"]
    head_bbox = msg["person_head_bbox"]
    face_bbox = msg["person_face_bbox"]
    lefteye_bbox = msg["person_eye_left_bbox"]
    righteye_bbox = msg["person_eye_right_bbox"]
    splits = msg["splits"]

    split_index = msg["split"]
    recording_index = msg["recording"]
    
    #build a dictionary to store the starting and ending indices for each folder
    indices = {}
    for i in np.unique(recording_index):
        indices[i] = [np.min(np.where(recording_index[0,:]==i)[0]), np.max(np.where(recording_index[0,:]==i)[0])+1]
    
    person_index = msg["person_identity"]
    frame_index = msg["frame"]
  
    total_num = recording_index.shape[1]
    
    length = {}
    for i in np.unique(recording_index):
        length[i] = [len(np.where(split_index[0, indices[i][0]:indices[i][1]]==0)[0]),
                     len(np.where(split_index[0, indices[i][0]:indices[i][1]]==1)[0]),
                     len(np.where(split_index[0, indices[i][0]:indices[i][1]]==2)[0]),
                     len(np.where(split_index[0, indices[i][0]:indices[i][1]]==3)[0]),]
    

    # # Build folders for saving image and label.
        
    # for i in range(4):
    #     if not os.path.exists(os.path.join(out_root, splits[0, i][0])):
    #         os.makedirs(os.path.join(out_root, splits[0, i][0], "Left"))
    #         os.makedirs(os.path.join(out_root, splits[0, i][0], "Right"))
    #         os.makedirs(os.path.join(out_root, splits[0, i][0], "Face"))

    # load face model
    face_model_load = np.loadtxt('face_model.txt')
    landmark_use = [20, 23, 26, 29, 15, 19]
    face_model = face_model_load[landmark_use, :]
    facePts = face_model.reshape(6, 1, 3)

    # make default camera matrix
    cam_matrix = np.zeros([3, 3], dtype = float)
    cam_matrix[0][0] = 1.32007013e+04
    cam_matrix[1][1] = 1.31924964e+04
    cam_matrix[2][2] = 1.0

    # store hr, ht
    hr_arr = np.zeros((total_num, 3))
    ht_arr = np.zeros((total_num, 3))
    
    hdf_path_train = os.path.join(out_root, 'train', 'rec_'+str(subject_index).zfill(3)+'.h5')
    hdf_path_val = os.path.join(out_root, 'val', 'rec_'+str(subject_index).zfill(3)+'.h5')
    hdf_path_test = os.path.join(out_root, 'test', 'rec_'+str(subject_index).zfill(3)+'.h5')
    hdf_path_unused = os.path.join(out_root, 'unused', 'rec_'+str(subject_index).zfill(3)+'.h5')
    
    
    if length[subject_index][0] > 0:
        output_h5_train = h5py.File(hdf_path_train, 'w')
        output_face_index_train = output_h5_train.create_dataset("face", shape=(length[subject_index][0], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_index_train = output_h5_train.create_dataset("left", shape=(length[subject_index][0], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_index_train = output_h5_train.create_dataset("right", shape=(length[subject_index][0], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_loose_index_train = output_h5_train.create_dataset("left_loose", shape=(length[subject_index][0], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_loose_index_train = output_h5_train.create_dataset("right_loose", shape=(length[subject_index][0], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_tight_index_train = output_h5_train.create_dataset("left_tight", shape=(length[subject_index][0], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_tight_index_train = output_h5_train.create_dataset("right_tight", shape=(length[subject_index][0], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_gaze_3d_index_train = output_h5_train.create_dataset("gaze3d", shape=(length[subject_index][0],3), 
                                                                 dtype=np.float, chunks=(1,3))
        output_gaze_2d_index_train = output_h5_train.create_dataset("gaze", shape=(length[subject_index][0],2),
                                                                 dtype=np.float, chunks=(1,2))
        output_head_pose_index_train = output_h5_train.create_dataset("head_pose", shape=(length[subject_index][0],2),
                                                                 dtype=np.float, chunks=(1,2))
    if length[subject_index][1] > 0:
        output_h5_val = h5py.File(hdf_path_val, 'w')
        output_face_index_val = output_h5_val.create_dataset("face", shape=(length[subject_index][1], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_index_val = output_h5_val.create_dataset("left", shape=(length[subject_index][1], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_index_val = output_h5_val.create_dataset("right", shape=(length[subject_index][1], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_loose_index_val = output_h5_val.create_dataset("left_loose", shape=(length[subject_index][1], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_loose_index_val = output_h5_val.create_dataset("right_loose", shape=(length[subject_index][1], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_tight_index_val = output_h5_val.create_dataset("left_tight", shape=(length[subject_index][1], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_tight_index_val = output_h5_val.create_dataset("right_tight", shape=(length[subject_index][1], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_gaze_3d_index_val = output_h5_val.create_dataset("gaze3d", shape=(length[subject_index][1],3), 
                                                                 dtype=np.float, chunks=(1,3))
        output_gaze_2d_index_val = output_h5_val.create_dataset("gaze", shape=(length[subject_index][1],2),
                                                                 dtype=np.float, chunks=(1,2))
        output_head_pose_index_val = output_h5_val.create_dataset("head_pose", shape=(length[subject_index][1],2),
                                                                 dtype=np.float, chunks=(1,2))
    if length[subject_index][2] > 0:
        output_h5_test = h5py.File(hdf_path_test, 'w')
        output_face_index_test = output_h5_test.create_dataset("face", shape=(length[subject_index][2], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_index_test = output_h5_test.create_dataset("left", shape=(length[subject_index][2], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_index_test = output_h5_test.create_dataset("right", shape=(length[subject_index][2], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_loose_index_test = output_h5_test.create_dataset("left_loose", shape=(length[subject_index][2], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_loose_index_test = output_h5_test.create_dataset("right_loose", shape=(length[subject_index][2], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_tight_index_test = output_h5_test.create_dataset("left_tight", shape=(length[subject_index][2], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_tight_index_test = output_h5_test.create_dataset("right_tight", shape=(length[subject_index][2], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_gaze_3d_index_test = output_h5_test.create_dataset("gaze3d", shape=(length[subject_index][2],3), 
                                                                 dtype=np.float, chunks=(1,3))
        output_gaze_2d_index_test = output_h5_test.create_dataset("gaze", shape=(length[subject_index][2],2),
                                                                 dtype=np.float, chunks=(1,2))
        output_head_pose_index_test = output_h5_test.create_dataset("head_pose", shape=(length[subject_index][2],2),
                                                                 dtype=np.float, chunks=(1,2))
    if length[subject_index][3] > 0:
        output_h5_unused = h5py.File(hdf_path_unused, 'w')
        output_face_index_unused = output_h5_unused.create_dataset("face", shape=(length[subject_index][3], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_index_unused = output_h5_unused.create_dataset("left", shape=(length[subject_index][3], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_index_unused = output_h5_unused.create_dataset("right", shape=(length[subject_index][3], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_loose_index_unused = output_h5_unused.create_dataset("left_loose", shape=(length[subject_index][3], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_loose_index_unused = output_h5_unused.create_dataset("right_loose", shape=(length[subject_index][3], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_left_tight_index_unused = output_h5_unused.create_dataset("left_tight", shape=(length[subject_index][3], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_right_tight_index_unused = output_h5_unused.create_dataset("right_tight", shape=(length[subject_index][3], image_size, image_size, 3), 
                                                                 dtype=np.float, chunks=(1,image_size,image_size,3))
        output_gaze_3d_index_unused = output_h5_unused.create_dataset("gaze3d", shape=(length[subject_index][3],3), 
                                                                 dtype=np.float, chunks=(1,3))
        output_gaze_2d_index_unused = output_h5_unused.create_dataset("gaze", shape=(length[subject_index][3],2),
                                                                 dtype=np.float, chunks=(1,2))
        output_head_pose_index_unused = output_h5_unused.create_dataset("head_pose", shape=(length[subject_index][3],2),
                                                                 dtype=np.float, chunks=(1,2))


    save_index_train = 0
    save_index_val = 0
    save_index_test = 0
    save_index_unused = 0
    
    # process each image
    # for i in range(total_num):
    for i in range(indices[subject_index][0], indices[subject_index][1]):
        im_path = os.path.join(root, "imgs",
            recordings[0, recording_index[0, i]][0],
            "head", '%06d' % person_index[0, i],
            '%06d.jpg' % frame_index[0, i]
            )

       	progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(i/total_num * 20))
        progressbar = "\r" + progressbar + f" {i}|{total_num}"
        print(progressbar, end = "", flush=True)

        # skip no face images
        if (face_bbox[i] == np.array([-1, -1, -1, -1])).all():
            continue

        category = splits[0, split_index[0, i]][0]
        gaze = gazes[i]

        img = cv2.imread(im_path)

        # change fx fy of camera matrix
        cam_matrix[0][2] = img.shape[0] * 1.0 / 2
        cam_matrix[1][2] = img.shape[1] * 1.0 / 2
    
        # get eye and mouth landmarks
        eye_mouth_landmarks = get_eye_mouth_landmarks(im_path, use_device=current_decide)

        if eye_mouth_landmarks is None:
            continue # no face is detected, skip this image
            
        eye_mouth_landmarks = eye_mouth_landmarks.astype(np.float32)
        eye_mouth_landmarks = eye_mouth_landmarks.reshape(6, 1, 2)

        # get head rotation and translation
        hr, ht = estimateHeadPose(eye_mouth_landmarks, facePts, cam_matrix, camera_distortion)
        hr_arr[i:i+1,:] = hr.reshape((1, 3))
        ht_arr[i:i+1,:] = ht.reshape((1, 3))

        # now normalize
        img_normalized_face, gaze_3d_normalized, head_pose_normalized = normalizeData_face(img, face_model, hr, ht, cam_matrix, gaze)

        [[img_normalized_left], [img_normalized_right]] = normalizeData(img, face_model, hr, ht, cam_matrix)

        [[img_normalized_left_tight], [img_normalized_right_tight]] = normalizeData_tight(img, face_model, hr, ht, cam_matrix)

        [[img_normalized_left_loose], [img_normalized_right_loose]] = normalizeData_loose(img, face_model, hr, ht, cam_matrix)

        gaze_3d_normalized = gaze_3d_normalized.reshape((3, ))
        gaze_2d_normalized = GazeTo2d(gaze_3d_normalized)

        head_pose = head_pose_normalized.reshape(1, 3)
        M = cv2.Rodrigues(head_pose)[0]
        Zv = M[:, 2]
        head_normalized_2d = np.array([math.asin(Zv[1]), math.atan2(Zv[0], Zv[2])])

        # cv2.imwrite(os.path.join(out_root, category, "Face", f"{i+1}.jpg"), img_normalized_face)
        # cv2.imwrite(os.path.join(out_root, category, "Left", f"{i+1}.jpg"), img_normalized_left)
        # cv2.imwrite(os.path.join(out_root, category, "Right", f"{i+1}.jpg"), img_normalized_right)
        # cv2.imwrite(os.path.join(out_root, category, "Left_tight", f"{i+1}.jpg"), img_normalized_left_tight)
        # cv2.imwrite(os.path.join(out_root, category, "Right_tight", f"{i+1}.jpg"), img_normalized_right_tight)
        # cv2.imwrite(os.path.join(out_root, category, "Left_loose", f"{i+1}.jpg"), img_normalized_left_loose)
        # cv2.imwrite(os.path.join(out_root, category, "Right_loose", f"{i+1}.jpg"), img_normalized_right_loose)

        
        if split_index[0, i] == 0:
            output_face_index_train[save_index_train] = img_normalized_face
            output_left_index_train[save_index_train] = img_normalized_left
            output_right_index_train[save_index_train] = img_normalized_right
            output_left_loose_index_train[save_index_train] = img_normalized_left_loose
            output_right_loose_index_train[save_index_train] = img_normalized_right_loose
            output_left_tight_index_train[save_index_train] = img_normalized_left_tight
            output_right_tight_index_train[save_index_train] = img_normalized_right_tight
            output_gaze_3d_index_train[save_index_train] = gaze_3d_normalized
            output_gaze_2d_index_train[save_index_train] = gaze_2d_normalized
            output_head_pose_index_train[save_index_train] = head_normalized_2d
            save_index_train += 1
        elif split_index[0, i] == 1:
            output_face_index_val[save_index_val] = img_normalized_face
            output_left_index_val[save_index_val] = img_normalized_left
            output_right_index_val[save_index_val] = img_normalized_right
            output_left_loose_index_val[save_index_val] = img_normalized_left_loose
            output_right_loose_index_val[save_index_val] = img_normalized_right_loose
            output_left_tight_index_val[save_index_val] = img_normalized_left_tight
            output_right_tight_index_val[save_index_val] = img_normalized_right_tight
            output_gaze_3d_index_val[save_index_val] = gaze_3d_normalized
            output_gaze_2d_index_val[save_index_val] = gaze_2d_normalized
            output_head_pose_index_val[save_index_val] = head_normalized_2d
            save_index_val += 1
        elif split_index[0, i] == 2:
            output_face_index_test[save_index_test] = img_normalized_face
            output_left_index_test[save_index_test] = img_normalized_left
            output_right_index_test[save_index_test] = img_normalized_right
            output_left_loose_index_test[save_index_test] = img_normalized_left_loose
            output_right_loose_index_test[save_index_test] = img_normalized_right_loose
            output_left_tight_index_test[save_index_test] = img_normalized_left_tight
            output_right_tight_index_test[save_index_test] = img_normalized_right_tight
            output_gaze_3d_index_test[save_index_test] = gaze_3d_normalized
            output_gaze_2d_index_test[save_index_test] = gaze_2d_normalized
            output_head_pose_index_test[save_index_test] = head_normalized_2d
            save_index_test += 1
        else:
            output_face_index_unused[save_index_unused] = img_normalized_face
            output_left_index_unused[save_index_unused] = img_normalized_left
            output_right_index_unused[save_index_unused] = img_normalized_right
            output_left_loose_index_unused[save_index_unused] = img_normalized_left_loose
            output_right_loose_index_unused[save_index_unused] = img_normalized_right_loose
            output_left_tight_index_unused[save_index_unused] = img_normalized_left_tight
            output_right_tight_index_unused[save_index_unused] = img_normalized_right_tight
            output_gaze_3d_index_unused[save_index_unused] = gaze_3d_normalized
            output_gaze_2d_index_unused[save_index_unused] = gaze_2d_normalized
            output_head_pose_index_unused[save_index_unused] = head_normalized_2d
            save_index_unused += 1

        #break
       
    # exit() 
    if length[subject_index][0] > 0:
        output_h5_train.close()
    if length[subject_index][1] > 0:
        output_h5_val.close()
    if length[subject_index][2] > 0:
        output_h5_test.close()
    if length[subject_index][3] > 0:
        output_h5_unused.close()
        

    # save hr ht
    np.savetxt('hr_arrary.txt', hr_arr, delimiter=',')
    np.savetxt('ht_arrary.txt', ht_arr, delimiter=',')

    

# pitch yaw
def GazeTo2d(gaze):
  yaw = np.arctan2(gaze[0], -gaze[2])
  pitch = np.arcsin(gaze[1])
  return np.array([pitch, yaw])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_index', type=int)
    args = parser.parse_args()
    print('--------------')
    print(args.subject_index)
    ImageProcessing_Gaze360(args.subject_index)
