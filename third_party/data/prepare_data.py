from manopth.manopth.manolayer import ManoLayer
import pickle
import torch
import argparse
import os
import numpy as np
import math
from PIL import Image
import sys
import csv
from tqdm import tqdm
from utils.color_map import get_color_map

jointsMapManoToSimple = [0,
                         1, 2, 3, 4,
                         5, 6, 7, 8,
                         9, 10, 11, 12,
                         13, 14, 15, 16,
                         17, 18, 19, 20]

mapping = [
    '', 'toy car', 'mug', 'laptop', 'storage furniture', 'bottle',
    'safe', 'bowl', 'bucket', 'scissors', '', 'pliers', 'kettle',
    'knife', 'trash can', '', '', 'lamp', 'stapler', '', 'chair'
]
cmap = np.array(get_color_map()[:7])
# plain = np.ones((1080,1920,3))
# cmap = [plain * np.array(c) for c in cmap]
color_label_json = 'definitions/motion segmentation/label.csv'
color2semantic = dict()
with open(color_label_json, 'r', encoding='utf-8') as fp:
    data = csv.DictReader(fp)
    for file in data:
        if file["Category"] == "Category": continue
        obj_id = file['Category ID']
        labels = ['background', file['Lable 1'], file['Lable 2'], file['Lable 3'], file['Lable 4'], file['Lable 5'], file['Lable 6']]
        color2semantic[obj_id] = labels
hand_color = np.ones((1080, 1920, 3)) * np.array([128, 128, 128])
obj_color = np.ones((1080, 1920, 3)) * np.array([153, 51, 51])

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    if not isinstance(pts3D, np.ndarray):
        pts3D = pts3D.squeeze(0).detach().cpu().numpy()
    else:
        pts3D = pts3D.squeeze(axis=0)
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 2

    return proj_pts

def showHandJoints(imgInOrg, gtIn, filename=None):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param filename: dump image name
    :return:
    '''
    import cv2

    # imgIn = np.zeros_like(imgInOrg)
    imgIn = np.copy(imgInOrg)

    # Set color for each finger
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    PYTHON_VERSION = sys.version_info[0]

    gtIn = np.round(gtIn).astype(np.int)

    if gtIn.shape[0]==1:
        imgIn = cv2.circle(imgIn, center=(gtIn[0][0], gtIn[0][1]), radius=3, color=joint_color_code[0],
                             thickness=-1)
    else:

        for joint_num in range(gtIn.shape[0]):

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)

        for limb_num in range(len(limbs)):

            x1 = gtIn[limbs[limb_num][0], 1]
            y1 = gtIn[limbs[limb_num][0], 0]
            x2 = gtIn[limbs[limb_num][1], 1]
            y2 = gtIn[limbs[limb_num][1], 0]
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if length < 150 and length > 5:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 3),
                                           int(deg),
                                           0, 360, 1)
                color_code_num = limb_num // 4
                if PYTHON_VERSION == 3:
                    limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
                else:
                    limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])

                cv2.fillConvexPoly(imgIn, polygon, color=limb_color)


    if filename is not None:
        cv2.imwrite(filename, imgIn)

    return imgIn


def getHandPose(pkl_path, side):
    f = open(pkl_path, 'rb')
    hand_info = pickle.load(f, encoding='latin1')
    f.close()
    kps2d = hand_info["kps2D"]
    return kps2d

def getSegment(image):
    seg_image = np.asarray(Image.open(image))
    # Below is for segmentation
    crop = (seg_image/255).sum(axis=-1)
    crop_image = np.where(crop==0, 0, 1)
    crop_image = (crop_image*255.).astype(np.uint8)
    seg = np.where(seg_image.sum(axis=-1) != 0)

    if len(seg)!=2 or len(seg[0])==0 or len(seg[1])==0:
        return 0, 1079, 0, 1919, crop_image
    else:
        top = int(max(seg[0].min()-50, 0))
        bottom = int(min(seg[0].max()+50, 1079))
        left = int(max(seg[1].min()-50, 0))
        right = int(min(seg[1].max()+50, 1919))
        return top, bottom, left, right, crop_image
    
def getMask(image, objID):
    mask_image = np.asarray(Image.open(image))
    label_ids = []
    for i in range(len(color2semantic[objID])):
        if 'Right Hand' in color2semantic[objID][i] or 'Left Hand' in color2semantic[objID][i]: label_ids.append(i)
    mask = np.zeros_like(hand_color)
    for id in label_ids:
        hand_mask = np.where((mask_image == cmap[id]).all(-1), 1, 0)[:,:,None]
        mask += hand_mask * hand_color
        mask_image = mask_image * (1-hand_mask)
    obj_mask = np.where(mask_image.sum(axis=-1) != 0, 1, 0)[:,:,None]
    mask += obj_mask * obj_color
    return mask
    
def prepareData(root):
    hand_right_pose_path = os.path.join(root, "handpose/refinehandpose_right")
    hand_left_pose_path = os.path.join(root,"handpose/refinehandpose_left")
    rgb_list = dict()

    with open('./release.txt', 'r') as f:
        for i in f.readlines():
            rgb_path_name = os.path.join(root, 'HOI4D_release', i.strip(),'align_rgb')
            seg2d_anno = os.path.join(root, "HOI4D_annotations")
            seg_path_name = os.path.join(seg2d_anno, i.strip(), "2Dseg")
            if not os.path.exists(seg_path_name):
                continue
            else:
                seg_exact_path_name = os.listdir(seg_path_name)[0]
                seg_path_name = os.path.join(seg_path_name, seg_exact_path_name)
            object_name = mapping[int(i.strip().split("/")[2][1:])]
            camera_index = i.strip().split("/")[0]
            mano_path_right = os.path.join(hand_right_pose_path, i.strip())
            mano_path_left = os.path.join(hand_left_pose_path, i.strip())
            if not os.path.exists(mano_path_right) and not os.path.exists(mano_path_left):
                continue
            elif os.path.exists(mano_path_right) and not os.path.exists(mano_path_left):
                rgb_list[rgb_path_name] = {"mano_right": mano_path_right, "mano_left": None, "camera":camera_index, "obj":object_name, "seg":seg_path_name}
            elif not os.path.exists(mano_path_right) and os.path.exists(mano_path_left):
                rgb_list[rgb_path_name] = {"mano_right": None, "mano_left": mano_path_left, "camera":camera_index, "obj":object_name, "seg":seg_path_name}
            else:
                rgb_list[rgb_path_name] = {"mano_right": mano_path_right, "mano_left": mano_path_left, "camera":camera_index,"obj":object_name, "seg":seg_path_name}

    step = 0
    interval = 5000            
    csv_file = [["image","skeleton","top","bottom","left","right","sentence","seg","mask"]]
    for rgb_path in tqdm(rgb_list):
        camera_intrin = os.path.join(root, f"camera_params/{rgb_list[rgb_path]['camera']}/intrin.npy")
        intrinsics = np.load(camera_intrin)
        rgb_img_list = os.listdir(rgb_path)
        mano_right_path = rgb_list[rgb_path]["mano_right"]
        mano_left_path = rgb_list[rgb_path]["mano_left"]
        seg_path = rgb_list[rgb_path]["seg"]

        for rgb in tqdm(rgb_img_list):
            step+=1
            if "jpg" not in rgb and "png" not in rgb:
                continue
            rgb_image = os.path.join(rgb_path, rgb) # image_path
            image = np.asarray(Image.open(rgb_image))
            if seg_path is not None:
                seg_image = os.path.join(seg_path, rgb).replace("jpg",'png')
                top, bottom, left, right, segmentation = getSegment(seg_image)
                obj_id = rgb_path.split("/")[-6]
                mask = getMask(seg_image, obj_id)
            else:
                continue
            if np.all(segmentation == 0) or np.all(mask == 0): continue

            rgb_index = int(rgb.split(".")[0])
            joint3d = []
            if mano_right_path is not None:
                right_pkl_path = os.path.join(mano_right_path, f"{rgb_index}.pickle")
                if os.path.exists(right_pkl_path):
                    right_kps = getHandPose(right_pkl_path, "right")
                    joint3d.append(right_kps)
            if mano_left_path is not None:
                left_pkl_path = os.path.join(mano_left_path, f"{rgb_index}.pickle")
                if os.path.exists(left_pkl_path):
                    left_kps = getHandPose(left_pkl_path, "left")
                    joint3d.append(left_kps)

            if len(joint3d) == 0:
                continue
            else:
                try:
                    image = np.zeros_like(image)
                    for joints in joint3d:
                        handKPS = joints #project_3D_points(intrinsics, joints, is_OpenGL_coords=False)
                        imgAnno = showHandJoints(image, handKPS)
                        image = imgAnno
                except:
                    continue
                
                # Skeleton saving
                handpose_path = rgb_path.replace("HOI4D_release", "skeleton")
                skeleton_root = root+"/skeleton/"
                handpose_path = skeleton_root + handpose_path[len(skeleton_root):].replace("/", "_") + rgb
                kp = Image.fromarray(imgAnno)
                kp.save(handpose_path)
                # Segmentation saving
                segmentation_path = rgb_path.replace("HOI4D_release", "seg")
                seg_root = root+"/seg/"
                segmentation_path = seg_root + segmentation_path[len(seg_root):].replace("/", "_") + rgb
                seg = Image.fromarray(segmentation)
                seg.save(segmentation_path)
                # mask saving
                mask_path = rgb_path.replace("HOI4D_release", "mask")
                mask_root = root+"/mask/"
                mask_path = mask_root + mask_path[len(mask_root):].replace("/", "_") + rgb
                mask = Image.fromarray(mask.astype(np.uint8))
                mask.save(mask_path)
                # Text prompt generation
                text_prompt = f"A hand is grasping a {rgb_list[rgb_path]['obj']}"

                im_json = [rgb_image, handpose_path, top, bottom, left, right, text_prompt, segmentation_path, mask_path]
                csv_file.append(im_json)

                if step % interval == 0:
                    with open(os.path.join(root, "sample.csv"), "a", newline='') as outfile:
                        writer = csv.writer(outfile)
                        writer.writerows(csv_file)
                        csv_file = []
                        outfile.close()
            

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.root_dir, "skeleton")):
        os.mkdir(os.path.join(args.root_dir, "skeleton"))
    if not os.path.exists(os.path.join(args.root_dir, "mask")):
        os.mkdir(os.path.join(args.root_dir, "mask"))
    if not os.path.exists(os.path.join(args.root_dir, "seg")):
        os.mkdir(os.path.join(args.root_dir, "seg"))
    prepareData(args.root_dir)