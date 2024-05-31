import json
import cv2
import os
from basicsr.utils import img2tensor
import numpy as np
from ldm.data.utils import AddMiDaS
import torch
from PIL import Image
import csv
from einops import repeat, rearrange
import copy
from ldm.modules.midas.api import MiDaSInference


class dataset_dex():
    def __init__(self, path_json, root_path_im='', root_path_depth='', root_path_skeleton='', image_size=(640,480), train = True):
        super(dataset_dex, self).__init__()
        self.files = []
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = csv.DictReader(fp)
            for file in data:
                if file["image"] == "image": continue
                if train:
                    if 'dexycb' in file['image']:
                        image_path = file["image"].replace("/data/mez005/data/", "/data/mez005/data2/")
                        skeleton_path = file["skeleton"].replace("/data/mez005//data/", "/data/mez005/data2/")
                        mask_path = file["mask"].replace("/data/mez005/data/", "/data/mez005/data2/")
                        seg_path = file["seg"].replace("/data/mez005/data/", "/data/mez005/data2/")
                    elif 'oakink' in file['image']:
                        image_path = file["image"]
                        labels = image_path.split("/")[-3].split("_")
                        if len(labels) == 4: continue
                        if labels[1] == "0004": continue
                        image_path = file["image"]
                        skeleton_path = file["skeleton"]
                        mask_path = file["mask"]
                        seg_path = file["seg"]
                    else:
                        image_path = file["image"]
                        skeleton_path = file["skeleton"]
                        mask_path = file["mask"]
                        seg_path = file["seg"]
                    self.files.append({'rgb': image_path, "skeleton":skeleton_path, "mask": mask_path, "sentence":file["sentence"], "seg":seg_path, "shape":{"top":file["top"],"bottom":file["bottom"],"left":file["left"],"right":file["right"]}})
                else:
                    self.files.append({'rgb': file["image"], "skeleton":file["skeleton"], "label":file["label"], "mask": file["mask"], "sentence":file["sentence"], "seg": file["seg"], "shape":{"top":file["top"],"bottom":file["bottom"],"left":file["left"],"right":file["right"]}})

        
        if isinstance(image_size, int):
            self.height = self.width = image_size
        elif isinstance(image_size, tuple):
            self.width = image_size[0]
            self.height = image_size[1]
        else:
            self.width = 640
            self.height = 480
        self.root_path_im = root_path_im
        self.root_path_depth = root_path_depth
        self.root_path_seleton = root_path_skeleton
        self.midas_trafo = AddMiDaS(model_type="dpt_hybrid")
        # self.midas = MiDaSInference(model_type="dpt_hybrid")
        
        
        
    def pt2np(self, x):
        x = ((x + 1.0) * .5).detach().cpu().numpy()
        return x

    def __getitem__(self, idx):
        file = self.files[idx]
        try:
            top_point = int(float(file["shape"]["top"]))
            bottom_point = int(float(file["shape"]["bottom"]))
            left_point = int(float(file["shape"]["left"]))
            right_point = int(float(file["shape"]["right"]))
        except:
            file = self.files[5]
            top_point = int(float(file["shape"]["top"]))
            bottom_point = int(float(file["shape"]["bottom"]))
            left_point = int(float(file["shape"]["left"]))
            right_point = int(float(file["shape"]["right"]))

        # # Avoid throwing information, Not usefull here
        # if "label" not in file: 
        #     number = file["rgb"][-10:-4]
        #     label_file = file["rgb"][:-16]+"labels_"+f'{number}.npz'
        #     label = np.load(label_file)
        #     joint3d = label['joint_3d'].reshape(-1, 21*3)
        #     joint3d = np.append(joint3d,label['pose_m'])
        # else:
        #     label = np.load(file['label'])
        #     joint3d = label['joint3d'].reshape(-1, 21*3)
        #     pose_hand = label["handparam"].reshape(-1, 51)
        #     joint3d/=1000
        #     joint3d = np.append(joint3d,pose_hand)


        im = cv2.imread(file["rgb"])
        file["shape"]["height"] = im.shape[0]
        file["shape"]["width"] = im.shape[1]

        # tmp = copy.deepcopy(im)
        # tmp = img2tensor(tmp, bgr2rgb=True, float32=True) / 255.

        im = im[top_point:bottom_point, left_point:right_point]
        height, width = im.shape[:2]
        align_size = max([height, width])
        top = bottom = (align_size - height) // 2
        right = left = (align_size - width) // 2
        
        # Read RGB images (3 channels)
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        im = cv2.resize(im, (512, 512))
        tmp = copy.deepcopy(im)
        tmp = img2tensor(tmp, bgr2rgb=True, float32=True) / 255.
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        # Read hand+object
        mask = cv2.imread(file["mask"])
        mask = mask[top_point:bottom_point, left_point:right_point]
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        mask = cv2.resize(mask, (512, 512))
        mask = img2tensor(mask, bgr2rgb=True, float32=True) / 255.

        # Read skeleton image (3 channels)
        skeleton = cv2.imread(file["skeleton"])
        skeleton = skeleton[top_point:bottom_point, left_point:right_point]
        skeleton = cv2.copyMakeBorder(skeleton, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        skeleton = cv2.resize(skeleton, (512, 512))
        skeleton = img2tensor(skeleton, bgr2rgb=True, float32=True) / 255.

        # Prepare depth input
        depth = 2 * tmp - 1
        depth = rearrange(depth, 'c h w -> h w c')
        depth = self.pt2np(depth)
        depth = self.midas_trafo.transform({"image": depth})["image"]
        depth = torch.from_numpy(depth)

        # Prepare segmentation image (1 channels)
        seg = cv2.imread(file["seg"], cv2.IMREAD_GRAYSCALE)
        seg = seg / 255
        seg = seg[top_point:bottom_point, left_point:right_point]
        seg = cv2.copyMakeBorder(seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        seg = cv2.resize(seg, (512, 512))
        if len(seg.shape)!=3:
            seg = seg[:, :, None]
        seg = img2tensor(seg, bgr2rgb=False, float32=True)
        
        sentence = file['sentence']
        return {'im': im, 'depth': depth, 'skeleton':skeleton, 'sentence': sentence, "seg": seg, "mask": mask, 'shape':file["shape"]}

    def __len__(self):
        return len(self.files)
