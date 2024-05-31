import cv2
from basicsr.utils import img2tensor
from ldm.data.utils import AddMiDaS
import torch
import csv
from einops import rearrange
import copy

class dataset_dex():
    def __init__(self, path_json, reg_path):
        super(dataset_dex, self).__init__()
        self.files = []
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = csv.DictReader(fp)
            for file in data:
                if file["image"] == "image": continue
                if 'oakink' in file['image']:
                    image_path = file["image"]
                    # filter out two-hand images
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

        self.reg_files = []
        if reg_path:
            with open(reg_path, 'r', encoding='utf-8') as fp:
                reg_data = csv.DictReader(fp)
                for file in reg_data:
                    if file["image"] == "image": continue
                    image_path = file["image"]
                    sentence = file["sentence"]
                    self.reg_files.append({'rgb': image_path, "sentence": sentence})
        
        self.midas_trafo = AddMiDaS(model_type="dpt_hybrid")
        self.reg_data_len = len(self.reg_files)

        # Back-up data
        self.backup_hoi = self.__getitemHOI__(0)
        self.backup_reg = self.__getitemReg__(0)

    def pt2np(self, x):
        x = ((x + 1.0) * .5).detach().cpu().numpy()
        return x
    
    def __getitemHOI__(self, idx):
        file = self.files[idx]
        top_point = int(float(file["shape"]["top"]))
        bottom_point = int(float(file["shape"]["bottom"]))
        left_point = int(float(file["shape"]["left"]))
        right_point = int(float(file["shape"]["right"]))
        im = cv2.imread(file["rgb"])
        file["shape"]["height"] = im.shape[0]
        file["shape"]["width"] = im.shape[1]

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

        # Read hand+object mask
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

        # Prepare segmentation image (1 channel)
        seg = cv2.imread(file["seg"], cv2.IMREAD_GRAYSCALE)
        seg = seg / 255
        seg = seg[top_point:bottom_point, left_point:right_point]
        seg = cv2.copyMakeBorder(seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        seg = cv2.resize(seg, (512, 512))
        if len(seg.shape)!=3:
            seg = seg[:, :, None]
        seg = img2tensor(seg, bgr2rgb=False, float32=True)
        sentence = file['sentence']

        return {"data":{'im': im, 'depth': depth, 'skeleton':skeleton, 'sentence': sentence, "seg": seg, "mask": mask, 'shape':file["shape"]}}
    
    def __getitemReg__(self, idx):
        if self.reg_data_len == 0: return {}
        reg_file = self.reg_files[idx%self.reg_data_len]
        '''Deal with regularization image'''
        reg_im = cv2.imread(reg_file["rgb"])
        reg_im = img2tensor(reg_im, bgr2rgb=True, float32=True) / 255.
        reg_sentence = reg_file["sentence"]
        return {"reg_data":{'im':reg_im, 'sentence':reg_sentence}}


    def __getitem__(self, idx):
        try:
            data = self.__getitemHOI__(idx)
        except:
            data = self.backup_hoi

        if self.reg_data_len != 0:
            try:
                data_reg = self.__getitemReg__(idx)
            except:
                data_reg = self.backup_reg
            data.update(data_reg)
        return data

    def __len__(self):
        return len(self.files)
