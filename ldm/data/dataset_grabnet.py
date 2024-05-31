import cv2
from basicsr.utils import img2tensor
from ldm.data.utils import AddMiDaS
import torch
import csv
from einops import rearrange
import copy


class dataset_grabnet():
    def __init__(self, path_json):
        super(dataset_grabnet, self).__init__()
        self.files = []
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = csv.DictReader(fp)
            for file in data:
                if file["image"] == "image": continue
                self.files.append({'rgb': file["image"], "skeleton":file["skeleton"], "mask": file["mask"], "sentence":file["sentence"], "seg":file["seg"], "shape":{"top":file["top"],"bottom":file["bottom"],"left":file["left"],"right":file["right"]}})
        self.midas_trafo = AddMiDaS(model_type="dpt_hybrid")
        
    def pt2np(self, x):
        x = ((x + 1.0) * .5).detach().cpu().numpy()
        return x

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file["rgb"].split(".")[0].split("/")[-1] # get the name of image
        im = cv2.imread(file["rgb"])
        tmp = copy.deepcopy(im)
        tmp = img2tensor(tmp, bgr2rgb=True, float32=True) / 255.
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        # Read hand+object mask
        mask = cv2.imread(file["mask"])
        mask = img2tensor(mask, bgr2rgb=True, float32=True) / 255.

        # Read skeleton image (3 channels)
        skeleton = cv2.imread(file["skeleton"])
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
        if len(seg.shape)!=3:
            seg = seg[:,:,None]
        seg = img2tensor(seg, bgr2rgb=False, float32=True)
        sentence = file['sentence']
        return {'im': im, 'depth': depth, 'skeleton':skeleton, 'sentence': sentence, "seg": seg, "mask": mask, 'shape':file["shape"], "name":name}

    def __len__(self):
        return len(self.files)
