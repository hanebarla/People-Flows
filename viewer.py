from utils import fix_model_state_dict

import h5py
import json
import PIL.Image as Image
import numpy as np
import os

from image import *
from model import CANNet2s
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from matplotlib import cm

from torchvision import transforms

from gradcam.utils import visualize_cam
from gradcam import GradCAMpp


device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

# json file contains the test images
test_json_path = './test.json'

# the folder to output density map and flow maps
output_folder = './plot'

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)



model = CANNet2s()
model = model.cuda()
checkpoint = torch.load('model_best.pth.tar')
model.load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
model.eval()

target_layer = model.frontend
gradcam_pp = GradCAMpp(model, target_layer)

pred= []
gt = []
images = []

for i in range(1):
    img_path = img_paths[i]

    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    index = int(img_name.split('.')[0])

    prev_index = int(max(1,index-5))
    prev_img_path = os.path.join(img_folder,'%03d.jpg'%(prev_index))

    prev_img = Image.open(prev_img_path).convert('RGB')
    img = Image.open(img_path).convert('RGB')

    prev_img = prev_img.resize((640,360))
    img = img.resize((640,360))

    torch_prev_img = transforms.ToTensor()(prev_img).to(device)
    torch_img = transforms.ToTensor()(img).to(device)

    normed_torch_prev_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_prev_img)[None]
    normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]

    gt_path = img_path.replace('.jpg','_resize.h5')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

    images.extend([torch_img.cpu(), heatmap_pp, result_pp])

grid_image = make_grid(images, nrow=3)
out = transforms.ToPILImage()(grid_image)
out.save("GradCAM/gradcam.png")