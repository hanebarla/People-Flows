from captum.attr import DeepLift
from captum.attr._utils import attribution

from utils import fix_model_state_dict, FlowGradCAMpp

import h5py
import json
import PIL.Image as Image
import numpy as np
import os

from image import *
import model
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from torchvision import transforms
import torchvision
from matplotlib import cm


normal_path = "checkpoint.pth.tar"
direct_path = "fdst.pth.tar"


# json file contains the test images
test_json_path = './test.json'
# the floder to output density map and flow maps
output_floder = ',/plot'

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CANnet = model.CANNet2s()
CANnet.to(device)
CANnet.load_state_dict(fix_model_state_dict(torch.load(normal_path)['state_dict']))
CANnet.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

dl = DeepLift(CANnet.to('cpu'), CANnet.frontend)

for i in range(0, 1):
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
    torch_prev_img = torchvision.transforms.ToTensor()(prev_img)
    torch_img = torchvision.transforms.ToTensor()(img)

    prev_img = transform(prev_img).cuda()
    img = transform(img).cuda()
    gt_path = img_path.replace('.jpg','_resize.h5')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    prev_img = prev_img.cuda()
    prev_img = Variable(prev_img)
    img = img.cuda()
    img = Variable(img)
    img = img.unsqueeze(0)
    prev_img = prev_img.unsqueeze(0)

    attrib = dl.attribute((prev_img, img), 0)
    print(attrib.size())
