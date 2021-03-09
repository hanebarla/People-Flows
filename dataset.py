import json
import os
import random
from matplotlib.pyplot import sci
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision
from image import *
import torchvision.transforms.functional as F
from torchvision import transforms
import csv
import cv2
import scipy.io
from scipy.ndimage.filters import gaussian_filter


class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, batch_size=1, num_workers=4):
        random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]

        prev_img, img, post_img, target = load_data(img_path, self.train)

        if self.transform is not None:
            prev_img = self.transform(prev_img)
            img = self.transform(img)
            post_img = self.transform(post_img)
        return prev_img, img, post_img, target


ras2bits = 0.71
IP = {0: 202.5, 1: 247.5, 2: 292.5, 3: 157.5, 5: 337.5, 6: 22.5, 7: 67.5, 8: 112.5}


class CrowdDatasets(torch.utils.data.Dataset):
    def __init__(self, Trainpath="Data/TrainData_Path.csv", transform=None, width=640, height=360, test_on=False):
        super().__init__()
        self.transform = transform
        self.width = width
        self.height = height
        self.out_width = int(width / 8)
        self.out_height = int(height / 8)
        self.test_on = test_on
        with open(Trainpath) as f:
            reader = csv.reader(f)
            self.Pathes = [row for row in reader]

    def __len__(self):
        return len(self.Pathes)

    def __getitem__(self, index):
        """
        CSV Pathlist reference
        -------
            train
                index 0: input image(step t),
                index 1: person label(step t),
                index 2: input label(step t-1),
                index 3: person label(step t-1),
                index 4: label flow(step t-1 2 t),
                index 5: input image(step t+1),
                index 6: preson label(step t+1),
                index 7: label flow(step t 2 t+1)

            test
                index 0: input image(step tm),
                index 1: person label(step tm),
                index 2: input image(step t),
                index 3: person label(step t),
                index 4: label flow(step tm 2 t)
        """
        pathlist = self.Pathes[index]
        t_img_path = pathlist[0]
        t_person_path = pathlist[1]
        t_m_img_path = pathlist[2]
        t_m_person_path = pathlist[3]
        t_m_t_flow_path = pathlist[4]
        t_p_img_path = pathlist[5]
        t_p_person_path = pathlist[6]
        t_t_p_flow_path = pathlist[7]

        t_input, t_person = self.gt_img_density(t_img_path, t_person_path)
        tm_input, tm_person = self.gt_img_density(t_m_img_path, t_m_person_path)
        tp_input, tp_person = self.gt_img_density(t_p_img_path, t_p_person_path)

        tm2t_flow = self.gt_flow(t_m_t_flow_path)
        t2tp_flow = self.gt_flow(t_t_p_flow_path)

        return tm_input, t_input, tp_input, t_person

    def IndexProgress(self, i, gt_flow_edge, h, s):
        oheight = self.out_height
        owidth = self.out_width
        if i == 4:
            grid_i = np.zeros((oheight, owidth, 1))
            return grid_i
        elif i == 9:
            gt_flow_edge_ndarr = np.array(gt_flow_edge)
            gtflow_sum = np.sum(gt_flow_edge_ndarr, axis=0)
            grid_i = gtflow_sum
            return grid_i
        else:
            grid_i = np.where((h >= IP[i] * ras2bits) & (h < ((IP[i] + 45) % 360) * ras2bits), 1, 0)
            grid_i = np.array(grid_i, dtype=np.uint8)
            grid_i = s * grid_i
            grid_i = cv2.resize(grid_i, (owidth, oheight))  # width, height
            grid_i_inner = grid_i[1:(oheight-1), 1:(owidth-1)]
            grid_i_edge = grid_i
            grid_i_inner = np.pad(grid_i_inner, 1)
            grid_i_edge[1:(oheight-1), 1:(owidth-1)] = 0
            grid_i_inner = np.reshape(grid_i_inner, (oheight, owidth, 1))  # height, width, channel
            grid_i_edge = np.reshape(grid_i_edge, (oheight, owidth, 1))
            gt_flow_edge.append(grid_i_edge)

            return grid_i_inner

    def gt_flow(self, path):

        if not os.path.isfile(path):
            return print("No such file: {}".format(path))

        gt_flow_list = []
        gt_flow_edge = []
        img = cv2.imread(path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        h, s, v = cv2.split(img_hsv)
        for i in range(10):
            grid_i = self.IndexProgress(i, gt_flow_edge, h, s)
            gt_flow_list.append(grid_i)

        gt_flow_img_data = np.concatenate(gt_flow_list, axis=2)
        gt_flow_img_data /= np.max(gt_flow_img_data)

        """
        # テスト
        traj = np.sum(gt_flow_img_data, axis=2)
        print(traj.shape)

        root = os.getcwd()
        imgfolder = root + "/images/"
        heatmap = cv2.resize(traj, (traj.shape[1]*8, traj.shape[0]*8))
        heatmap = np.array(heatmap*255, dtype=np.uint8)
        cv2.imwrite(imgfolder+"flow_test.png", heatmap)
        # print(gt_flow_img_data.shape)
        # print(np.max(np.sum(gt_flow_img_data, axis=2)))
        """

        gt_flow_img_data = transforms.ToTensor()(gt_flow_img_data)

        return gt_flow_img_data

    def gt_img_density(self, img_path, mask_path):

        if not os.path.isfile(img_path):
            return print("No such file: {}".format(img_path))
        if not os.path.isfile(mask_path):
            return print("No such file: {}".format(mask_path))

        input_img = cv2.imread(img_path)
        input_img = cv2.resize(input_img, (self.width, self.height))  # width, height
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        mask_img = cv2.imread(mask_path, 0)
        if mask_img is None:
            return print("CRC error: {}".format(mask_path))
        mask_img = np.reshape(mask_img, (mask_img.shape[0], mask_img.shape[1], 1))

        input_img = input_img / 255  # range [0:1]
        mask_img = cv2.resize(mask_img, (self.out_width, self.out_height)) / 255  # width, height

        input_img = self.transform(input_img)
        mask_img = transforms.ToTensor()(mask_img)
        # print("input max: {}".format(torch.max(input_img)))
        # print("intpu min: {}".format(torch.min(input_img)))

        return input_img, mask_img


class VeniceDataset(Dataset):
    def __init__(self, pathjson=None, transform=None, width=640, height=360) -> None:
        super().__init__()
        with open(pathjson, "r") as f:
            self.allpath = json.load(f)
        self.transform = transform
        self.width = width
        self.height = height

    def __len__(self) -> int:
        return len(self.allpath)

    def __getitem__(self, index: int):
        prev_path = self.allpath[index]["prev"]
        now_path = self.allpath[index]["now"]
        next_path = self.allpath[index]["next"]
        target_path = self.allpath[index]["target"]

        prev_img = cv2.imread(prev_path)
        prev_img = cv2.resize(prev_img, (self.width, self.height))
        prev_img = prev_img / 255.0
        prev_img = self.transform(prev_img)

        now_img = cv2.imread(now_path)
        now_img = cv2.resize(now_img, (self.width, self.height))
        now_img = now_img / 255.0
        now_img = self.transform(now_img)

        next_img = cv2.imread(next_path)
        next_img = cv2.resize(next_img, (self.width, self.height))
        next_img = next_img / 255.0
        next_img = self.transform(next_img)

        target_dict = scipy.io.loadmat(target_path)
        target = np.zeros((int(self.height/8), int(self.width/8)))

        for p in range(target_dict['annotation'].shape[0]):
            target[int(target_dict['annotation'][p][1]/16), int(target_dict['annotation'][p][0]/16)] = 1

        target = gaussian_filter(target, 3) * 64
        target = torch.from_numpy(target.astype(np.float32)).clone()

        return prev_img, now_img, next_img, target
