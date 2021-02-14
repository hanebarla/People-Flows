import h5py
import torch
import shutil
from collections import OrderedDict
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.functional import norm
import torchvision
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


ChannelToLocation = ['aboveleft', 'above', 'aboveright',
                     'left', 'center', 'right',
                     'belowleft', 'below', 'belowright']


class CompareOutput():
    def __init__(self, img_dict_keys):
        self.losses_dict = {}  # input, label, **loss outpt

        img_kind = 0
        for k in img_dict_keys:
            self.losses_dict[k] = []
            img_kind += 1

        self.img_width = img_kind
        self.img_height = 0
        self.figure = None
        self.axes = []
        self.cm = cm.hsv
        self.norm = Normalize()

    def append_pred(self, loss_dict):
        for k in loss_dict:
            if loss_dict[k] is not None:
                self.losses_dict[k].append(loss_dict[k])

    def _set_img(self):
        self.img_height = len(self.losses_dict['input'])
        self.figure = plt.figure(figsize=(self.img_height * 4, self.img_width * 5))
        for h in range(self.img_height):
            temp_ax = []
            for w in range(self.img_width):
                temp_ax.append(self.figure.add_subplot(self.img_height,
                                                       self.img_width,
                                                       h*self.img_width + w + 1))
            self.axes.append(temp_ax)

    def plot_img(self):
        self._set_img()

        for h in range(self.img_height):
            for w, k in enumerate(self.losses_dict):

                content = self.losses_dict[k][h]
                if h == 0:
                    self.axes[h][w].set_title(k, fontsize=18)

                if content[0] == 'img':
                    self.axes[h][w].imshow(content[1])
                else:
                    colors = -np.arctan2(content[1][2], content[1][3])
                    self.norm.autoscale(colors)
                    self.axes[h][w].quiver(content[1][0],
                                           content[1][1],
                                           content[1][2],
                                           content[1][3],
                                           color=self.cm(self.norm(colors)),
                                           angles='xy', scale_units='xy', scale=1)
                    self.axes[h][w].set_ylim(0, 45)
                    self.axes[h][w].set_xlim(0, 80)
                    self.axes[h][w].set_aspect('equal')

    def save_fig(self, name='images/demo.png'):
        self.figure.savefig(name, dpi=300)


def tm_output_to_dense(output):
    output_sum = np.zeros_like(output[:, :, 0])
    for i in range(9):
        output_sum += output[:, :, i]

    temp_max = np.max(output_sum)
    output_sum /= temp_max

    return output_sum


def output_to_img(output):
    output_num = output

    o_max = np.max(output_num)
    heats_u = np.zeros_like(output_num[0, :, :])
    heats_v = np.zeros_like(output_num[0, :, :])

    for i in range(9):
        out = output_num[i, :, :]
        # mean = np.mean(out)
        # std = np.std(out)
        print("{} max: {}".format(ChannelToLocation[i], np.max(out)))
        print("{} min: {}".format(ChannelToLocation[i], np.min(out)))
        heatmap = np.array(255*(out/o_max), dtype=np.uint8)

        if i == 0:
            heats_u -= heatmap/255/np.sqrt(2)
            heats_v += heatmap/255/np.sqrt(2)
        elif i == 1:
            heats_v += heatmap/255
        elif i == 2:
            heats_u += heatmap/255/np.sqrt(2)
            heats_v += heatmap/255/np.sqrt(2)
        elif i == 3:
            heats_u -= heatmap/255
        elif i == 5:
            heats_u += heatmap/255
        elif i == 6:
            heats_u -= heatmap/255/np.sqrt(2)
            heats_v -= heatmap/255/np.sqrt(2)
        elif i == 7:
            heats_v -= heatmap/255
        elif i == 8:
            heats_u += heatmap/255/np.sqrt(2)
            heats_v -= heatmap/255/np.sqrt(2)

    x, y = heats_u.shape[0], heats_u.shape[1]
    imX = np.zeros_like(heats_u)
    for i in range(y):
        imX[:, i] = np.linspace(x, 0, x)
    imY = np.zeros_like(heats_v)
    for i in range(x):
        imY[i, :] = np.linspace(0, y, y)

    return (imY, imX, heats_u, heats_v)


def NormalizeQuiver(output):
    output_num = output

    o_max = np.max(output_num)
    heats_u = np.zeros_like(output_num[0, :, :])
    heats_v = np.zeros_like(output_num[0, :, :])

    for i in range(9):
        out = output_num[i, :, :]
        # mean = np.mean(out)
        # std = np.std(out)
        # print("{} max: {}".format(ChannelToLocation[i], np.max(out)))
        # print("{} min: {}".format(ChannelToLocation[i], np.min(out)))
        heatmap = np.array(255*(out/o_max), dtype=np.uint8)

        if i == 0:
            heats_u -= heatmap/(255 * np.sqrt(2))
            heats_v += heatmap/(255 * np.sqrt(2))
        elif i == 1:
            heats_v += heatmap/255
        elif i == 2:
            heats_u += heatmap/(255 * np.sqrt(2))
            heats_v += heatmap/(255 * np.sqrt(2))
        elif i == 3:
            heats_u -= heatmap/255
        elif i == 5:
            heats_u += heatmap/255
        elif i == 6:
            heats_u -= heatmap/(255 * np.sqrt(2))
            heats_v -= heatmap/(255 * np.sqrt(2))
        elif i == 7:
            heats_v -= heatmap/255
        elif i == 8:
            heats_u += heatmap/(255 * np.sqrt(2))
            heats_v -= heatmap/(255 * np.sqrt(2))

    x, y = heats_u.shape[0], heats_u.shape[1]
    imX = np.zeros_like(heats_u)
    for i in range(y):
        imX[:, i] = np.linspace(x, 0, x)
    imY = np.zeros_like(heats_v)
    for i in range(x):
        imY[i, :] = np.linspace(0, y, y)

    v_leng = np.sqrt(heats_u * heats_u + heats_v * heats_v)
    v_leng_true = v_leng > 0
    imX = imX[v_leng_true]
    imY = imY[v_leng_true]
    heats_u_cut = heats_u[v_leng_true] / v_leng[v_leng_true]
    heats_v_cut = heats_v[v_leng_true] / v_leng[v_leng_true]
    # heats_u_cut = heats_u[v_leng_true]
    # heats_v_cut = heats_v[v_leng_true]
    # cut_lengs = np.sqrt(heats_u_cut * heats_u_cut + heats_v_cut * heats_v_cut)
    # heats_u_cut = heats_u_cut / cut_lengs
    # heats_v_cut = heats_v_cut / cut_lengs

    return (imY, imX, heats_u_cut, heats_v_cut)


def output_res_img(label, output):
    res = (label - output) * (label - output)
    res_max = np.max(res)
    res = res / res_max

    return res


def hsvToflow(hsv):
    flow = np.zeros((hsv.shape[0], hsv.shape[1], 10))

    flow[:, :, 5] = np.where((hsv[:, :, 0] < 12) | (hsv[:, :, 0] >= 169), hsv[:, :, 2], 0)
    flow[:, :, 2] = np.where((hsv[:, :, 0] >= 12) & (hsv[:, :, 0] < 35), hsv[:, :, 2], 0)
    flow[:, :, 1] = np.where((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] < 57), hsv[:, :, 2], 0)
    flow[:, :, 0] = np.where((hsv[:, :, 0] >= 57) & (hsv[:, :, 0] < 80), hsv[:, :, 2], 0)
    flow[:, :, 3] = np.where((hsv[:, :, 0] >= 80) & (hsv[:, :, 0] < 102), hsv[:, :, 2], 0)
    flow[:, :, 6] = np.where((hsv[:, :, 0] >= 102) & (hsv[:, :, 0] < 124), hsv[:, :, 2], 0)
    flow[:, :, 7] = np.where((hsv[:, :, 0] >= 124) & (hsv[:, :, 0] < 147), hsv[:, :, 2], 0)
    flow[:, :, 8] = np.where((hsv[:, :, 0] >= 147) & (hsv[:, :, 0] < 169), hsv[:, :, 2], 0)

    return flow
