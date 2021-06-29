import os
from model import CANNet2s
from utils import save_checkpoint, fix_model_state_dict

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.nn.functional as F

import numpy as np
import argparse
import json
import cv2
import dataset
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error

parser = argparse.ArgumentParser(description='PyTorch CANNet2s')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')
parser.add_argument('--dataset', default="FDST")
parser.add_argument('--load_model', default="checkpoint.pth.tar")

dloss_on = True


def dataset_factory(dlist, arguments, mode="train"):
    if arguments.dataset == "FDST":
        if mode == "train":
            return dataset.listDataset(dlist, shuffle=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]),
                                       ]),
                                       train=True,
                                       batch_size=args.batch_size,
                                       num_workers=args.workers)
        else:
            return dataset.listDataset(dlist,
                                       shuffle=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]),
                                       ]), train=False)
    elif arguments.dataset == "CrowdFlow":
        return dataset.CrowdDatasets(dlist,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                                     ],),
                                     test_on=True
                                     )
    elif arguments.dataset == "venice":
        return dataset.VeniceDataset(dlist,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                                     ])
                                     )
    else:
        raise ValueError


if __name__ == "__main__":
    global args, best_prec1

    best_prec1 = 200

    args = parser.parse_args()
    args.lr = 1e-4
    args.batch_size    = 1
    args.momentum      = 0.95
    # args.decay         = 5*1e-4
    args.decay         = 1e-3
    args.start_epoch   = 0
    args.epochs = 200
    args.workers = 8
    args.seed = int(time.time())
    # args.print_freq = 400
    args.print_freq = 10
    args.pretrained = True
    if args.dataset  == "FDST":
        with open(args.train_json, 'r') as outfile:
            train_list = json.load(outfile)
        with open(args.val_json, 'r') as outfile:
            val_list = json.load(outfile)
    elif args.dataset == "CrowdFlow":
        train_list = args.train_json
        val_list = args.val_json
    elif args.dataset == "venice":
        train_list = args.train_json
        val_list = args.val_json
    elif args.dataset == "other":
        train_list = args.train_json
        val_list = args.val_json
    else:
        raise ValueError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(args.seed)

    # model load
    model = CANNet2s()
    if args.pretrained:
        checkpoint = torch.load(str(args.load_model))
        model.load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
        try:
            best_prec1 = checkpoint['val']
        except KeyError:
            print("No Key: val")

    # multi gpu
    if torch.cuda.device_count() > 1:
        print("You can use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    val_dataset = dataset_factory(val_list, args, mode="val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1
    )

    gt_num = []
    gt_max = []
    pred_num = []
    pred_max = []

    for i,(prev_img, img, post_img, target) in enumerate(val_loader):
        prev_img = prev_img.to(device, dtype=torch.float)
        prev_img = Variable(prev_img)

        img = img.to(device, dtype=torch.float)
        img = Variable(img)

        with torch.no_grad():
            prev_flow = model(prev_img, img)
            prev_flow_inverse = model(img, prev_img)

        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0,:] = 1.0
        mask_boundry[-1,:] = 1.0
        mask_boundry[:,0] = 1.0
        mask_boundry[:,-1] = 1.0

        mask_boundry = Variable(mask_boundry.cuda())


        reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry

        overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).type(torch.FloatTensor)


        gt_num.append(torch.sum(target).item())
        gt_max.append(torch.max(target).item())
        pred_num.append(torch.sum(overall).item())
        pred_max.append(torch.max(target).item())

    gt_ave = np.mean(np.array(gt_num))
    pred_ave = np.mean(np.array(pred_num))
    gt_max_ave = np.mean(np.array(gt_max))
    pred_max_ave = np.mean(np.array(pred_max))
    print("GT Num average: {}".format(gt_ave * (1 / gt_max_ave)))
    print("Pred Num average: {}".format(pred_ave * (1 / pred_max_ave)))
