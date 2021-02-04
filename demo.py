import json
import PIL.Image as Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import argparse
from utils import *
import model
from torchvision import transforms
from torch.autograd import Variable

normal_path = "checkpoint.pth.tar"
direct_path = "fdst.pth.tar"


def demo():
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the csv file of the Datasets path.
                                                 In default, path is 'Data/TestData_Path.csv'
                                                 """)

    parser.add_argument('-p', '--path', default='TestData_Path.csv')  # Testdata path csv
    parser.add_argument('-wd', '--width', type=int, default=640)  # image width that input to model
    parser.add_argument('-ht', '--height', type=int, default=360)  # image height thta input to model
    parser.add_argument('-nw', '--normal_weight', default=normal_path)
    parser.add_argument('-dw', '--direct_weight', default=direct_path)
    parser.add_argument('-num', '--img_num', default=10)

    args = parser.parse_args()
    test_d_path = args.path
    normal_weights = args.normal_weight
    direct_weights = args.direct_weight
    num = args.img_num

    # json file contains the test images
    test_json_path = './test.json'
    # the floder to output density map and flow maps
    output_floder = ',/plot'

    with open(test_json_path, 'r') as outfile:
        img_paths = json.load(outfile)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CANnet = model.CANNet2s()
    CANnet.to(device)
    CANnet.load_state_dict(fix_model_state_dict(torch.load(normal_weights)['state_dict']))
    CANnet.eval()

    D_CANnet = model.CANNet2s()
    D_CANnet.to(device)
    D_CANnet.load_state_dict(fix_model_state_dict(torch.load(direct_weights)['state_dict']))
    D_CANnet.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    img_dict_keys = ['input',
                     'normal',
                     'normal_quiver',
                     'direct',
                     'direct_quiver']

    img_dict = {
        img_dict_keys[0]: ('img', None),
        img_dict_keys[1]: ('img', None),
        img_dict_keys[2]: ('quiver', None),
        img_dict_keys[3]: ('img', None),
        img_dict_keys[4]: ('quiver', None)
    }

    DemoImg = CompareOutput(img_dict_keys)

    for i in range(num):
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


        with torch.set_grad_enabled(False):
            output_normal = CANnet(prev_img, img)
            # output_normal = sigma(output_normal) - 0.5

            output_direct = D_CANnet(prev_img, img)
            # output_direct = sigma(output_direct) - 0.5

        input_num = prev_img[0, :, :, :].detach().cpu().numpy()
        input_num = input_num.transpose((1, 2, 0))

        normal_num = output_normal[0, :, :, :].detach().cpu().numpy()
        normal_quiver = NormalizeQuiver(normal_num)
        normal_num = normal_num.transpose((1, 2, 0))

        direct_num = output_direct[0, :, :, :].detach().cpu().numpy()
        direct_quiver = NormalizeQuiver(direct_num)
        direct_num = direct_num.transpose((1, 2, 0))

        normal_dense = tm_output_to_dense(normal_num)
        direct_dense = tm_output_to_dense(direct_num)

        img_dict = {
            img_dict_keys[0]: ('img', input_num),
            img_dict_keys[1]: ('img', normal_dense),
            img_dict_keys[2]: ('quiver', normal_quiver),
            img_dict_keys[3]: ('img', direct_dense),
            img_dict_keys[4]: ('quiver', direct_quiver),
        }

        DemoImg.append_pred(img_dict)

        print("{} / {} done\r".format((i+1), num), end="")

    DemoImg.plot_img()
    DemoImg.save_fig()


if __name__ == "__main__":
    demo()
