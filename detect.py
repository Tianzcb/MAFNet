'''
Our code is partially adapted from RedNet (https://github.com/JinDongJiang/RedNet)
'''
import argparse
import os
import time
import torch
import transforms as Transforms

from torch.utils.data import DataLoader
import torch.optim
from torch import nn
import dataset

from tensorboardX import SummaryWriter

import MAFNet


from utils import utils
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log
from torch.optim.lr_scheduler import LambdaLR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--data-dir', default='data2/images/test', metavar='DIR',
                    help='path to dataset-D')
parser.add_argument('--checkpoint', action='store_true', default='./model_data2/best_train_fu.pth',
                    help='Using Pytorch checkpoint or not')
parser.add_argument("--classes", type=int, default=["CYST", "FNH", "HCC", "HEM"],
                    help="num classes (default: None)")
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def detect():
    transforms = [
        Transforms.ToTensor()
    ]
    test_data = dataset.FFL(args.data_dir, transforms, phase_train=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    model = MAFNet.Net(num_classes=len(args.classes)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    correct_all = torch.zeros(1).squeeze().cuda()
    total_all = torch.zeros(1).squeeze().cuda()
    correct = list(0. for i in range(len(args.classes)))
    total = list(0. for i in range(len(args.classes)))


    for batch_idx, sample in enumerate(test_loader):
        NC = sample['NC'].to(device)
        ART = sample['ART'].to(device)
        PV = sample['PV'].to(device)
        target = sample['label'].to(device)
        pred_scales = model(NC, ART, PV, args.checkpoint)
        prediction = torch.argmax(pred_scales, 1)  # 选择第一维度中最大的类别
        #print(prediction)
        correct_all += (prediction == target).sum().float()  # 得到的类别和标签的类别比对，并统计相同的数值
        total_all += len(target)  # 一共的类别数也可以理解为一个batch_size

        res = prediction == target  # 获得一组含有true和false的tensor，如果第一张图标签和预测一样为true否则为false以此类推
        for label_idx in range(len(target)):
            label_single = target[label_idx]
            correct[label_single] += res[label_idx].item()  # 统计单个类别中 预测正确的个数（如果为ture是1，false为0）
            total[label_single] += 1  # 统计单类别个数 比如在一个batch_size中CYST的个数


    acc_str = 'Accuracy: %f' % (sum(correct) / sum(total))
    for acc_idx in range(len(args.classes)):
        try:
            acc = correct[acc_idx] / total[acc_idx]
        except:
            acc = 0
        finally:
            acc_str += '\tclassID:%d\tacc:%f\t' % (acc_idx + 1, acc)
    print("*" * 160)
    print(acc_str)
    print("*" * 160)


if __name__ == '__main__':
    detect()
