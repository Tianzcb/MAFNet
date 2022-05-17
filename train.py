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

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import MAFNet

from utils import utils
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log
from torch.optim.lr_scheduler import LambdaLR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--data-dir', default='data1/images/train', metavar='DIR',
                    help='path to dataset-D')
parser.add_argument('--val-dir', default='data1/images/val', metavar='DIR',
                    help='path to dataset-D')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument("--batch_size", type=int, default=4, help='batch size (default: 16)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-10, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=2, type=int,
                    metavar='N', help='print batch frequency (default: 50)')  #
parser.add_argument('--save-epoch-freq', '-s', default=50, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')  #
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')  #
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', default='./model_data1/', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')
parser.add_argument("--classes", type=int, default=["CYST", "FNH", "HCC", "HEM"],
                    help="num classes (default: None)")
parser.add_argument('--weight_with_optimizer', default='./model_data1/best_train.pth')
parser.add_argument('--gpu_id', default='0')
args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")


def train():
    transforms = [
        Transforms.RondomFlip(),
        Transforms.RandomRotate(15),
        Transforms.Log(0.5),
        Transforms.Blur(0.2),
        Transforms.ToTensor()
    ]

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 上面俩行表示是否使用gpu进行训练
    print("Device: %s" % device)

    train_data = dataset.FFL(args.data_dir, transforms, phase_train=True)
    val_data = dataset.FFL(args.val_dir, transforms, phase_train=True)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)

    if args.last_ckpt:
        model = MAFNet.Net(num_classes=len(args.classes), pretrained=False).to(device)
    else:
        model = MAFNet.Net(num_classes=len(args.classes), pretrained=False).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    loss_func = nn.CrossEntropyLoss().to(device)
    model.train()
    model.to(device)
    print(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    lr_decay_lambda = lambda epoch: args.lr_decay_rate ** (epoch // args.lr_epoch_per_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)
    best_loss, best_loss_eval = float('inf'), float('inf')
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())# 时间戳，保存文件的名字
    writer = SummaryWriter('runs/' + timestamp)# 开启可视化loss下降网页窗口
    checkpoint = ''


    for epoch in range(int(args.start_epoch), args.epochs):
        train_loss = 0
        correct_all = torch.zeros(1).squeeze().cuda()
        total_all = torch.zeros(1).squeeze().cuda()
        correct = list(0. for i in range(len(args.classes)))
        total = list(0. for i in range(len(args.classes)))
        scheduler.step(epoch)

        for batch_idx, sample in enumerate(train_loader):
            NC = sample['NC'].to(device)
            ART = sample['ART'].to(device)
            PV = sample['PV'].to(device)
            target_scales = sample['label'].to(device)
            optimizer.zero_grad()

            pred_scales = model(NC, ART, PV, checkpoint)
            loss = loss_func(pred_scales, target_scales)
            train_loss = train_loss + loss.item()
            if loss < best_loss:
                best_loss = loss
                save_ckpt(args.ckpt_dir, model, optimizer, epoch, '_train')
            loss.backward()
            optimizer.step()

            prediction = torch.argmax(pred_scales, 1)  # 选择第一维度中最大的类别
            correct_all += (prediction == target_scales).sum().float()  # 得到的类别和标签的类别比对，并统计相同的数值
            total_all += len(target_scales)  # 一共的类别数也可以理解为一个batch_size

            res = prediction == target_scales  # 获得一组含有true和false的tensor，如果第一张图标签和预测一样为true否则为false以此类推
            for label_idx in range(len(target_scales)):
                label_single = target_scales[label_idx]
                correct[label_single] += res[label_idx].item()  # 统计单个类别中 预测正确的个数（如果为ture是1，false为0）
                total[label_single] += 1  # 统计单类别个数 比如在一个batch_size中CYST的个数

        print('epoch: %d | train loss: %.4f ' % (epoch, train_loss))

        if (epoch + 1) % 2 == 0:
            model.eval()
            correct_all = torch.zeros(1).squeeze().cuda()
            total_all = torch.zeros(1).squeeze().cuda()
            correct1 = list(0. for i in range(len(args.classes)))
            total1 = list(0. for i in range(len(args.classes)))
            eval_loss = 0
            for batch_idx_val, sample_val in enumerate(val_loader):
                NC = sample_val['NC'].to(device)
                ART = sample_val['ART'].to(device)
                PV = sample_val['PV'].to(device)
                target_val = sample_val['label'].to(device)
                output_val = model(NC, ART, PV)

                valid_loss = loss_func(output_val, target_val)
                eval_loss += valid_loss.item()
                if eval_loss < best_loss_eval:
                    best_loss = eval_loss
                    save_ckpt(args.ckpt_dir, model, optimizer, epoch, '_val')

                prediction = torch.argmax(output_val, 1)  # 选择第一维度中最大的类别
                correct_all += (prediction == target_val).sum().float()  # 得到的类别和标签的类别比对，并统计相同的数值
                total_all += len(target_val)  # 一共的类别数也可以理解为一个batch_size

                res = prediction == target_val  # 获得一组含有true和false的tensor，如果第一张图标签和预测一样为true否则为false以此类推
                for label_idx in range(len(target_val)):
                    label_single = target_val[label_idx]
                    correct1[label_single] += res[label_idx].item()  # 统计单个类别中 预测正确的个数（如果为ture是1，false为0）
                    total1[label_single] += 1  # 统计单类别个数 比如在一个batch_size中CYST的个数

            f = open("ACM_fusion2.txt", "a+")
            acc_str = 'Accuracy: %f' % (sum(correct1) / sum(total1))
            for acc_idx in range(len(args.classes)):
                try:
                    acc = correct1[acc_idx] / total1[acc_idx]
                except:
                    acc = 0
                finally:
                    acc_str += '\tclassID:%d\tacc:%f\t' % (acc_idx + 1, acc)
            print("*" * 160)
            print(acc_str)
            print("*" * 160)
            print(acc_str, file=f)
            f.close()
            if (epoch + 1) % 3 == 0:
                writer.add_scalar("train_loss", train_loss, epoch)
                writer.add_scalar("eval_loss", eval_loss, epoch)
                writer.add_scalar('val_Accuracy', (sum(correct1) / sum(total1)), epoch)  # 上面三行是把值传入日志中
                writer.add_scalar('train_Accuracy', (sum(correct) / sum(total)), epoch)  # 上面三行是把值传入日志中

            save_ckpt(args.ckpt_dir, model, optimizer, epoch, '_last')
    writer.close()
    print("Training completed ")


if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    train()
