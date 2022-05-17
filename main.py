import argparse
import os
import time
import transforms as Transforms
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from dataset_fusion import DataSet
from utils.utils import save_ckpt
#from torch.utils.tensorboard import SummaryWriter
import ResNet50
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from utils.utils import load_ckpt




def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_root", type=str, default='data_PV/images/train',
                        help="path to Dataset")
    parser.add_argument("--val_root", type=str, default='data_PV/images/val',
                        help="path to Dataset")
    parser.add_argument("--classes", type=int, default=["CYST", "FNH", "HCC", "HEM"],
                        help="num classes (default: None)")
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-10, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                        help='decay rate of learning rate (default: 0.8)')
    parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                        help='epoch of per decay of learning rate (default: 150)')
    parser.add_argument('--epochs', default=500)
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--ckpt-dir', default='./model_PV/', metavar='DIR',
                        help='path to save checkpoints')
    parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--weight_with_optimizer', default='./model_PV/best_train.pth')
    parser.add_argument('--gpu_id', default='0')
    return parser


def main():
    opts = get_argparser().parse_args()# 通过自定义函数get_argparser获取参数

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 上面俩行表示是否使用gpu进行训练
    print("Device: %s" % device)

    transforms = [
        Transforms.RondomFlip(), # 随机翻转
        Transforms.RandomRotate(15),# 随机旋转
        Transforms.Log(0.5), # 对数操作
        Transforms.Blur(0.2),# 均值滤波
        # Transforms.ToGray(),# 转灰度值
        Transforms.ToTensor()#转为tensor，并归一化至[0-1]
    ]
    names = opts.classes  # 获取类别名字列表
    train_dataset = DataSet(opts.train_root, names, transforms)
    val_dataset = DataSet(opts.val_root, names, transforms)
    train_dataLoader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=0)# 获取训练集
    val_dataLoader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=0)  # 获取验证集


    model = ResNet50(num_classes=len(names)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr,
                                momentum=opts.momentum, weight_decay=opts.weight_decay)
    global_step = 0

    if opts.last_ckpt:
        global_step, opts.start_epoch = load_ckpt(model, optimizer, opts.last_ckpt, device)

    lr_decay_lambda = lambda epoch: opts.lr_decay_rate ** (epoch // opts.lr_epoch_per_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    best_loss, best_loss_eval = float('inf'), float('inf')
    loss_func = nn.CrossEntropyLoss().to(device)#  使用交叉熵损失函数

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())# 时间戳，保存文件的名字
    writer = SummaryWriter('summary/' + timestamp)# 开启可视化loss下降网页窗口

    for epoch in range(int(opts.start_epoch), opts.epochs):
        scheduler.step(epoch)
        model.train().cuda()
        correct_all = torch.zeros(1).squeeze().cuda()
        total_all = torch.zeros(1).squeeze().cuda()
        correct = list(0. for i in range(len(opts.classes)))
        total = list(0. for i in range(len(opts.classes)))
        train_loss = 0

        for step, (batch_x, batch_y) in enumerate(train_dataLoader):
            batch_x = batch_x.to(device=device) # 使用gpu加载图片
            batch_y = batch_y.to(device=device) # 使用gpu加载标签
            batch_x = batch_x.type(torch.FloatTensor).cuda()
            batch_x = batch_x.cuda()
            optimizer.zero_grad()
            output = model(batch_x).cuda()# 网络结果的输出
            loss = loss_func(output, batch_y)# 计算loss
            train_loss = train_loss + loss.item()# 获得本次训练的loss(可能是多张图片训练即batch_size>1，所以在外循环设置一个临时变量得到该epoch的loss)
            if loss < best_loss:
                best_loss = loss
                save_ckpt(opts.ckpt_dir, model, optimizer, epoch, '_train')
            loss.backward()
            optimizer.step()
            global_step += 1

            prediction = torch.argmax(output, 1)  # 选择第一维度中最大的类别
            correct_all += (prediction == batch_y).sum().float()  # 得到的类别和标签的类别比对，并统计相同的数值
            total_all += len(batch_y)  # 一共的类别数也可以理解为一个batch_size

            res = prediction == batch_y  # 获得一组含有true和false的tensor，如果第一张图标签和预测一样为true否则为false以此类推
            for label_idx in range(len(batch_y)):
                label_single = batch_y[label_idx]
                correct[label_single] += res[label_idx].item()  # 统计单个类别中 预测正确的个数（如果为ture是1，false为0）
                total[label_single] += 1  # 统计单类别个数 比如在一个batch_size中CYST的个数

        print('epoch: %d | train loss: %.4f ' % (epoch, train_loss))


        if (epoch + 1) % 2 == 0:
            model.eval().cuda()
            correct_all = torch.zeros(1).squeeze().cuda()
            total_all = torch.zeros(1).squeeze().cuda()
            correct1 = list(0. for i in range(len(opts.classes)))
            total1 = list(0. for i in range(len(opts.classes)))
            eval_loss = 0
            for step, (batch_x, batch_y) in enumerate(val_dataLoader):
                batch_x = batch_x.to(device=device)  # 使用gpu加载图片
                batch_y = batch_y.to(device=device)  # 使用gpu加载标签
                batch_x = batch_x.type(torch.FloatTensor).cuda()
                batch_x = batch_x.cuda()
                output_val = model(batch_x).cuda()  # 网络结果的输出
                valid_loss = loss_func(output_val, batch_y)  # 计算loss

                eval_loss += valid_loss.item()
                if valid_loss < best_loss_eval:
                    best_loss_eval = valid_loss
                    save_ckpt(opts.ckpt_dir, model, optimizer, epoch, '_val')

                prediction = torch.argmax(output_val, 1)  # 选择第一维度中最大的类别
                correct_all += (prediction == batch_y).sum().float()  # 得到的类别和标签的类别比对，并统计相同的数值
                total_all += len(batch_y)  # 一共的类别数也可以理解为一个batch_size

                res = prediction == batch_y  # 获得一组含有true和false的tensor，如果第一张图标签和预测一样为true否则为false以此类推
                for label_idx in range(len(batch_y)):
                    label_single = batch_y[label_idx]
                    correct1[label_single] += res[label_idx].item()  # 统计单个类别中 预测正确的个数（如果为ture是1，false为0）
                    total1[label_single] += 1  # 统计单类别个数 比如在一个batch_size中CYST的个数

            f = open("PV.txt", "a+")
            acc_str = 'Accuracy: %f' % (sum(correct1) / sum(total1))
            for acc_idx in range(len(opts.classes)):
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
                writer.add_scalar("train_loss", train_loss, epoch) # 把值传入日志中
                writer.add_scalar("eval_loss", eval_loss, epoch)
                writer.add_scalar('val_Accuracy', (sum(correct1) / sum(total1)), epoch)  # 上面三行是把值传入日志中
                writer.add_scalar('train_Accuracy', (sum(correct) / sum(total)), epoch)  # 上面三行是把值传入日志中
                writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step=global_step)

            save_ckpt(opts.ckpt_dir, model, optimizer, epoch, '_last')
    writer.close()
    print("Training completed ")


if __name__ == '__main__':
    main()

