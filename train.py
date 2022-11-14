from pickle import TRUE
import yaml
import os
import time
import argparse
import numpy as np
import random
import warnings
import shutil

import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from loader.OCT2017_dataloader import OCT2017
from model import get_model
from metrics import AverageMeter, ProgressMeter, accuracy
from schedulers import WarmUpLR
from utils import get_logger
from metrics import runningScore
from label_smooth import LabelSmoothing

best_acc1 = 0

def main(model_name, num_classes, config, writer, logger, args):
    # 配置
    torch.cuda.set_device(args.local_rank)
    cudnn.benchmark = True
# 深度学习网络模型中初始的权值参数通常都是初始化成随机数
# 而使用梯度下降法最终得到的局部最优解对于初始位置点的选择很敏感
# 为了能够完全复现作者的开源深度学习代码，随机种子的选择能够减少一定程度上
# 算法结果的随机性，也就是更接近于原始作者的结果
# 即产生随机种子意味着每次运行实验，产生的随机数都是相同的
    if args.seed is not None:
        # 1. cudnn

        cudnn.benchmark = False        # 大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置 的高效算法，来达到优化运行效率的问题。
        cudnn.deterministic = True  #避免波动
        # 2. PyTorch
        torch.manual_seed(args.local_rank)#为cpu设置随机种子
        torch.cuda.manual_seed_all(args.local_rank)#为所有cuda设备设置随机种子
        # 3. Python & NumPy
        np.random.seed(args.seed)
        random.seed(args.seed)

        torch.set_printoptions(precision=10)  # 设置打印选项, precision–浮点数输出的精度位数
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    global best_acc1
    batch_size = config['train']['batch_size']

    # 导入数据
    train_dataset = OCT2017(augmentation=False, split="train",)
    trainloader = data.DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=config["train"]["num_workers"],
                                  pin_memory=True, shuffle=True)
    #batch_size:每个batch有多少个样本(int类型）
    #shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
    #如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.
    #这个参数决定了有几个进程来处理data loading


    val_dataset = OCT2017(augmentation=False, split="val",)
    valloader = data.DataLoader(val_dataset, batch_size=config['val']['batch_size'],
                                num_workers=config["val"]["num_workers"],
                                pin_memory=True, shuffle=True)
    # metrics
    # running_metrics_val = runningScore(val_dataset.num_class)

    # model
    if config["train"]["resume"] is not None:
        pre_train = False
    else:
        pre_train = True
        print('Use pre-trained model from ImageNet')
    model = get_model(model_name, num_classes=num_classes, in_channels=1, pretrained=pre_train).cuda()
    # model = EfficientNet.from_pretrained(model_name, num_classes=4, in_channels=1)
    model = model.cuda()

    optimizer_params = {k: v for k, v in config["train"]["optimizer"].items()}
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    # criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([2, 1])).cuda()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([3, 1])).cuda()
    criterion = LabelSmoothing(0.1).cuda()  # 试试用 0.1
    # if num_classes == 2:
    #     criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([3, 1])).cuda()
    # else:
    #     criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3, 1])).cuda()

    if config["train"]['warmup_epoch'] > 0:
        base_scheduler = MultiStepLR(optimizer, milestones=config["train"]["lr_schedule"]["milestones"])
        scheduler = WarmUpLR(optimizer, base_scheduler,  mode="linear", warmup_epoch=config["train"]['warmup_epoch'])
        print('Use WamrUp , warmup_epoch:', scheduler.warmup_epoch)
        print(scheduler.get_lr())
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=10, verbose=TRUE, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # resume from checkpoint
    start_epoch = 0
    if config["train"]["resume"] is not None:
        if os.path.isfile(config["train"]["resume"]):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(config["training"]["resume"]))
            checkpoint = torch.load(config["train"]["resume"])
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("Loaded checkpoint from epoch{}".format(checkpoint["epoch"]))

    # 验证
    if args.evaluate:
        print('start validate on val_set')
        validate(valloader, model, criterion, config, logger)
        return

    # 训练
    for epoch in range(start_epoch, config['train']['epoch']):
        train_acc1, train_loss = train(trainloader, model, criterion, optimizer, epoch, config, logger)
        val_acc1, val_loss = validate(valloader, model, criterion, config, logger)

        scheduler.step(epoch)

        # tensorboardX
        writer.add_scalars("loss", {'train_loss': train_loss, 'val_loss': val_loss}, epoch + 1)
        writer.add_scalars("Acc1", {'train_Acc1': train_acc1, 'val_acc1': val_acc1}, epoch + 1)
        writer.close

        # Save checkpoint
        if val_acc1 > best_acc1:
            best_acc1 = max(val_acc1, best_acc1)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(logdir, 'amp_checkpoint.pt'))


def train(trainloader, model, criterion, optimizer, epoch, config, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(trainloader), batch_time, data_time, losses, top1,
                             prefix="Epoch: [{}]".format(epoch))
    batch_size = config['train']['batch_size']

    # switch to train mode
    model.train()
    end = time.time()

    for i, (images, labels) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda()
        labels = labels.cuda()
        # compute output
        output = model(images)
        loss = criterion(input=output, target=labels)  # weight=torch.cuda.FloatTensor([])

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()  # 计算反向传播
        optimizer.step()  # 参数更新

        if i % config["train"]['print_interval'] == 0:
            # measure accuracy and record loss
            acc1 = accuracy(output.data, labels, topk=(1, ))[0]  # 注意，前k个预测类别的精度，k不要超过自己的类别数

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update((time.time() - end) / config["train"]['print_interval'])
            end = time.time()  # print的时间不会影响吗？？
            print('lr=%d'%optimizer.state_dict()['param_groups'][0]['lr'])
            print_str = "Epoch: [{0}][{1}/{2}]\t" \
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t" \
                        "Speed {3:.3f} ({4:.3f})\t" \
                        "Loss {loss.val:.10f} ({loss.avg:.4f})\t" \
                        "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                         epoch, i, len(trainloader),
                         batch_size / batch_time.val,
                         batch_size / batch_time.avg,
                         batch_time=batch_time,
                         loss=losses, top1=top1)
            logger.info(print_str)
            print(print_str)
            # progress.print(i)
    return top1.avg, losses.avg


def validate(valloader, model, criterion, config, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(valloader), batch_time, losses, top1, prefix='Test: ')
    batch_size = config['val']['batch_size']
              
    # switch to evaluate mode
    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(valloader):
            images = images.cuda()
            labels = labels.cuda()

            output = model(images)
            loss = criterion(output, labels)

            acc1 = accuracy(output.data, labels, topk=(1, ))[0]

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config["val"]['print_interval'] == 0:
                # TODO: this should also be done with the ProgressMeter
                print_str = "Test: [{0}/{1}]\t" \
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t" \
                            "Speed {2:.3f} ({3:.3f})\t" \
                            "Loss {loss.val:.4f} ({loss.avg:.4f})\t" \
                            "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                                i, len(valloader),
                                batch_size / batch_time.val,
                                batch_size / batch_time.avg,
                                batch_time=batch_time,
                                loss=losses, top1=top1)
                logger.info(print_str)
                print(print_str)
            # progress.print(i)
    print(' * Overall Prec@1 On Val Dataset: {top1.avg:.3f}'.format(top1=top1))
    logger.info(' * Overall Prec@1 On Val Dataset: {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('-e', '--evaluate', dest='evaluate', default=False, action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--sync_bn', default=False, action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed', default=None, type=int,
                        help='fix seed. ')
    args = parser.parse_args()

    model_name = 'resnet18'  # 'efficientnet-b0'
    num_classes = 2
    config_path = 'config/config.yml'

    with open(config_path) as f:
        config = yaml.load(f)

    if config["train"]["resume"] is not None:
        logdir = config["train"]["resume"][:-len(config["train"]["resume"].split('/')[-1])]
    else:
        logdir = os.path.join('runs', model_name, time.strftime('%m%d_%H%M', time.localtime(time.time())))  # 以时间区分log文件夹

    writer = SummaryWriter(logdir)
    logger = get_logger(logdir)
    shutil.copy(config_path, logdir)

    main(model_name, num_classes, config, writer, logger, args)


# python train.py