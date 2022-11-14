import torch
import time
import os
import argparse

from torch.utils import data
import torch.nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from loader.OCT2017_dataloader import OCT2017
from model import get_model, get_resize_shape
from utils import fast_collate, reduce_tensor, data_prefetcher, \
    show_score, show_class_metrics, M_model_weihgt2S_model_weight
from metrics import runningScore, AverageMeter, accuracy


def main(model_name, weight_dir, args, ngpus_per_node=2):
    cudnn.benchmark = True
    gpu = args.local_rank
    world_size = 1
    batch_size = args.batch_size

    if args.distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)

        gpu = args.local_rank
        world_size = dist.get_world_size()
        batch_size = int(args.batch_size / ngpus_per_node)
    torch.cuda.set_device(gpu)

    test_dataset = OCT2017(augmentation=False, split="test",
                           is_transforms=False,
                           resize_shape=get_resize_shape(model_name)
                           )
    test_sampler = None
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    testloader = data.DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 shuffle=True,
                                 sampler=test_sampler,
                                 collate_fn=fast_collate)
    running_metrics = runningScore(test_dataset.num_class)

    model = get_model(model_name, num_classes=2, in_channels=1).cuda()
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        # 测试时不再使用混合精度

    if os.path.isfile(weight_dir):
        print("Loading model and optimizer from checkpoint '{}'".format(weight_dir))
        checkpoint = torch.load(weight_dir, lambda storage, loc: storage.cuda(args.local_rank))
        # model.load_state_dict_test(checkpoint["model"])  # 单GPU训练
        print(checkpoint['epoch'])
        if args.distributed:
            model.load_state_dict(checkpoint["model"])
        else:
            M_model_weihgt2S_model_weight(model, checkpoint["model"])  # 是为分布式训练准备的

    if args.distributed:
        multi_gpu_test(model, testloader, batch_size, world_size, running_metrics)
    else:
        single_gpu_test(model, testloader, running_metrics)


def single_gpu_test(model, data_loader, running_metrics):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    prefetcher = data_prefetcher(data_loader)
    images, labels = prefetcher.next()
    i = 0
    while images is not None:
        i += 1
        with torch.no_grad():
            output = model(images)
        acc1 = accuracy(output.data, labels, topk=(1,))[0]
        top1.update(acc1[0], images.size(0))
        running_metrics.update(preds=output.data, gts=labels)
        print('Test: [{}/{}]\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
               i, len(data_loader),
               top1=top1))
        images, labels = prefetcher.next()

    # 显示整体metrics
    show_metrics(running_metrics)


def show_metrics(running_metrics):
    # metrics
    print("\ntest metrics")
    score, class_precision, class_recall, class_specificity, class_f1 = running_metrics.get_scores()
    show_score(score)
    show_class_metrics("f1_score", class_f1, running_metrics.n_classes)
    show_class_metrics("precision", class_precision, running_metrics.n_classes)
    show_class_metrics("recall", class_recall, running_metrics.n_classes)
    show_class_metrics("specificity", class_specificity, running_metrics.n_classes)


def multi_gpu_test(model, testloader, batch_size, world_size, running_metrics):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(testloader)
    images, labels = prefetcher.next()
    i = 0
    while images is not None:
        i += 1
        # compute output
        with torch.no_grad():
            output = model(images)

        # measure accuracy and record loss
        acc1 = accuracy(output.data, labels, topk=(1, ))[0]
        acc1 = reduce_tensor(acc1[0], world_size)

        top1.update(acc1, images.size(0))
        # TODO: running_metrics是否需要reduce_tensor 待测试  要
        running_metrics.update(preds=output.data, gts=labels)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        images, labels = prefetcher.next()
        if args.local_rank == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(testloader),
                    world_size * batch_size / batch_time.val,
                    world_size * batch_size / batch_time.avg,
                    batch_time=batch_time,
                    top1=top1))

    # 显示整体metrics
    if args.local_rank == 0:  # 要用torch.distributed.launch 不然args.local_rank需要传入函数
        show_metrics(running_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--distributed', default=False, action='store_true',
                        help='multi_gpu_test')
    parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size per process (default: 4)')
    parser.set_defaults(evaluate=False)
    args = parser.parse_args()
    main('resnet18', './runs/resnet18/0706_1426/amp_checkpoint.pt', args, ngpus_per_node=1)
# 多GPU测试
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 test.py --distributed
# 单GPU测试
# python test.py --local_rank 0
