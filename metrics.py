import torch
from sklearn.metrics import hamming_loss
import numpy as np


class runningScore(object):  # 输入为Tesor
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = torch.zeros((n_classes, n_classes))

    def update(self, gts, preds):
        _, preds = preds.max(dim=1)
        for lt, lp in zip(gts, preds):
            assert type(lt) == type(lp)
            self.confusion_matrix[lt, lp] += 1
        # print(gts, preds)
        # print(self.confusion_matrix)

    def get_scores(self):
        hist = self.confusion_matrix
        acc = torch.diag(hist).sum() / hist.sum()
        # acc = torch.diag(hist[1:, 1:]).sum() / hist[1:, 1:].sum()
        # acc_cls = torch.zeros([4])
        # for i in range(4):
        #     if i == 0:
        #         acc_cls[i] = (hist[i, i] + hist[i+1:, i+1:].sum()) / hist.sum()
        #     elif i == 3:
        #         acc_cls[i] = (hist[i, i] + hist[:i, :i].sum()) / hist.sum()
        #     else:
        #         acc_cls[i] = (hist[i, i] + hist[:i, :i].sum() + hist[i+1:, i+1:].sum()
        #                       + hist[:i, i+1:].sum() + hist[i+1:, :i].sum()) / hist.sum()
        # acc = acc_cls[1:].mean()

        eps = 1e-7
        precision_cls = torch.diag(hist) / (hist.sum(dim=1) + eps)
        mean_precision = torch.mean(precision_cls)
        recall_cls = torch.diag(hist) / (hist.sum(dim=0) + eps)  # mean_recall = np.nanmean(recall_cls) 使用eps来避免NaN值
        mean_recall = torch.mean(recall_cls)
        TN = hist.sum() + torch.diag(hist) -hist.sum(dim=0) - hist.sum(dim=1)
        specificity_cls = TN / (TN + hist.sum(dim=1) - torch.diag(hist))
        mean_specificity = torch.mean(specificity_cls)

        iou = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist) + eps)  # 好像没错，但非方阵一定会报错
        f1_score_cls = 2 * iou / (1 + iou)
        f1_score = torch.mean(f1_score_cls)
        # freq = hist.sum(dim=1) / hist.sum()
        # fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

        return (
            {
                "Cls Overall Acc\t": acc.numpy(),
                "Cls Mean F1_Score\t": f1_score.numpy(),
                "Cls Mean precision\t": mean_precision.numpy(),
                "Cls Mean recall\t": mean_recall.numpy(),
                "Cls Mean specificity\t": mean_specificity.numpy(),
            },
            precision_cls,
            recall_cls,
            specificity_cls,
            f1_score_cls,
        )

    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # 转置
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(10, batch_time, losses, top1, prefix='Test: ')
    progress.print(1)