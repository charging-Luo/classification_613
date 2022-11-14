from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    # 有助于减缓模型在初始阶段对mini-batch的提前过拟合现象，保持分布的平稳
    # 有助于保持模型深层的稳定性
    # https://www.zhihu.com/question/338066667
    # 直观理解就是最开始的时候，loss大，如果学习率太大，gradient也就弄的很大，容易崩，结果什么都学不到。
    # 所以最开始步子小一些，等模型收敛到合适的位置，loss不再爆炸，再加大学习率开始快速学习
    # warmup_iters设置为epoch的2~3倍   或者等待training error below 80%
    def __init__(
        self, optimizer, scheduler, mode="linear", warmup_epoch=2, gamma=0.2, last_epoch=-1
    ):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_epoch = warmup_epoch
        self.gamma = gamma
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cold_lrs = self.scheduler.get_lr()

        if self.last_epoch < self.warmup_epoch:
            if self.mode == "linear":
                alpha = self.last_epoch / float(self.warmup_epoch)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == "constant":
                factor = self.gamma
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.mode))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs