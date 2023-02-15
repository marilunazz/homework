from typing import List
import math
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Learning rate scheduler based on exponential decay.
    """

    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.num_epochs = num_epochs
        self.max_update = 40
        self.final_lr = 0
        self.warmup_steps = 0
        self.warmup_begin_lr = 0
        self.max_steps = self.max_update - self.warmup_steps
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_warmup_lr(self, epoch):
        increase = (
            (self.base_lrs[0] - self.warmup_begin_lr)
            * float(epoch)
            / float(self.warmup_steps)
        )
        return self.warmup_begin_lr + increase

    def get_lr(
        self,
    ) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        """
        Gets the learning rate
        """
        # k = 0.1
        # print(self.base_lrs)
        # lrs = [self.base_lrs[0] * (-k**epoch) for epoch in range(self.num_epoch + 1)]
        self.base_lr_orig = self.base_lrs[0]
        lrs = []
        for epoch in range(self.num_epochs):
            if epoch < self.warmup_steps:
                lrs.append(self.get_warmup_lr(epoch))
            if epoch <= self.max_update:
                lrs.append(
                    self.final_lr
                    + (self.base_lr_orig - self.final_lr)
                    * (
                        1
                        + math.cos(
                            math.pi * (epoch - self.warmup_steps) / self.max_steps
                        )
                    )
                    / 2
                )
        return lrs
