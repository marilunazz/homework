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
        self.max_steps = 40
        self.final_lr = 0
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(
        self,
    ) -> List[float]:
        """
        Gets the learning rate
        """
        lrs = []
        for epoch in range(self.num_epochs):
            if epoch <= self.max_steps:
                lrs.append(
                    self.final_lr
                    + (self.base_lrs[0] - self.final_lr)
                    * (1 + math.cos(math.pi * epoch / self.max_steps))
                    / 2
                )
        return lrs
