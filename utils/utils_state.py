import math
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


# https://github.com/XPixelGroup/ClassSR
class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            if self.restarts.index(self.last_epoch) + 1 == len(self.T_period):
                print('Already trained.')
                exit()
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]
    

def make_optim_sched(param_list, optimizer_spec, lr_scheduler_spec, load_sd=False):
    Optimizer = {
        'adam': Adam
    }[optimizer_spec['name']]
    Scheduler = {
        'CosineAnnealingLR_Restart': CosineAnnealingLR_Restart,
        'CosineAnnealingLR': CosineAnnealingLR
    }[lr_scheduler_spec['name']]

    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    lr_scheduler = Scheduler(optimizer, **lr_scheduler_spec['args'])
    if load_sd: # jointly loading state_dict with all initialized does matter
        optimizer.load_state_dict(optimizer_spec['sd'])
        lr_scheduler.load_state_dict(lr_scheduler_spec['sd'])
    return optimizer, lr_scheduler