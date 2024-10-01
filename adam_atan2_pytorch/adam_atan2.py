from __future__ import annotations
from typing import Callable

import torch
from torch import atan2, sqrt
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# class

class AdamAtan2(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay = 0.,
        regen_reg_rate = 0.,
        decoupled_wd = False,
        a = 1.27,
        b = 1.
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.
        assert regen_reg_rate >= 0.
        assert not (weight_decay > 0. and regen_reg_rate > 0.)

        self._init_lr = lr
        self.decoupled_wd = decoupled_wd

        defaults = dict(
            lr = lr,
            betas = betas,
            a = a,
            b = b,
            weight_decay = weight_decay,
            regen_reg_rate = regen_reg_rate,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, regen_rate, beta1, beta2, a, b, state, init_lr = p.grad, group['lr'], group['weight_decay'], group['regen_reg_rate'], *group['betas'], group['a'], group['b'], self.state[p], self._init_lr

                # maybe decoupled weight decay

                if self.decoupled_wd:
                    wd /= init_lr

                # weight decay

                if wd > 0.:
                    p.mul_(1. - lr * wd)

                # regenerative regularization from Kumar et al. https://arxiv.org/abs/2308.11958

                if regen_rate > 0. and 'param_init' in state:
                    param_init = state['param_init']
                    p.lerp_(param_init, lr / init_lr * regen_rate)

                # init state if needed

                if len(state) == 0:
                    state['steps'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                    if regen_rate > 0.:
                        state['param_init'] = p.clone()

                # get some of the states

                exp_avg, exp_avg_sq, steps = state['exp_avg'], state['exp_avg_sq'], state['steps']

                steps += 1

                # bias corrections

                bias_correct1 = 1. - beta1 ** steps
                bias_correct2 = 1. - beta2 ** steps

                # decay running averages

                exp_avg.lerp_(grad, 1. - beta1)
                exp_avg_sq.lerp_(grad * grad, 1. - beta2)

                # the following line is the proposed change to the update rule
                # using atan2 instead of a division with epsilon in denominator
                # a * atan2(exp_avg / bias_correct1, b * sqrt(exp_avg_sq / bias_correct2))

                den = exp_avg_sq.mul(b * b / bias_correct2).sqrt_()
                update = exp_avg.mul(1. / bias_correct1).atan2_(den)

                # update parameters

                p.add_(update, alpha = -lr * a)

                # increment steps

                state['steps'] = steps

        return loss
