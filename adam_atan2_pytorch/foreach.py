from __future__ import annotations
from typing import Tuple, List, Callable

import torch
from torch import atan2, sqrt, Tensor
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

# slow foreach atan2

def slow_foreach_atan2_(nums: List[Tensor], dens: List[Tensor]):
    for num, den, in zip(nums, dens):
        num.atan2_(den)

# class

class AdamAtan2(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay = 0.,
        a = 1.27,
        b = 1.,
        foreach_atan2_fn: Callable | None = None
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.
        assert all([hasattr(torch, f'_foreach_{attr}_') for attr in ('mul', 'add', 'sign', 'lerp')]), 'this version of torch does not have the prerequisite foreach functions'

        self._init_lr = lr

        self._foreach_atan2_ = default(
            foreach_atan2_fn,
            getattr(torch, '_foreach_atan2_', None),
            slow_foreach_atan2_
        )

        defaults = dict(
            lr = lr,
            betas = betas,
            a = a,
            b = b,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None
    ):
        init_lr = self._init_lr

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            wd, lr, beta1, beta2, a, b = group['lr'], group['weight_decay'], *group['betas'], group['a'], group['b']

            # accumulate List[Tensor] for foreach inplace updates

            params = []
            grads = []
            grad_squared = []
            exp_avgs = []
            exp_avg_sqs = []

            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, state = p.grad, self.state[p]

                # decoupled weight decay

                if wd > 0.:
                    wd /= init_lr

                # init state if needed

                if len(state) == 0:
                    state['steps'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                # get some of the states

                exp_avg, exp_avg_sq, steps = state['exp_avg'], state['exp_avg_sq'], state['steps']

                steps += 1

                # bias corrections

                bias_correct1 = 1. - beta1 ** steps
                bias_correct2 = 1. - beta2 ** steps

                # append to list

                params.append(p)
                grads.append(grad)
                grad_squared.append(grad * grad)
                exp_avgs.append(exp_avg)
                exp_avg_sqs.append(exp_avg_sq)

                # update steps

                state['steps'] = steps

            # weight decay

            torch._foreach_mul_(params, 1. - lr * wd)

            # decay running averages

            torch._foreach_lerp_(exp_avgs, grads, 1. - beta1)
            torch._foreach_lerp_(exp_avg_sqs, grad_squared, 1. - beta2)

            # clone for update

            updates = [t.clone() for t in exp_avgs]
            den = [t.clone() for t in exp_avg_sqs]

            # calculate update atan2(exp_avg / bias_correct1, b * sqrt(exp_avg_sq / bias_correct2))

            torch._foreach_mul_(updates, 1. / bias_correct1)

            torch._foreach_mul_(den, b * b / bias_correct2)
            torch._foreach_sqrt_(den)

            self._foreach_atan2_(updates, den)

            # update params

            torch._foreach_add_(params, updates, alpha = -lr * a)

        return loss
