from __future__ import annotations
from typing import Callable

import torch
from torch import atan2, sqrt
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# class

class Adopt(Optimizer):
    """
    the proposed Adam substitute from University of Tokyo

    Algorithm 2 in https://arxiv.org/abs/2411.02853
    """

    def __init__(
        self,
        params,
        lr = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        eps = 1e-6,
        weight_decay = 0.,
        decoupled_wd = True
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.

        self._init_lr = lr
        self.decoupled_wd = decoupled_wd

        defaults = dict(
            lr = lr,
            betas = betas,
            eps = eps,
            weight_decay = weight_decay,
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

                grad, lr, wd, beta1, beta2, eps, state, init_lr = p.grad, group['lr'], group['weight_decay'], *group['betas'], group['eps'], self.state[p], self._init_lr

                # maybe decoupled weight decay

                if self.decoupled_wd:
                    wd /= init_lr

                # weight decay

                if wd > 0.:
                    p.mul_(1. - lr * wd)

                # init state if needed

                if len(state) == 0:
                    state['steps'] = 0
                    state['m'] = torch.empty_like(grad)
                    state['v'] = grad * grad

                # get some of the states

                m, v, steps = state['m'], state['v'], state['steps']

                # for the first step do nothing

                if steps == 0:
                    state['steps'] += 1
                    continue

                # logic

                steps += 1

                # calculate m

                grad_sq = grad * grad

                next_m = grad.div(v.sqrt().clamp(min = eps)) # they claim that a max(value, eps) performs better than adding the epsilon

                m.lerp_(next_m, 1. - (beta1 * int(steps > 1)))

                # then update parameters

                p.add_(m, alpha = -lr)

                # update exp grad sq (v)

                v.lerp_(grad_sq, 1. - beta2)

                # increment steps

                state['steps'] = steps

        return loss
