from __future__ import annotations
from typing import Callable

import torch
from torch import atan2, sqrt
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# class

class AdoptAtan2(Optimizer):
    """
    the proposed Adam substitute from University of Tokyo
    combined with the proposed atan2 method for ridding of the eps from Google

    Algorithm 3 in https://arxiv.org/abs/2411.02853
    """

    def __init__(
        self,
        params,
        lr = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay = 0.,
        regen_reg_rate = 0.,
        decoupled_wd = True,
        cautious_factor = 1., # set to 0. for zeroing out any updates not in same direction as gradient as in https://arxiv.org/abs/2411.16085
        a = 1.27,
        b = 1.
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.
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
            cautious_factor = cautious_factor
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

                grad, lr, wd, regen_rate, cautious_factor, beta1, beta2, a, b, state, init_lr = p.grad, group['lr'], group['weight_decay'], group['regen_reg_rate'], group['cautious_factor'], *group['betas'], group['a'], group['b'], self.state[p], self._init_lr

                # maybe decoupled weight decay

                if self.decoupled_wd:
                    wd /= init_lr

                # regenerative regularization from Kumar et al. https://arxiv.org/abs/2308.11958

                if regen_rate > 0. and 'param_init' in state:
                    param_init = state['param_init']
                    p.lerp_(param_init, lr / init_lr * regen_rate)

                # weight decay

                if wd > 0.:
                    p.mul_(1. - lr * wd)

                # init state if needed

                if len(state) == 0:
                    state['steps'] = 0
                    state['m'] = torch.zeros_like(grad)
                    state['v'] = grad * grad

                    if regen_rate > 0.:
                        state['param_init'] = p.clone()

                # get some of the states

                m, v, steps = state['m'], state['v'], state['steps']

                # for the first step do nothing

                if steps == 0:
                    state['steps'] += 1
                    continue

                # calculate m

                grad_sq = grad * grad

                update = grad.atan2(b * v.sqrt())

                m.lerp_(update, 1. - beta1)

                # maybe cautious update - algorithm 2 in https://arxiv.org/abs/2411.16085

                scale = 1.

                if cautious_factor < 1.:
                    align_mask = (update * grad) > 0
                    scale = torch.where(align_mask, torch.ones_like(grad), cautious_factor)
                    scale /= scale.mean().clamp(min = 1e-5)

                # then update parameters

                p.add_(m * scale, alpha = -lr * a)

                # update exp grad sq (v)

                v.lerp_(grad_sq, 1. - beta2)

                # increment steps

                state['steps'] += 1

        return loss
