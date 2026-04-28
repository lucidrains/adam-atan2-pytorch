from __future__ import annotations
from typing import Callable

import torch
from torch import atan2, sqrt
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# tensor helpers

def orthog_proj(x, y, double_precision = False):
    assert x.shape == y.shape
    shape = x.shape

    x, y = x.flatten(), y.flatten()

    if double_precision:
        dtype = x.dtype
        x, y = x.double(), y.double()

    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    out = orthogonal.reshape(*shape)

    if double_precision:
        out = out.to(dtype)

    return out

# class

class AdamAtan2(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay = 0.,
        regen_reg_rate = 0.,
        orig_grad_ema_beta = 0.9,
        orthog_grad = True,
        orthog_proj_double_precision = True,
        decoupled_wd = False,
        cautious_wd = False,
        cautious_factor = 1., # set to 0. for zeroing out any updates not in same direction as gradient as in https://arxiv.org/abs/2411.16085
        a = 1.27,
        b = 1.
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.
        assert regen_reg_rate >= 0.
        assert not (weight_decay > 0. and regen_reg_rate > 0.)
        assert 0. <= cautious_factor <= 1.

        self._init_lr = lr
        self.decoupled_wd = decoupled_wd

        defaults = dict(
            lr = lr,
            betas = betas,
            a = a,
            b = b,
            weight_decay = weight_decay,
            regen_reg_rate = regen_reg_rate,
            orthog_grad = orthog_grad,
            orig_grad_ema_beta = orig_grad_ema_beta,
            orthog_proj_double_precision = orthog_proj_double_precision,
            cautious_wd = cautious_wd,
            cautious_factor = cautious_factor
        )

        super().__init__(params, defaults)

        # independent of lr

        if decoupled_wd:
            for group in self.param_groups:
                group['weight_decay'] /= lr
                group['regen_reg_rate'] /= lr

    # resetting the ema of the original grad, say at the boundary of a new video or trajectory, or could even be determined by the uncertainty of some predictive module

    @torch.no_grad()
    def reset_(
        self,
        key = None
    ):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                if 'grad_emas' not in state:
                    continue

                if not exists(key):
                    state['grad_emas'].clear()
                elif key in state['grad_emas']:
                    state['grad_emas'][key].zero_()

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None,
        store_grad_key: str | None = 'orig_grad',
        orthog_against_key: str | None = 'orig_grad'
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, regen_rate, orthog_grad, orig_grad_ema_beta, orthog_proj_double_precision, cautious_wd, cautious_factor, beta1, beta2, a, b, state, init_lr = p.grad, group['lr'], group['weight_decay'], group['regen_reg_rate'], group['orthog_grad'], group['orig_grad_ema_beta'], group['orthog_proj_double_precision'], group['cautious_wd'], group['cautious_factor'], *group['betas'], group['a'], group['b'], self.state[p], self._init_lr

                # regenerative regularization from Kumar et al. https://arxiv.org/abs/2308.11958

                if regen_rate > 0. and 'param_init' in state:
                    param_init = state['param_init']
                    p.lerp_(param_init, lr * regen_rate)

                # init state if needed

                if len(state) == 0:
                    state['steps'] = 0

                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                    state['grad_emas'] = dict()

                    if regen_rate > 0.:
                        state['param_init'] = p.clone()

                # get some of the states

                exp_avg, exp_avg_sq, steps = state['exp_avg'], state['exp_avg_sq'], state['steps']

                steps += 1

                # orthogonal gradients for decorrelating successive gradient updates
                # e.g. for learning on temporal streams (https://arxiv.org/abs/2504.01961)
                # or auxiliary RL losses to prevent clashing between representation learning and policy gradients

                if orthog_grad:
                    new_grad = grad
                    grad_emas = state['grad_emas']

                    if exists(orthog_against_key) and orthog_against_key in grad_emas:
                        new_grad = orthog_proj(grad, grad_emas[orthog_against_key], double_precision = orthog_proj_double_precision)

                    if exists(store_grad_key):
                        if store_grad_key not in grad_emas:
                            grad_emas[store_grad_key] = torch.zeros_like(grad)

                        grad_emas[store_grad_key].lerp_(grad, 1. - orig_grad_ema_beta)

                    grad = new_grad

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

                # maybe cautious update - algorithm 2 in https://arxiv.org/abs/2411.16085

                if cautious_factor < 1.:
                    align_mask = (update * grad) > 0
                    scale = torch.where(align_mask, torch.ones_like(grad), cautious_factor)
                    update *= (scale / scale.mean().clamp(min = 1e-5))

                # maybe weight decay

                if wd > 0.:
                    # maybe cautious
                    # https://arxiv.org/abs/2510.12402

                    wd_mask = (update * p > 0).float() if cautious_wd else 1.

                    p.mul_(1. - lr * wd * wd_mask)

                # update parameters

                p.add_(update, alpha = -lr * a)

                # increment steps

                state['steps'] = steps

        return loss
