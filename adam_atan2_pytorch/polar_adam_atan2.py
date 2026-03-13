from __future__ import annotations
import math
from typing import Callable

import torch
from torch import atan2, sqrt
from torch.optim.optimizer import Optimizer

from einops import pack, unpack
from itertools import repeat

# functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def transpose(t):
    return t.transpose(-1, -2)

# polar express variant
# https://arxiv.org/abs/2505.14387

POLAR_EXPRESS_COEFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically
]

# safety factor for numerical stability (but exclude last polynomial )
POLAR_EXPRESS_COEFS = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in POLAR_EXPRESS_COEFS[:-1]] + [POLAR_EXPRESS_COEFS[-1]]

def polar_express(
    t,
    steps = 5,
    coefs = POLAR_EXPRESS_COEFS,
    cast_bfloat16 = False,
    eps = 1e-7,
    bypass_update_fn: Callable[[int], bool] | None = None
):
    if exists(bypass_update_fn) and bypass_update_fn(t.ndim):
        return t

    shape = t.shape
    should_transpose = shape[-2] > shape[-1]

    if should_transpose:
        t = transpose(t)

    t, packed_shape = pack([t], '* i j')

    orig_dtype = t.dtype

    if cast_bfloat16:
        t = t.bfloat16()

    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)
    t = t / 1.01

    hs = coefs[:steps] + list(repeat(coefs[-1], max(0, steps - len(coefs))))

    for a, b, c in hs:
        A = t @ transpose(t)
        B = b * A + c * A @ A
        t = a * t + B @ t

    t, = unpack(t, packed_shape, '* i j')

    if should_transpose:
        t = transpose(t)

    if cast_bfloat16:
        t = t.to(orig_dtype)

    return t

# class

class PolarAdamAtan2(Optimizer):
    def __init__(
        self,
        polar_params,
        params,
        lr = 1e-4,
        polar_lr = None,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay = 0.,
        regen_reg_rate = 0.,
        decoupled_wd = False,
        cautious_factor = 1., # set to 0. for zeroing out any updates not in same direction as gradient as in https://arxiv.org/abs/2411.16085
        a = 1.27,
        b = 1.,
        polar_rms_factor = 0.2,
        polar_steps = 5,
        polar_beta1 = 0.95,
        polar_express_coefs = POLAR_EXPRESS_COEFS,
        polar_eps = 1e-7,
        polar_cast_bfloat16 = False,
        polar_bypass_update_fn: Callable[[int], bool] | None = lambda ndim: ndim < 2 or ndim > 3,
        remove_polar_params_from_params = True
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.
        assert regen_reg_rate >= 0.
        assert not (weight_decay > 0. and regen_reg_rate > 0.)
        assert 0. <= cautious_factor <= 1.

        self._init_lr = lr

        polar_lr = default(polar_lr, lr)
        self._init_polar_lr = polar_lr

        self.decoupled_wd = decoupled_wd

        beta1, beta2 = betas

        defaults = dict(
            lr = lr,
            beta1 = beta1,
            beta2 = beta2,
            a = a,
            b = b,
            weight_decay = weight_decay,
            regen_reg_rate = regen_reg_rate,
            cautious_factor = cautious_factor,
            use_polar = False,
            polar_steps = polar_steps,
            polar_express_coefs = polar_express_coefs,
            polar_eps = polar_eps,
            polar_rms_factor = polar_rms_factor,
            polar_cast_bfloat16 = polar_cast_bfloat16,
            polar_bypass_update_fn = polar_bypass_update_fn
        )

        if remove_polar_params_from_params:
            params = list(set(params) - set(polar_params))

        param_groups = [
            dict(params = params, lr = lr),
            dict(params = polar_params, lr = polar_lr, beta1 = polar_beta1, rms_factor = polar_rms_factor, use_polar = True)
        ]

        super().__init__(param_groups, defaults)

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

                use_polar = group.get('use_polar', False)

                grad, lr, wd, regen_rate, cautious_factor, beta1, beta2, a, b, state, init_lr, init_polar_lr = p.grad, group['lr'], group['weight_decay'], group['regen_reg_rate'], group['cautious_factor'], group['beta1'], group['beta2'], group['a'], group['b'], self.state[p], self._init_lr, self._init_polar_lr

                param_init_lr = init_lr if not use_polar else init_polar_lr

                # set lr scale to 1. if polar update

                if use_polar:
                    a = 1.

                # maybe decoupled weight decay

                if self.decoupled_wd:
                    wd /= param_init_lr

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

                    if not use_polar:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    if regen_rate > 0.:
                        state['param_init'] = p.clone()

                # get some of the states

                exp_avg, steps = state['exp_avg'], state['steps']

                steps += 1

                # bias corrections

                bias_correct1 = 1. - beta1 ** steps

                if not use_polar:
                    exp_avg_sq = state['exp_avg_sq']
                    bias_correct2 = 1. - beta2 ** steps

                # decay running averages

                exp_avg.lerp_(grad, 1. - beta1)

                if not use_polar:
                    exp_avg_sq.lerp_(grad * grad, 1. - beta2)

                    # the following line is the proposed change to the update rule
                    # using atan2 instead of a division with epsilon in denominator
                    # a * atan2(exp_avg / bias_correct1, b * sqrt(exp_avg_sq / bias_correct2))

                    den = exp_avg_sq.mul(b * b / bias_correct2).sqrt_()
                    update = exp_avg.mul(1. / bias_correct1).atan2_(den)

                    # maybe cautious update - algorithm 2 in https://arxiv.org/abs/2411.16085
                else:

                    polar_steps, polar_coefs, polar_eps, polar_rms_factor, polar_cast_bfloat16, polar_bypass_update_fn = group['polar_steps'], group['polar_express_coefs'], group['polar_eps'], group['polar_rms_factor'], group['polar_cast_bfloat16'], group['polar_bypass_update_fn']

                    # Polar Express from Amsel et al.
                    # https://arxiv.org/abs/2505.14387

                    update = polar_express(
                        exp_avg,
                        steps = polar_steps,
                        coefs = polar_coefs,
                        eps = polar_eps,
                        cast_bfloat16 = polar_cast_bfloat16,
                        bypass_update_fn = polar_bypass_update_fn
                    )

                    # incorporate the match adam RMS from Kimi team
                    # https://kexue.fm/archives/11267

                    polar_update_scale = math.sqrt(max(exp_avg.shape[-2:])) * polar_rms_factor

                    update = update * polar_update_scale

                if cautious_factor < 1.:
                    align_mask = (update * grad) > 0
                    scale = torch.where(align_mask, torch.ones_like(grad), cautious_factor)
                    update *= (scale / scale.mean().clamp(min = 1e-5))

                # update parameters

                p.add_(update, alpha = -lr * a)

                # increment steps

                state['steps'] = steps

        return loss
