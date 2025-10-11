from __future__ import annotations
from typing import Callable

import torch
from torch import atan2, sqrt
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# muon related

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):
    if t.ndim <= 3:
        return t

    shape = t.shape
    should_transpose = shape[-2] > shape[-1]

    if should_transpose:
        t = t.transpose(-1, -2)

    t, packed_shape = pack([t], '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ t.transpose(-1, -2)
        B = b * A + c * A @ A
        t = a * t + B @ t

    t, = unpack(t, packed_shape, '* i j')

    if should_transpose:
        t = t.transpose(-1, -2)

    return t

# class

class MuonAdamAtan2(Optimizer):
    def __init__(
        self,
        muon_params,
        params,
        lr = 1e-4,
        muon_lr = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay = 0.,
        regen_reg_rate = 0.,
        decoupled_wd = False,
        cautious_factor = 1., # set to 0. for zeroing out any updates not in same direction as gradient as in https://arxiv.org/abs/2411.16085
        a = 1.27,
        b = 1.,
        muon_steps = 5,
        muon_beta1 = 0.95,
        muon_newton_schulz5_coefs = (3.4445, -4.7750, 2.0315),
        muon_eps = 1e-7,
        remove_muon_params_from_params = True
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.
        assert regen_reg_rate >= 0.
        assert not (weight_decay > 0. and regen_reg_rate > 0.)
        assert 0. <= cautious_factor <= 1.

        self._init_lr = lr

        muon_lr = default(muon_lr, lr)
        self._init_muon_lr = muon_lr

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
            use_muon = False,
            muon_steps = muon_steps,
            muon_newton_schulz5_coefs = muon_newton_schulz5_coefs,
            muon_eps = muon_eps,
        )

        if remove_muon_params_from_params:
            params = list(set(params) - set(muon_params))

        param_groups = [
            dict(params = params, lr = lr),
            dict(params = muon_params, lr = muon_lr, beta1 = muon_beta1, use_muon = True)
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

                use_muon = group['use_muon']

                grad, lr, wd, regen_rate, cautious_factor, beta1, beta2, a, b, state, init_lr, init_muon_lr = p.grad, group['lr'], group['weight_decay'], group['regen_reg_rate'], group['cautious_factor'], group['beta1'], group['beta2'], group['a'], group['b'], self.state[p], self._init_lr, self._init_muon_lr

                param_init_lr = init_lr if not use_muon else init_muon_lr

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

                    if not use_muon:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    if regen_rate > 0.:
                        state['param_init'] = p.clone()

                # get some of the states

                exp_avg, steps = state['exp_avg'], state['steps']

                steps += 1

                # bias corrections

                bias_correct1 = 1. - beta1 ** steps

                if not use_muon:
                    exp_avg_sq = state['exp_avg_sq']
                    bias_correct2 = 1. - beta2 ** steps

                # decay running averages

                exp_avg.lerp_(grad, 1. - beta1)

                if not use_muon:
                    exp_avg_sq.lerp_(grad * grad, 1. - beta2)

                    # the following line is the proposed change to the update rule
                    # using atan2 instead of a division with epsilon in denominator
                    # a * atan2(exp_avg / bias_correct1, b * sqrt(exp_avg_sq / bias_correct2))

                    den = exp_avg_sq.mul(b * b / bias_correct2).sqrt_()
                    update = exp_avg.mul(1. / bias_correct1).atan2_(den)

                    # maybe cautious update - algorithm 2 in https://arxiv.org/abs/2411.16085
                else:

                    muon_steps, muon_coefs, muon_eps = group['muon_steps'], group['muon_newton_schulz5_coefs'], group['muon_eps']

                    # Muon from Keller Jordan
                    # https://kellerjordan.github.io/posts/muon/

                    update = newtonschulz5(
                        exp_avg,
                        steps = muon_steps,
                        coefs = muon_coefs,
                        eps = muon_eps
                    )

                if cautious_factor < 1.:
                    align_mask = (update * grad) > 0
                    scale = torch.where(align_mask, torch.ones_like(grad), cautious_factor)
                    update *= (scale / scale.mean().clamp(min = 1e-5))

                # update parameters

                p.add_(update, alpha = -lr * a)

                # increment steps

                state['steps'] = steps

        return loss
