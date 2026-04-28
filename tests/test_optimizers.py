import torch
from torch import nn

from adam_atan2_pytorch.adam_atan2_with_orthog_grad import AdamAtan2

def test_rl_aux_orthog():
    model = nn.Linear(10, 2)
    opt = AdamAtan2(model.parameters(), lr = 1e-3, orthog_grad = True)
    
    state = torch.randn(4, 10)

    opt.zero_grad()
    model(state).sum().backward()
    opt.step(store_grad_key = 'rl_grad', orthog_against_key = None)

    opt.zero_grad()
    (model(state) ** 2).sum().backward()
    
    opt.step(store_grad_key = None, orthog_against_key = 'rl_grad')

def test_reset_emas():
    model = nn.Linear(10, 2)
    opt = AdamAtan2(model.parameters(), lr = 1e-3, orthog_grad = True)
    
    state = torch.randn(4, 10)

    opt.zero_grad()
    model(state).sum().backward()
    opt.step(store_grad_key = 'rl_grad', orthog_against_key = None)
    opt.step(store_grad_key = 'aux_grad', orthog_against_key = None)

    # verify nested keys exist
    opt_state = opt.state[model.weight]
    assert 'rl_grad' in opt_state['grad_emas']
    assert 'aux_grad' in opt_state['grad_emas']

    # test reset specific key
    opt.reset_(key='rl_grad')
    assert opt_state['grad_emas']['rl_grad'].norm().item() == 0.0
    assert opt_state['grad_emas']['aux_grad'].norm().item() > 0.0

    # test reset all
    opt.reset_()
    assert len(opt_state['grad_emas']) == 0
