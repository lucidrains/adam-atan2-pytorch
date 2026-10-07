"""Muon partitions must retain iterables and positional checkpoint identity."""

import copy
import itertools
import math

import pytest
import torch

from adam_atan2_pytorch.muon_adam_atan2 import MuonAdamAtan2


@pytest.mark.parametrize("remove_overlap", [True, False])
def test_generator_parameters_reach_both_optimizer_updates(remove_overlap):
    matrix = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    ordinary = [bias, matrix] if remove_overlap else [bias]
    optimizer = MuonAdamAtan2(
        (parameter for parameter in [matrix]),
        (parameter for parameter in ordinary),
        lr=0.01,
        betas=(0.0, 0.0),
        muon_beta1=0.0,
        muon_steps=0,
        remove_muon_params_from_params=remove_overlap,
    )
    assert [id(parameter) for parameter in optimizer.param_groups[0]["params"]] == [
        id(bias)
    ]
    assert [id(parameter) for parameter in optimizer.param_groups[1]["params"]] == [
        id(matrix)
    ]
    bias.grad = torch.ones_like(bias)
    matrix.grad = torch.ones_like(matrix)
    optimizer.step()
    # First Adam-atan2 step has unbiased m=v=1; zero NS iterations normalize
    # the matrix by its Frobenius norm, followed by the documented RMS factor.
    expected_bias = -0.01 * 1.27 * math.pi / 4
    expected_matrix = -0.01 * math.sqrt(2) * 0.2 / 2
    assert torch.allclose(
        bias, torch.full_like(bias, expected_bias), rtol=1e-12, atol=1e-14
    )
    assert torch.allclose(
        matrix, torch.full_like(matrix, expected_matrix), rtol=1e-12, atol=1e-14
    )
    assert optimizer.state[matrix]["steps"] == optimizer.state[bias]["steps"] == 1


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))))
def test_partition_preserves_every_supplied_non_muon_parameter_order(order):
    parameters = [
        torch.nn.Parameter(torch.ones(index + 1, dtype=torch.float64))
        for index in range(4)
    ]
    matrix = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
    supplied = [parameters[index] for index in order]
    supplied.insert(2, matrix)
    optimizer = MuonAdamAtan2([matrix], supplied)
    assert [id(parameter) for parameter in optimizer.param_groups[0]["params"]] == [
        id(parameters[index]) for index in order
    ]
    assert [id(parameter) for parameter in optimizer.param_groups[1]["params"]] == [
        id(matrix)
    ]


def test_partition_keeps_existing_overlap_and_duplicate_removal():
    first = torch.nn.Parameter(torch.ones(2))
    second = torch.nn.Parameter(torch.ones(3))
    matrix = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = MuonAdamAtan2([matrix], [second, matrix, second, first, matrix, first])
    assert [id(parameter) for parameter in optimizer.param_groups[0]["params"]] == [
        id(second),
        id(first),
    ]
    assert [id(parameter) for parameter in optimizer.param_groups[1]["params"]] == [
        id(matrix)
    ]


def test_checkpoint_round_trip_preserves_supplied_parameter_state_positions():
    original = [
        torch.nn.Parameter(torch.zeros(index + 2, dtype=torch.float64))
        for index in range(4)
    ]
    optimizer = MuonAdamAtan2([], original, lr=0.01, betas=(0.7, 0.9))
    for step in range(3):
        for index, parameter in enumerate(original):
            parameter.grad = torch.full_like(parameter, (index + 1) * (step + 1) / 10)
        optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    # Optimizer.state_dict uses group order as its positional parameter contract.
    assert checkpoint["param_groups"][0]["params"] == list(range(4))
    for index in range(4):
        expected_first = sum(
            0.3 * 0.7 ** (2 - step) * (index + 1) * (step + 1) / 10 for step in range(3)
        )
        expected_second = sum(
            0.1 * 0.9 ** (2 - step) * ((index + 1) * (step + 1) / 10) ** 2
            for step in range(3)
        )
        state = checkpoint["state"][index]
        assert state["exp_avg"].shape == original[index].shape
        assert torch.allclose(
            state["exp_avg"],
            torch.full_like(original[index], expected_first),
            rtol=1e-12,
            atol=1e-14,
        )
        assert torch.allclose(
            state["exp_avg_sq"],
            torch.full_like(original[index], expected_second),
            rtol=1e-12,
            atol=1e-14,
        )
    restored = [
        torch.nn.Parameter(parameter.detach().clone()) for parameter in original
    ]
    restored_optimizer = MuonAdamAtan2([], restored, lr=0.01, betas=(0.7, 0.9))
    restored_optimizer.load_state_dict(copy.deepcopy(checkpoint))
    for index, (parameter, restored_parameter) in enumerate(zip(original, restored)):
        gradient = torch.full_like(parameter, -(index + 1) / 10)
        parameter.grad = gradient.clone()
        restored_parameter.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    for parameter, restored_parameter in zip(original, restored):
        assert torch.equal(parameter, restored_parameter)
    # Loading and stepping the copy does not modify the saved checkpoint.
    for index in range(4):
        assert checkpoint["state"][index]["steps"] == 3
