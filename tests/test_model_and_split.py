import torch

from data import split_indices
from model import BasicBlock, ResNet


def test_residual_connection_changes_output():
    block = BasicBlock(64, 64).eval()
    x = torch.randn(2, 64, 8, 8)
    with torch.no_grad():
        with_skip = block(x)
        block.residual = False
        without_skip = block(x)
    assert with_skip.shape == without_skip.shape
    assert not torch.allclose(with_skip, without_skip)


def test_model_outputs_ten_logits_at_both_depths():
    for depth in (18, 34):
        model = ResNet(depth).eval()
        with torch.no_grad():
            assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_split_is_reproducible_and_disjoint():
    train, validation = split_indices(100, 20, 42)
    assert len(train) == 80 and len(validation) == 20
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(100))
    assert (train, validation) == split_indices(100, 20, 42)
