import torch

from model import get_sequences
import numpy as np


def test_get_sequences_fwd():
    # GIVEN
    seq = [torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2, 3, 4])]
    # WHEN
    res, lens = get_sequences(seq)
    # THEN
    assert res.size() == (4, 2)
    assert lens.numpy().tolist() == [4, 3]
    np.testing.assert_array_equal(res[:, 0], [1, 2, 3, 4])


def test_get_sequences_bwd():
    # GIVEN
    seq = [torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2, 3, 4])]
    # WHEN
    res, lens = get_sequences(seq, reverse=True)
    # THEN
    assert res.size() == (4, 2)
    assert lens.numpy().tolist() == [4, 3]
    np.testing.assert_array_equal(res[:, 0], [4, 3, 2, 1])
