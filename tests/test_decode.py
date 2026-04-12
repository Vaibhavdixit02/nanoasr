import torch
from nanoasr.decode import greedy_decode, greedy_decode_batch
from nanoasr.vocab import BLANK_IDX


def _make_logprobs(indices, vocab_size=28):
    """Create fake log_probs where the given index wins at each timestep."""
    T = len(indices)
    lp = torch.full((T, vocab_size), -100.0)
    for t, idx in enumerate(indices):
        lp[t, idx] = 0.0
    return lp


def test_greedy_decode_hi():
    # blank, blank, h(7), h(7), blank, i(8), blank, blank, blank, blank
    indices = [27, 27, 7, 7, 27, 8, 27, 27, 27, 27]
    lp = _make_logprobs(indices)
    assert greedy_decode(lp) == "hi"


def test_greedy_decode_hello():
    # h, blank, e, blank, l, blank, l, blank, o
    indices = [7, 27, 4, 27, 11, 27, 11, 27, 14]
    lp = _make_logprobs(indices)
    assert greedy_decode(lp) == "hello"


def test_greedy_decode_all_blank():
    indices = [BLANK_IDX] * 10
    lp = _make_logprobs(indices)
    assert greedy_decode(lp) == ""


def test_greedy_decode_repeated_char():
    # a, a, a without blanks -> collapses to single "a"
    indices = [0, 0, 0]
    lp = _make_logprobs(indices)
    assert greedy_decode(lp) == "a"


def test_greedy_decode_batch_basic():
    # batch of 2, different lengths
    lp1 = _make_logprobs([7, 27, 8])          # "hi"
    lp2 = _make_logprobs([0, 27, 1, 27, 2])   # "abc"
    max_T = 5
    batch = torch.full((2, max_T, 28), -100.0)
    batch[0, :3] = lp1
    batch[1, :5] = lp2
    lengths = torch.tensor([3, 5])
    results = greedy_decode_batch(batch, lengths)
    assert results == ["hi", "abc"]
