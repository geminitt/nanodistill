"""Downloads are put together from copies whose only damage is zeroed blocks."""
import numpy as np
import pytest

fetch = pytest.importorskip("nanodistill.fetch")   # needs modal


def test_merge_repairs_zeroed_blocks_from_other_copies():
    B = fetch.BLOCK
    true = np.random.default_rng(0).integers(1, 255, 10 * B, dtype=np.uint8)
    a, b, c = true.copy(), true.copy(), true.copy()
    a[2 * B:4 * B] = 0; b[3 * B:5 * B] = 0; c[7 * B:8 * B] = 0    # overlapping damage in a and b
    assert (fetch.merge([a, b, c]) == true).all()


def test_merge_refuses_what_it_cannot_repair():
    B = fetch.BLOCK
    true = np.random.default_rng(1).integers(1, 255, 4 * B, dtype=np.uint8)
    a, b = true.copy(), true.copy()
    a[:B] = 0; b[:B] = 0                                             # zero in every copy
    with pytest.raises(RuntimeError):
        fetch.merge([a, b])
    c = true.copy(); c[5] ^= 1                                        # damage that is not zeros
    with pytest.raises(RuntimeError):
        fetch.merge([true, c])
