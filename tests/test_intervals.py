import numpy as np

from neuralopt.certification import Interval


def test_interval_random_containment():
    rng = np.random.default_rng(2)
    for _ in range(100):
        a, b = sorted(rng.uniform(-3, 3, 2))
        c, d = sorted(rng.uniform(-3, 3, 2))
        x, y = rng.uniform(a, b), rng.uniform(c, d)
        ia, ib = Interval(a, b), Interval(c, d)
        for got, true in [
            (ia + ib, x + y),
            (ia - ib, x - y),
            (ia * ib, x * y),
            (ia.square(), x * x),
            (ia.sin(), np.sin(x)),
        ]:
            assert got.lo <= true <= got.hi
