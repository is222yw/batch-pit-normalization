import torch
from torch import Tensor
from math import sqrt

_sqrt2 = sqrt(2.0)


def standard_normal_ppf(x: Tensor) -> Tensor:
    """
    Clips a tensor such that the result does not become NaN or inf.
    """
    # Values smaller/larger than the following will return (-)inf,
    # so we gotta clip them.
    _min = 9e-8
    _max = 1.0 - _min
    x = torch.clip(input=x, min=_min, max=_max)
    res = _sqrt2 * torch.special.erfinv(2.0 * x - 1.0)
    assert not torch.any(torch.isnan(res)) and not torch.any(torch.isinf(res))
    return res


def standard_normal_cdf(x: Tensor) -> Tensor:
    """
    A standard normal CDF (mean=0, var=1).
    """
    return 0.5 * (1.0 + torch.special.erf(x / _sqrt2))
