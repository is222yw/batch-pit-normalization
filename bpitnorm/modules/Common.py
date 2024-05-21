import torch
from torch import Tensor
from math import sqrt

_sqrt2 = sqrt(2.0)


def standard_normal_ppf(x: Tensor, min_: float = 9e-8, max_: float = 1.0 - 9e-8) -> Tensor:
    """
    Clips a tensor such that the result does not become NaN or inf.
    """
    # Values smaller/larger than the following will return (-)inf,
    # so we gotta clip them.
    x = torch.clip(input=x, min=min_, max=max_)
    res = _sqrt2 * torch.special.erfinv(2.0 * x - 1.0)
    assert not torch.any(torch.isnan(res)) and not torch.any(torch.isinf(res))
    return res


def standard_normal_cdf(x: Tensor) -> Tensor:
    """
    A standard normal CDF (mean=0, var=1).
    """
    res = 0.5 * (1.0 + torch.special.erf(x / _sqrt2))
    return res
