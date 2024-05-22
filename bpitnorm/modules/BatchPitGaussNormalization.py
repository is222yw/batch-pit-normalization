from torch import nn, Tensor, device
from torch.nn import BatchNorm1d
from torch.distributions.normal import Normal
from bpitnorm.modules.Common import standard_normal_ppf



def make_default_batchnorm1d(num_feats: int, dev: device | str) -> BatchNorm1d:
    """
    Creates an instance of BatchNorm1d with default parameters.
    Sets affine=True and track_running_stats=True as required
    by BatchPitGaussNorm1d.

    Author: Sebastian Hönel
    """
    assert isinstance(num_feats, int) and num_feats > 0, 'You must provide the number of features as a positive integer.'
    return BatchNorm1d(num_features=num_feats, affine=True, track_running_stats=True, device=dev)




class BatchPitGaussNorm1d(nn.Module):
    """
    A simple but potentially also effective version of the BatchPitNormalization
    that uses the CDF of a normal distribution to perform the PIT, based on the
    running mean and variance of each feature. Uses BatchNorm1d internally to
    track these
    """
    def __init__(self, dev: device | str, bn1d: BatchNorm1d=None, normal_backtransform: bool=False, num_feats: int=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.normal_backtransform = normal_backtransform

        if not isinstance(bn1d, BatchNorm1d):
            bn1d = make_default_batchnorm1d(num_feats=num_feats, dev=dev)

        assert bn1d.affine and bn1d.track_running_stats, 'The provided BatchNorm1d must have affine=True and track_running_stats=True.'
        self.bn1d = bn1d

    

    def forward(self, x: Tensor) -> Tensor:
        self.bn1d.train(mode=self.training)
        self.bn1d.forward(x)

        running_mean = self.bn1d.get_buffer('running_mean')
        running_var = self.bn1d.get_buffer('running_var')

        n = Normal(loc=running_mean, scale=running_var)
        x = n.cdf(x)

        if self.normal_backtransform:
            x = standard_normal_ppf(x=x)
        else:
            x -= 0.5
        return x
