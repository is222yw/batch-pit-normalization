import torch
import numpy as np
import unittest

from bpitnorm.modules.BatchPitGaussNormalization import BatchPitGaussNorm1d
from torch import Tensor, cuda, device
from torch.nn import BatchNorm1d



dev: device = device('cuda' if cuda.is_available() else 'cpu')



class BatchPitGaussNorm1d_test(unittest.TestCase):
    def test_mean_var_track(self):
        torch.manual_seed(1337)
        num_samples, num_feats = 100, 50

        x: Tensor = torch.rand(size=(num_samples, num_feats)).to(device=dev)
        layer = BatchPitGaussNorm1d(num_feats=num_feats, dev=dev)

        x = layer(x)

        r_mean = layer.bn1d.get_buffer('running_mean').cpu().numpy()
        r_var = layer.bn1d.get_buffer('running_var').cpu().numpy()
        zeros = np.zeros_like(r_mean)

        assert not np.allclose(a=r_mean, b=zeros, rtol=1e-10, atol=1e-15)
        assert not np.allclose(a=r_var, b=zeros, rtol=1e-10, atol=1e-15)


    def test_init_throws(self):
        lambdas = [
            lambda: BatchPitGaussNorm1d(dev=dev, bn1d=None, num_feats='foo'),
            lambda: BatchPitGaussNorm1d(dev=dev, bn1d=BatchNorm1d(num_features=10, affine=False, track_running_stats=True)),
            lambda: BatchPitGaussNorm1d(dev=dev, bn1d=BatchNorm1d(num_features=10, affine=True, track_running_stats=False))]
        
        for l in lambdas:
            threw = False
            try:
                l()
            except AssertionError:
                threw = True
            
            assert threw
