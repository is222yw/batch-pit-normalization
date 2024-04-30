import torch
import unittest

from bpitnorm.modules.BatchPitNormalization import BatchPitNorm1d
from numpy import abs, ndarray, quantile, sum
from typing import Callable
from torch import Tensor, cuda, device

from math import exp


dev: device = device('cuda' if cuda.is_available() else 'cpu')



def sigmoid(x: float) -> float:
    return 1.0 / (1 + exp(-x))


class BatchPitNorm1d_test(unittest.TestCase):
    def test_normalization(self):
        torch.manual_seed(1337)
        num_feats, num_samples, num_pit_samples = 25, 100, 500

        x: Tensor = torch.rand(size=(num_samples, num_feats)).to(device=dev)
        cdf_data: Tensor = torch.rand(size=(num_pit_samples, num_feats)).to(device=dev)

        bpn1d = BatchPitNorm1d(num_features=num_feats, num_pit_samples=num_pit_samples, take_num_samples_when_full=0, dev=dev, normal_backtransform=False, trainable_bandwidths=False)
        bpn1d.fill(data=cdf_data)
        bpn1d.eval() # Important, so that the model stops filling from the batches.

        vmap_result = bpn1d.forward(x=x)
        x = x.cpu().numpy()

        vmap_result: ndarray = vmap_result.detach().cpu().numpy()
        cdf_data: ndarray = cdf_data.cpu().numpy()
        from scipy.stats.distributions import norm

        normal_dist = norm(loc=0.0, scale=1.0)

        std_normal_cdf = normal_dist.cdf    
        std_normal_ppf = normal_dist.ppf

        def kde_cdf(data: ndarray, bw: float = None) -> Callable[[float], float]:
            if bw is None:
                q25 = quantile(a=data, q=.25)
                q75 = quantile(a=data, q=.75)
                IQR = q75 - q25
                bw = 0.9 * min(data.std(), IQR / 1.34) * float(data.size)**(-.2)
            return lambda x_val: 1.0 / data.size * sum(std_normal_cdf((x_val - data) / bw))
        
        _min = 9e-8
        _max = 1.0 - _min
        
        # Let's check this feature-wise.
        for feat_idx in range(num_feats):
            bw: float = None
            if bpn1d.trainable_bandwidths:
                bw = sigmoid(bpn1d.bw[0, feat_idx].item())
            cdf = kde_cdf(data=cdf_data[:, feat_idx], bw=bw)

            for sample_idx in range(num_samples):
                val_expected = cdf(x[sample_idx, feat_idx])
                if bpn1d.normal_backtransform:
                    val_expected = std_normal_ppf(min(_max, max(_min, val_expected)))
                else:
                    val_expected -= 0.5
                val_actually = vmap_result[sample_idx, feat_idx]
                if abs(val_actually - val_expected) > 4e-5: # 1 / 5000
                    raise Exception((sample_idx, feat_idx, abs(val_actually - val_expected)))

    def test_fill(self):
        bpn1d = BatchPitNorm1d(num_features=10, num_pit_samples=100, take_num_samples_when_full=10, dev=dev)

        assert bpn1d.size == 0 and bpn1d.capacity_left == 100
        batch = torch.rand(size=(32,10)).to(device=dev)
        bpn1d.forward(batch)
        assert bpn1d.size == 32 and bpn1d.capacity_left == 68
        bpn1d.eval()
        bpn1d.forward(batch) # should not fill this time in eval mode
        assert bpn1d.size == 32 and bpn1d.capacity_left == 68

        bpn1d.train(True) # Must be set to allow filling
        for _ in range(1000):
            bpn1d.fill(batch)
        
        assert bpn1d.size == 100
