import torch
from torch import nn, Tensor, empty, fill, device, nan
from typing import Callable, Self
from bpitnorm.modules.Common import standard_normal_cdf, standard_normal_ppf



class BatchPitNorm1d(nn.Module):
    """
    Batch Probability Integral Transform Normalization (or, "Batch-Pit Normalization").
    Estimates a Gaussian kernel density per each feature based on the observed data for
    each feature. Then uses the CDF of the density to transform each feature such that
    it has a uniform distribution. This may then be further transformed into another
    distribution (built-in support for normal).

    This layer does not require the data to be normalized in any way. Similar to ordinary
    Batch Normalization, it will correct covariate shift. Beyond that, it will modify the
    distribution of the data flowing through to be, e.g., perfectly uniform or normal.

    Author: Sebastian Hönel
    """
    def __init__(self, num_features: int, num_pit_samples: int, take_num_samples_when_full: int, dev: device, normal_backtransform: bool = True, trainable_bandwidths: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert num_pit_samples > 0, 'Require at least one sample for PIT normalization.'
        assert take_num_samples_when_full >= 0

        self.num_pit_samples = num_pit_samples
        self.take_num_samples_when_full = take_num_samples_when_full
        self.num_features = num_features

        self.trainable_bandwidths = trainable_bandwidths
        if trainable_bandwidths:
            self.bw = torch.nn.Parameter(data=torch.rand(size=(1, self.num_features,)), requires_grad=True).to(device=dev)
        else:
            self.bw = fill(input=torch.empty(size=(1, self.num_features,)), value=nan)

        self.size = 0 # Here we keep track of by how much the values are filled

        values = empty(size=(self.num_pit_samples, self.num_features), device=dev)
        values = fill(input=values, value=nan)
        self.register_buffer(name='cdf_data', persistent=True, tensor=values)

        self.normal_backtransform = normal_backtransform
    
    @property
    def values(self) -> Tensor:
        return self.get_buffer(target='cdf_data')
    
    @property
    def is_full(self) -> bool:
        return self.size == self.num_pit_samples
    
    @property
    def capacity_left(self) -> int:
        return self.num_pit_samples - self.size
    
    def fill(self, data: Tensor) -> Self:
        assert self.training, 'Must be in training mode to allow filling.'
        batch_size = data.shape[0]
        cap_left = self.capacity_left

        if cap_left >= batch_size:
            # Full take, store the entire batch's data in our values.
            self.values[self.size:(self.size + batch_size)] = data
            self.size += batch_size
        elif cap_left > 0:
            # Take the first elements, then call this method again with remainder of batch.
            self.values[self.size:self.num_pit_samples] = data[0:cap_left]
            self.size += cap_left
            # Choose accordingly for the remaining values:
            self.fill(data=data[cap_left:batch_size])
        else:
            if self.take_num_samples_when_full == 0:
                return self
            # No capacity left.
            use_batch_indexes = torch.randperm(n=min(batch_size, self.take_num_samples_when_full))
            use_values_indexes = torch.randperm(n=self.num_pit_samples)[0:min(batch_size, self.take_num_samples_when_full)]
            self.values[use_values_indexes] = data[use_batch_indexes]

        return self
    

    def make_cdf(self, data: Tensor, bw: float) -> Callable[[float], float]:
        num_samples = data.shape[0]
        if not self.trainable_bandwidths:
            q25 = torch.quantile(input=data, q=.25, dim=0)
            q75 = torch.quantile(input=data, q=.75, dim=0)
            IQR = q75 - q25
            bw = 0.9 * torch.min(data.std(), IQR / 1.34) * float(num_samples)**(-.2)
        else:
            bw = torch.sigmoid(bw) # Ensure it's positive.
        return lambda use_x: 1.0 / num_samples * torch.sum(standard_normal_cdf((use_x - data) / bw))
    

    def process_merged(self, all_data: Tensor, bandwidths: Tensor) -> Tensor:
        size = self.size
        data_cdf = all_data[0:size]
        data_sample = all_data[size:(size + all_data.shape[0])]

        cdf = self.make_cdf(data=data_cdf, bw=bandwidths)
        vcdf = torch.vmap(cdf, in_dims=0, out_dims=0)

        return vcdf(data_sample)
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        # First let's fill up the buffered values for the underlying CDFs.
        if self.training:
            self.fill(data=x)
        else:
            assert self.size > 0, 'Cannot compute forward pass without sample for the integral transform.'

        all_data = torch.vstack((self.values[0:self.size], x))
        assert all_data.shape[0] == self.size + batch_size
        vfunc = torch.vmap(self.process_merged, in_dims=1, out_dims=1)
        result = vfunc(all_data, self.bw)

        if self.normal_backtransform:
            result = standard_normal_ppf(x=result)
        else:
            result -= 0.5
        return result
