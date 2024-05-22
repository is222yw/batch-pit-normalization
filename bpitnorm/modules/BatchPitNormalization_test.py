import torch
import unittest


from bpitnorm.modules.BatchPitNormalization import BatchPitNorm1d
from numpy import abs, ndarray, quantile, sum
from typing import Callable
from torch import Tensor, cuda, device, nn
from torch.optim import Adam
from torch.autograd import Variable

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils._bunch import Bunch
from dataclasses import dataclass


@dataclass
class DataBunch:
    train_X: pd.DataFrame
    train_y: pd.DataFrame
    scaler_X: StandardScaler
    valid_X: pd.DataFrame = None
    valid_y: pd.DataFrame = None
    scaler_Y: StandardScaler = None

from math import exp, isnan, isinf


dev: device = device('cuda' if cuda.is_available() else 'cpu')



def sigmoid(x: float) -> float:
    return 1.0 / (1 + exp(-x))


def get_iris_dataset(test_size: float=None, seed: int=0xbeef) -> DataBunch:
    temp: Bunch = load_iris(as_frame=True)
    X: pd.DataFrame
    y: pd.DataFrame
    X, y = temp.data, temp.target

    if test_size is None:
        scaler = StandardScaler().fit(X=X)
        X = pd.DataFrame(data=scaler.transform(X=X), columns=X.columns)
        # There is validation data and for Iris, Y is one-hot (no scaling of Y)
        return DataBunch(train_X=X, train_y=y, scaler_X=scaler)
    
    assert isinstance(test_size, float) and not isnan(test_size) \
        and not isinf(test_size) and test_size > 0.01 and test_size < 0.99
    
    train_X, valid_X, train_y, valid_y = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    
    scaler_X = StandardScaler().fit(X=train_X)
    train_X = pd.DataFrame(data=scaler_X.transform(X=train_X), columns=train_X.columns)
    valid_X = pd.DataFrame(data=scaler_X.transform(X=valid_X), columns=valid_X.columns)

    # Again, there is no scaling for Y
    return DataBunch(train_X=train_X, valid_X=valid_X, train_y=train_y, valid_y=valid_y, scaler_X=scaler_X)


class BatchPitNorm1d_test(unittest.TestCase):
    def test_normalization(self):
        torch.manual_seed(1337)
        num_feats, num_samples, num_pit_samples = 25, 100, 500

        x: Tensor = (100.0 * torch.rand(size=(num_samples, num_feats)) - 50.0).to(device=dev)
        cdf_data: Tensor = (50.0 * torch.randn(size=(num_pit_samples, num_feats))).to(device=dev)

        bpn1d = BatchPitNorm1d(num_features=num_feats, num_pit_samples=num_pit_samples, take_num_samples_when_full=0, dev=dev, normal_backtransform=False, trainable_bandwidths=False, bw_select='RuleOfThumb')
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
                # The tolerance, unfortunately, has to be quite large because we seem to
                # get relatively large differences in the Rule-of-Thumb bandwidth calculations
                # between torch and numpy.
                if abs(val_actually - val_expected) > 2e-3:
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
    

    def test_toy_example(self):
        """
        A toy-test in the sense that we make an ordinary network that has BPitNorm not only
        as the first layer but also somewhere in between.
        """
        data_bunch = get_iris_dataset(test_size=0.4, seed=1)

        model = nn.Sequential(
            BatchPitNorm1d(num_features=4, num_pit_samples=100, take_num_samples_when_full=1, dev=dev, bw_select='ISJ'),
            nn.Linear(in_features=4, out_features=15, device=dev),
            nn.Dropout(p=0.3),
            nn.SiLU(inplace=True),
            BatchPitNorm1d(num_features=15, num_pit_samples=100, take_num_samples_when_full=0, dev=dev, bw_select='Silverman'),
            nn.Linear(in_features=15, out_features=3, device=dev),
            nn.SiLU(inplace=True),
            nn.Softmax(dim=1)).to(device=dev)
        
        optim = Adam(params=model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        X_train = Variable(torch.from_numpy(data_bunch.train_X.to_numpy())).float().to(device=dev)
        y_train = Variable(torch.from_numpy(data_bunch.train_y.to_numpy())).long().to(device=dev)
        X_test  = Variable(torch.from_numpy(data_bunch.valid_X.to_numpy())).float().to(device=dev)
        y_test  = Variable(torch.from_numpy(data_bunch.valid_y.to_numpy())).long().to(device=dev)

        EPOCHS  = 50 if dev.type == 'cuda' else 10
        loss_list = np.zeros((EPOCHS,))
        accuracy_list = np.zeros((EPOCHS,))

        torch.manual_seed(0)

        for epoch in range(EPOCHS):
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            loss_list[epoch] = loss.item()
            
            # Zero gradients
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            with torch.no_grad():
                y_pred = model(X_test)
                correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
                accuracy_list[epoch] = correct.mean()
        
        assert accuracy_list[0] < accuracy_list[-1], f'{accuracy_list[0]} - {accuracy_list[-1]}'
        print((accuracy_list[0], accuracy_list[-1]))
