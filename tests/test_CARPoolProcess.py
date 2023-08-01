import pytest
import numpy as np
import src.CARPoolKernels as CARPoolKernels

def get_params(ndim):
    params = {"scale":np.ones(ndim),
              "amp":np.ones(ndim),
              "deltaP":np.ones(ndim)
    }
    return params

def test_loss():
    params = get_params(1)
    theta = np.random.normal(0,1,10)