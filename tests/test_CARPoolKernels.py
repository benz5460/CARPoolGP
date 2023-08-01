import pytest
import numpy as np
import src.CARPoolKernels as CARPoolKernels

def get_params(ndim):
    params = {"scale":np.ones(ndim),
              "amp":np.ones(ndim),
              "deltaP":np.ones(ndim)
    }
    return params
    
def test_VWKernel_1D():
    theta = np.random.normal(0, 1, 10)
    params = get_params(1)
    kernel = CARPoolKernels.VWKernel(params["amp"], params["scale"])
    cov = kernel(theta, theta)
    assert np.shape(cov) == (10,10)
    
def test_VWKernel_ND():
    theta = np.random.normal(0, 1, (10, 23))
    params = get_params(23)
    kernel = CARPoolKernels.VWKernel(params["amp"], params["scale"])
    cov = kernel(theta, theta)
    assert np.shape(cov) == (10,10)
    
def test_XKernel_1D():
    theta = np.random.normal(0, 1, 10)
    params = get_params(1)
    kernel = CARPoolKernels.XKernel(params["amp"], params["scale"], params["deltaP"])
    cov = kernel(theta, theta)
    assert np.shape(cov) == (10,10)
    
def test_XKernel_ND():
    theta = np.random.normal(0, 1, (10, 23))
    params = get_params(23)
    kernel = CARPoolKernels.XKernel(params["amp"], params["scale"], params["deltaP"])
    cov = kernel(theta, theta)
    assert np.shape(cov) == (10,10)
    
def test_EKernel_1D():
    theta = np.random.normal(0, 1, 10)
    params = get_params(1)
    kernel = CARPoolKernels.EKernel(params["amp"], params["scale"])
    cov = kernel(theta, theta)
    assert np.shape(cov) == (10,10)
    
def test_EKernel_ND():
    theta = np.random.normal(0, 1, (10, 23))
    params = get_params(23)
    kernel = CARPoolKernels.EKernel(params["amp"], params["scale"])
    cov = kernel(theta, theta)
    assert np.shape(cov) == (10,10)
    
    