"""
This code performs gaussian process regression given some set of data. It interacts with the CARPoolKernels classes to do this and tinygp.

The minimization routine is done with jax which we believe to be optimal. 
"""
import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as linalg
from src import CARPoolKernels

@jax.jit
@jax.value_and_grad
def loss(params, theta, surrogate_theta, Y):
    """
    Return the loss and gradient of the loss for gradient descent
    """
    cov = build_CARPoolCov(params, theta, surrogate_theta)
    
    # Compute liklihood
    alpha, scale_tril = decomp(cov, Y, jnp.exp(params['log_mean']))
    L = log_liklihood(scale_tril, alpha)

    return -L

@jax.jit
def build_CARPoolCov(params, theta, surrogate_theta, noise=0):
    N_theta     = len(theta)
    N_surrogates = len(surrogate_theta)

    # Build Kernels with current parameter values
    Vkernel = CARPoolKernels.VWKernel(jnp.exp(params["log_ampV"]), jnp.exp(params["log_scaleV"]))
    Wkernel = CARPoolKernels.VWKernel(jnp.exp(params["log_ampW"]), jnp.exp(params["log_scaleW"]))
    Xkernel =  CARPoolKernels.XKernel(jnp.exp(params["log_ampX"]), jnp.exp(params["log_scaleX"]), jnp.exp(params["log_deltaP"]))
    V       = Vkernel(theta, theta)
    W       = Wkernel(surrogate_theta, surrogate_theta)
    X       = Xkernel(theta, surrogate_theta)
    C       = jnp.block([[V, X],[X.T, W]])

    if noise is None:
        return C
    
    # Build the noise fluctutaions
    Mkernel =CARPoolKernels.EKernel(jnp.exp(params["log_scaleM"]))
    M       = Mkernel(theta, surrogate_theta) * jnp.exp(params["log_jitterV"]) * jnp.exp(params["log_jitterW"]) * np.eye(N_theta, N_surrogates)
    IsigmaV = jnp.exp(params["log_jitterV"])**2 * jnp.eye(N_theta)
    IsigmaW = jnp.exp(params["log_jitterW"])**2 * jnp.eye(N_surrogates)


    noise = jnp.block([[IsigmaV, M], [M.T, IsigmaW]])
    cov   = C + noise
    return cov

def predict(Y, cov, cov_new, mu_y):
    """
    mean = Ks C^{-1} (Y-\mu_Y) + \mu_Y
    cov = Kss - Ks C^{-1}Ks^T

    Ks = covariance of new thetas with old thetas,
    Kss= covariance of new thetas
    C  = covariance from GP
    x  = Value of params
    gp_mean = mean function

    returns mean and cov
    """
    ltn = cov_new.shape[0] - cov.shape[0]
    mean = cov_new[:ltn, ltn:] @ np.linalg.inv(cov)@(Y - mu_y) + mu_y
    cov = cov_new[:ltn, :ltn] - \
        cov_new[:ltn, ltn:] @ np.linalg.inv(cov) @  cov_new[:ltn, ltn:].T
    return mean, cov

@jax.jit
def decomp(cov, Q, mean):
    scale_tril = linalg.cholesky(cov, lower=True)
    alpha = linalg.solve_triangular(scale_tril, Q-mean, lower=True)
    return alpha, scale_tril


@jax.jit
def invdecomp(cov, Q, mean):
    scale_tril = linalg.cholesky(cov, lower=True)
    alpha = linalg.solve_triangular(scale_tril, Q-mean, lower=True, trans=1)
    return alpha, scale_tril


@jax.jit
def log_liklihood(scale_tril, alpha):
    return -0.5 * jnp.sum(jnp.square(alpha)) - \
        jnp.sum(jnp.log(jnp.diag(scale_tril))) + \
        0.5 * scale_tril.shape[0] * jnp.log(2 * jnp.pi)
