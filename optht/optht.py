"""Optimal hard threshold for matrix denoising."""

import logging

import numpy as np
from scipy import integrate

# Create logger
log = logging.getLogger(__name__)


def optht(beta, sv=None, sigma=None):
    """Compute optimal hard threshold for singular values.

    Off-the-shelf method for determining the optimal singular value truncation
    (hard threshold) for matrix denoising.

    The method gives the optimal location both in the case of the known or
    unknown noise level.

    Parameters
    ----------
    beta : scalar or array_like
        Scalar determining the aspect ratio of a matrix, i.e., `beta = m/n',
        where `m >= n'.
        Instead the input matrix can be provided and the aspect ratio is
        determined automatically.

    sv : array_like
        The singular values for the given input matrix.

    sigma : real, optional
        Noise level if known.

    Returns
    -------
    k : int
        Optimal target rank.

    Notes
    -----
    Code is adapted from Matan Gavish and David Donoho, see [1]_.

    References
    ----------
    .. [1] Gavish, Matan, and David L. Donoho.
       "The optimal hard threshold for singular values is 4/sqrt(3)"
        IEEE Transactions on Information Theory 60.8 (2014): 5040-5053.
        http://arxiv.org/abs/1305.5870
    """
    # Compute aspect ratio of the input matrix
    if isinstance(beta, np.ndarray):
        m = min(beta.shape)
        n = max(beta.shape)
        beta = m / n

    if beta < 0 or beta > 1:
        raise ValueError('beta must be in (0,1].')

    if sigma is None:
        # Sigma unknown
        log.info('Sigma unknown.')

        coef = _optimal_SVHT_coef_sigma_unknown(beta)
        log.info(f'Approximated coefficient w(beta): {coef}')

        coef = (_optimal_SVHT_coef_sigma_known(beta)
                / np.sqrt(_MedianMarcenkoPastur(beta)))
        log.info(f'Optimal coefficient w(beta): {coef}')

        if sv is not None:
            # Compute cutoff
            cutoff = coef * np.median(sv)
            log.info(f'Cutoff value: {cutoff}')
            # Compute rank
            k = np.max(np.where(sv > cutoff)) + 1
            log.info(f'Target rank: {k}')
            return k
    else:
        # Sigma known
        log.info('Sigma known.')

        coef = _optimal_SVHT_coef_sigma_known(beta)
        log.info(f'w(beta) value: {coef}')

        if sv is not None:
            # Compute cutoff
            cutoff = coef * np.sqrt(len(sv)) * sigma
            log.info('Cutoff value: {cutoff}')
            # Compute rank
            k = np.max(np.where(sv > cutoff)) + 1
            log.info('Target rank: {k}')
            return k
    return coef


def _optimal_SVHT_coef_sigma_known(beta):
    """Implement Equation (11)."""
    return np.sqrt(2 * (beta + 1) + (8 * beta)
                   / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1)))


def _optimal_SVHT_coef_sigma_unknown(beta):
    """Implement Equation (5)."""
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


def _MarPas(x, topSpec, botSpec, beta):
    """Implement Marcenko-Pastur distribution."""
    if (topSpec - x) * (x - botSpec) > 0:
        return np.sqrt((topSpec - x) *
                       (x - botSpec)) / (beta * x) / (2 * np.pi)
    else:
        return 0


def _MedianMarcenkoPastur(beta):
    """Compute median of Marcenko-Pastur distribution."""

    botSpec = lobnd = (1 - np.sqrt(beta))**2
    topSpec = hibnd = (1 + np.sqrt(beta))**2
    change = 1

    while change & ((hibnd - lobnd) > .001):
        change = 0
        x = np.linspace(lobnd, hibnd, 10)
        y = np.zeros_like(x)
        for i in range(len(x)):
            yi, err = integrate.quad(
                _MarPas,
                a=x[i],
                b=topSpec,
                args=(topSpec, botSpec, beta),
            )
            y[i] = 1.0 - yi

        if np.any(y < 0.5):
            lobnd = np.max(x[y < 0.5])
            change = 1

        if np.any(y > 0.5):
            hibnd = np.min(x[y > 0.5])
            change = 1

    return (hibnd + lobnd) / 2.
