from __future__ import division
import numpy as np
import scipy.integrate as si
from numpy.testing import assert_raises


def optht(beta, sv=None, sigma=None, trace=True):
    """
    Optimal hard threshold for singular values

    Off-the-shelf method for determining the optimal singular value truncation
    (hard threshold) for matrix denoising.    
    
    The method gives the optimal location both in the case of the konwn or unknown
    noise level.


    Parameters
    ----------
    beta : scalar or array_like
        Scalar determining the aspect ratio of a matrix, i.e., `beta = m/n', where
        `m >= n'.
        Instead the input matrix can be provided and the aspect ratio is determined
        automatically. 
        
    sv : array_like  
        The singular values for the given input matrix.  
    
    sigma : real, optional
        Noise level if known.
    
    trace : bool `{True, False}`
        Print results.         

    Returns
    -------
    k : int
        Optimal target rank.
    

    Notes
    -----
    Code is adapted from Matan Gavish and David Donoho, see [1].
       
    References
    ----------
    [1] Gavish, Matan, and David L. Donoho. 
    "The optimal hard threshold for singular values is 4/sqrt(3)" 
    IEEE Transactions on Information Theory 60.8 (2014): 5040-5053.    
    http://arxiv.org/abs/1305.5870
    
    Examples
    --------


    """

    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2016>                               ***
    #***                       License: BSD 3 clause                       ***
    #*************************************************************************
          
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # compute aspect ratio of the input matrix
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if isinstance(beta, (np.ndarray)):
        m = min(beta.shape)
        n = max(beta.shape)
        beta = m/n
    
    if beta < 0 or beta > 1:
        raise ValueError('beta must be in (0,1].')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sigma unknown
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
    if sigma is None:
        if trace==True: print('Sigma unknown:')
        if trace==True: print('*************')

        coef = optimal_SVHT_coef_sigma_unknown(beta)
        if trace==True: print('approximated coefficent w(beta): ', coef)
            
            
        coef = optimal_SVHT_coef_sigma_known(beta) / np.sqrt(MedianMarcenkoPastur(beta))
        if trace==True: print('optimal coefficent w(beta): ', coef) 


        if sv is not None:
            cutoff = coef * np.median(sv)
            if trace==True: print('cutoff value: ', cutoff)

            k = np.max( np.where( sv>cutoff ) ) + 1
            if trace==True: print('target rank: ', k)

            return k
    # End else   

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sigma known
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
    else:
        if trace==True: print('Sigma known:')
        if trace==True: print('*************')        
        
        coef = optimal_SVHT_coef_sigma_known(beta)
        if trace==True: print('w(beta) value: ', coef) 

        
        if sv is not None:
            cutoff = coef * np.sqrt(len(sv)) * sigma
            if trace==True: print('cutoff value: ', cutoff)
            
            k = np.max( np.where( sv>cutoff ) ) + 1
            if trace==True: print('target rank: ', k)
            
            return k 
    # End else        
    return coef
        

# Equation (11)
def optimal_SVHT_coef_sigma_known(beta):
    return np.sqrt(2 * (beta + 1) + (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta +1)))

# Equation (5)
def optimal_SVHT_coef_sigma_unknown(beta):
    return 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43


# Marcenko-Pastur distribution
def MarPas(x, topSpec, botSpec, beta):
    if (topSpec-x)*(x-botSpec) > 0:
        return np.sqrt((topSpec-x) * (x-botSpec)) / (beta * x) / (2 * np.pi)
    else:
        return 0


def MedianMarcenkoPastur(beta):
    botSpec = lobnd = (1 - np.sqrt(beta))**2
    topSpec = hibnd = (1 + np.sqrt(beta))**2  
    change = 1

    while(change & ((hibnd-lobnd) > .001 ) ):
        change = 0
        x = np.linspace(lobnd, hibnd, 10)
        y = np.zeros_like(x)
        for i in range(len(x)):
            yi, err = si.quad(MarPas, a=x[i], b=topSpec, args=(topSpec, botSpec, beta))
            y[i] = 1-yi

        if np.any( y < 0.5 ):
            lobnd = np.max( x[y < 0.5] )
            change = 1
            
        if np.any( y > 0.5 ):
            hibnd = np.min( x[y > 0.5] )
            change = 1
            
    return (hibnd+lobnd) / 2.







