import numpy as np
from celmech.poisson_series import *

def list_multinomial_exponents(pwr,ndim):
    """
    List exponents appearing in the expansion of the multinomial 
    
    .. math:: 
    (x_1 + x_2 + ... + x_\mathrm{ndim})^pwr

    Parameters
    ----------
    pwr : int
        Exponent of multinomial
    ndim : int
        Number of variables in multinomial

    Returns
    -------
    list
        A list of lists of length `ndim` representing the exponents of each
        varaible :math:`x_i`
    """
    if ndim==1:
        return [[pwr],]
    else:
        x =[]
        for pow1 in range(0,pwr+1):
            subpows = list_multinomial_exponents(pwr-pow1,ndim-1)
            x+=[[pow1]+y for y in subpows]
        return x
    
from math import factorial
def multinomial_coefficient(p, ks):
    """Calculate multinomial coefficient for given p and ks"""
    num = factorial(p)
    denom = 1
    for k in ks:
        denom *= factorial(k)
    return num // denom


def power_of_linear_combination_to_series(coeffs,pwr,conj):
    """
    Return a Poisson series representation of the multinomial
    
    .. math::
        (c_1*x_1 + c_2*x_2 + ... + c_dim*x_dim)^pwr
    
    or its complex conjugate.

    Parameters
    ----------
    pwr : int
        Exponent
    coeffs : ndarray
        Array of coefficients
    conj : bool
        If True, evaluate complex conjugate of multinomial. Otherwise just
        compute the mulitnomial expansion.

    Returns
    -------
    PoissonSeries
        Expansion of the multinomial.
    """
    dim = len(coeffs)
    Nzeros = np.zeros(dim,dtype=int)
    if conj:
        to_term = lambda val,exponent: PSTerm(val,Nzeros,exponent,[],[])
    else:
        to_term = lambda val,exponent: PSTerm(val,exponent,Nzeros,[],[])
    terms = []
    for exponent in list_multinomial_exponents(pwr,dim):
        c = multinomial_coefficient(pwr,exponent)
        val = c*np.prod(np.power(coeffs,exponent))
        terms.append(to_term(val,exponent))
    series = PoissonSeries.from_PSTerms(terms)
    return series