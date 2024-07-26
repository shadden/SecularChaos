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


def power_of_linear_combination_to_series(coeffs,pwr,conj,eccentricity):
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
    eccentricity : bool
        If True, create terms of poisson series for eccentricity varialbes,
        otherwise, create term for inclinations variables.

    Returns
    -------
    PoissonSeries
        Expansion of the multinomial.
    """
    dim = len(coeffs)
    twoNzeros = np.zeros(2*dim,dtype=int)
    Nzeros = np.zeros(dim,dtype = int)
    if conj:
        if eccentricity:
            to_term = lambda val,exponent: PSTerm(val,twoNzeros,np.concatenate((exponent,Nzeros)),[],[])
        else:
            to_term = lambda val,exponent: PSTerm(val,twoNzeros,np.concatenate((Nzeros,exponent)),[],[])

    else:
        if eccentricity:
            to_term = lambda val,exponent: PSTerm(val,np.concatenate((exponent,Nzeros)),twoNzeros,[],[])
        else:
            to_term = lambda val,exponent: PSTerm(val,np.concatenate((Nzeros,exponent)),twoNzeros,[],[])
    terms = []
    for exponent in list_multinomial_exponents(pwr,dim):
        c = multinomial_coefficient(pwr,exponent)
        val = c*np.prod(np.power(coeffs,exponent))
        terms.append(to_term(val,exponent))
    series = PoissonSeries.from_PSTerms(terms)
    return series

def linear_transform_secular_poisson_series_term(term,Npl,Te,Ti):
    """
    Compute a Poisson series representation of a single Poisson series term
    after a linear transformation of complex eccentricity and inclination
    variables.

    Parameters
    ----------
    term : celmech.poisson_series.PSTerm
        Term to convert.
    Npl : int
        Number of planets for which the secular Poisson series is being
        constructed.
    Te : ndarray
        Matrix diagonalizing the complex eccentricity dynamics.
    Ti : ndarray
        Matrix diagonalizing the complex inclination dynamics.

    Returns
    -------
    celmech.poisson_series.PoissonSeries
        Resulting Poisson series after applying linear transformation to
        individual term.
    """
    t0 = PSTerm(term.C,np.zeros(2*Npl),np.zeros(2*Npl),[],[])
    series = PoissonSeries.from_PSTerms([t0])
    ## Un-barred terms ##
    # eccentricity 
    for i,pwr in enumerate(term.k[:Npl]):
        if pwr>0:
            x = power_of_linear_combination_to_series(Te[i],pwr,False,True)
            series = series * x
    # inclination
    for i,pwr in enumerate(term.k[Npl:]):
        if pwr>0:
            x = power_of_linear_combination_to_series(Ti[i],pwr,False,False)
            series = series * x
    ## Barred terms ##
    # eccentricity #
    for i,pwr in enumerate(term.kbar[:Npl]):
        if pwr>0:
            x = power_of_linear_combination_to_series(Te[i],pwr,True,True)
            series = series * x
    # inclination
    for i,pwr in enumerate(term.kbar[Npl:]):
        if pwr>0:
            x = power_of_linear_combination_to_series(Ti[i],pwr,True,False)
            series = series * x
    return series

def linear_transform_secular_poisson_series(pseries,Te,Ti):
    """
    Given a Poisson series in complex eccentricity and inclination variables,
    compute a new Poisson series resulting from a linear transformations of
    these complex variables.

    Parameters
    ----------
    pseries : celmech.poisson_series.PoissonSeries
        The original PoissonSeries in complex eccentricity and inclination
        variables.
    Te : ndarray
        The matrix defining the linear transformation of complex eccentricity
        variables as :math:`x_i = [T_e]_{ij}u_j`
    Ti : ndarray
        The matrix defining the linear transformation of complex inclination
        variables as :math:`y_i = [T_e]_{ij}v_j`

    Returns
    -------
    celmech.poisson_series.PoissonSeries
        A Poisson series in the new, transformed variables.
    """
    Npl = pseries.N//2
    tr_series = PoissonSeries(pseries.N,0)
    for term in pseries.terms:
        tr_series+=linear_transform_secular_poisson_series_term(term,Npl,Te,Ti)
    return tr_series