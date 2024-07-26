import numpy as np
from celmech.poisson_series import *
from collections import defaultdict

def group_resonances_poisson_series(pseries):
    """
    Group terms in a Poisson series into separate, linearly independent
    resonances.

    Parameters
    ----------
    pseries : celmech.poisson_series.PoissonSeries
        Series to group terms for

    Returns
    -------
    dict
        Dictionary containing Poisson series objects associated with resonances.
        Resonances are indicated by the dictionary keys.
    """
    groups=defaultdict(lambda : PoissonSeries(pseries.N,pseries.M))
    resonances = []
    for term in pseries.terms:
        kvec = term.kbar - term.k
        if np.all(kvec==0):
            groups['secular'] += term.as_series()
        else:
            s = np.sign(kvec[kvec!=0][0])
            key = tuple(s * kvec // np.gcd.reduce(kvec))
            groups[key] += term.as_series()
    return groups

def mass_matrix_from_secular_terms(pseries):
    N = pseries.N
    Minv = np.zeros((N,N))
    eye = np.eye(N)
    for i in range(N):
        for j in range(i,N):
            keyvec = np.concatenate((eye[i],eye[i])) + np.concatenate((eye[j],eye[j]))
            Minv[i,j] = np.real(pseries[tuple(keyvec)])
    Minv += Minv.T
    return Minv
                