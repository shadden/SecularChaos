from fractions import Fraction
from celmech.disturbing_function import list_resonance_terms
import numpy as np

from math import gcd

def farey_sequence(order):
    """Generate the Farey sequence of order `order`, returning fractions as tuples."""
    fractions = []  # List to store the fractions
    
    # Loop through potential numerators and denominators
    for denominator in range(1, order + 1):
        for numerator in range(0, denominator + 1):
            # Include the fraction if the greatest common divisor is 1 (i.e., they are co-prime)
            if gcd(numerator, denominator) == 1:
                fractions.append((numerator, denominator))
    
    # Sort the fractions by their numerical value
    fractions.sort(key=lambda x: x[0] / x[1])
    
    return fractions

def find_bracketing_fractions(value, order):
    """Find two fractions in the Farey sequence of specified order that bracket a given float `value`."""
    if not (0 <= value <= 1):
        raise ValueError("Value must be between 0 and 1.")
    
    # Get the Farey sequence
    fractions = farey_sequence(order)
    
    # Initialize variables to find bracketing fractions
    lower_fraction = (0, 1)  # Start with 0/1 as the initial lower bound
    upper_fraction = (1, 1)  # Start with 1/1 as the initial upper bound

    # Find the bracketing fractions
    for i in range(len(fractions) - 1):
        # Check if the current fraction is less than the value and the next is greater
        if fractions[i][0] / fractions[i][1] <= value < fractions[i+1][0] / fractions[i+1][1]:
            lower_fraction = fractions[i]
            upper_fraction = fractions[i+1]
            break

    return lower_fraction, upper_fraction

def generate_mmr_terms(J,maxorder,eccentricities=True, inclinations=True):
    r"""
    Generate a list of disturbing function resonance terms falling within the
    period ratio range :math:`\frac{J-1}{J}<P/P'<\frac{J}{J+1}` up to the
    specified order in inclinations and eccentricities.

    Parameters
    ----------
    J : int
        Define the bounding first-order MMRs of the period ratio range.
    maxorder : int
        Maximum order in inclination and eccentricity.
    eccentricities : bool, optional
        Whether to include eccentricity-type terms, by default True
    inclinations : bool, optional
        Whether to include inclination-type terms, by default True

    Returns
    -------
    list
        list of (k,nu) pairs specifying disturbing function terms.
    """
    res_terms_list = []
    Fk = farey_sequence(maxorder)
    for n,d in Fk:
        j = d * J + n
        k = d
        res_terms = list_resonance_terms(j,k,min_order=k,max_order=maxorder,eccentricities=eccentricities, inclinations=inclinations)
        res_terms_list += res_terms
    return res_terms_list

def resonance_crosses_Pratio_region(kvec,Jin,Jout):
    r"""
    Determine whether the three-body resonance defined by
    :math:`\mathbf{k}\cdot\mathbf{n}` intersects the region in period-ratio
    space defined by
    .. math::
        \begin{align}
        \frac{J_\mathrm{in} - 1}{J_\mathrm{in}} &< n_2/n_1 < \frac{J_\mathrm{in}}{J_\mathrm{in} + 1}
        \\
        \frac{J_\mathrm{out} - 1}{J_\mathrm{out}} &< n_3/n_2 < \frac{J_\mathrm{out}}{J_\mathrm{out} + 1}        
        \end{align}
    
    Parameters
    ----------
    kvec : ndarray
        Array  of 3BR integeger coefficeints
    Jin : float
        Defines the bounding resonances of the inner planet pair
    Jout : float
        Defines the boudning resonances of the outer planet pair

    Returns
    -------
    bool
        True if the specified 3BR crosses the period-ratio region.
    """
    xmin,xmax = (Jin-1)/Jin,Jin/(Jin+1)
    ymin,ymax = (Jout-1)/Jout,Jout/(Jout+1)
    k1,k2,k3 = kvec
    res_y2x = lambda y: k1/(k2 + k3 * y)
    xres_min,xres_max = np.sort([res_y2x(y) for y in (ymin,ymax)])
    return not (xres_max < xmin or xres_min > xmax)
from celmech.poisson_series import DFTerm_as_PSterms, PoissonSeries_to_GeneratingFunctionSeries
def generate_3br_terms(pham,i1,i2,i3,jk_pairs,max_order, inclinations=True,eccentricities=True):
    omega = pham.flow_func(*pham.state.values)[:pham.N_dof:3]
    omega = omega.reshape(-1)
    # Vector of freq. derivs, dn_i/dLambda_i
    domega = np.diag(pham.jacobian_func(*pham.state.values)[:pham.N_dof:3,pham.N_dof::3])
    for jin,kin,jout,kout,s in jk_pairs:
        # inner
        res_in = list_resonance_terms(jin,kin,min_order=kin,max_order=max_order,inclinations=inclinations,eccentricities=eccentricities)
        res_in = [(k,nu) for (k,nu) in res_in if k[0]==jin] # get rid of higher-order harmonics
        # outer
        res_out = list_resonance_terms(jout,kout,min_order=kout,max_order=max_order,inclinations=inclinations,eccentricities=eccentricities)
        res_out = [(k,nu) for (k,nu) in res_out if k[0]==jout] # get rid of higher-order harmonics
        for k,nu in mmr_list:
            h_in_terms+=DFTerm_as_PSterms(pham,i1,i2,k,nu,(0,0))


