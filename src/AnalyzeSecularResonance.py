import numpy as np


def compute_sturm_sequence(p):
    """ Compute the Sturm sequence for polynomial p. """
    derivative = np.polyder(p)
    sturm_sequence = [np.array(p),derivative]
    while np.any(derivative):
        # Perform polynomial division
        quotient, remainder = np.polydiv(sturm_sequence[-2], sturm_sequence[-1])
        # Append the negative remainder
        sturm_sequence.append(-remainder)
        # Update the derivative
        derivative = np.polyder(sturm_sequence[-1])
    # Clean up small coefficients and remove zero polynomials
    sturm_sequence = [np.trim_zeros(poly, 'f') for poly in sturm_sequence if np.any(poly)]
    return sturm_sequence

def evaluate_polynomial(p, x):
    """ Evaluate polynomial p at point x. """
    return np.polyval(p, x)

def count_sign_changes(values):
    """ Count the number of sign changes in a list of values, ignoring zeros. """
    non_zero_values = [v for v in values if v != 0]
    sign_changes = sum(
        1 for i in range(len(non_zero_values) - 1)
        if np.sign(non_zero_values[i]) != np.sign(non_zero_values[i+1])
    )
    return sign_changes

def count_real_roots(coefficients, a, b):
    """ Determine the number of real roots of the polynomial within the interval [a, b]. """
    sturm_sequence = compute_sturm_sequence(coefficients)
    # Evaluate the Sturm sequence at the endpoints of the interval
    values_at_a = [evaluate_polynomial(poly, a) for poly in sturm_sequence]
    values_at_b = [evaluate_polynomial(poly, b) for poly in sturm_sequence]
    # Count sign changes at the endpoints
    sign_changes_a = count_sign_changes(values_at_a)
    sign_changes_b = count_sign_changes(values_at_b)
    # The number of real roots in the interval [a, b]
    return sign_changes_a - sign_changes_b

import sympy as sp
from collections import defaultdict
def resonance_terms_to_poly(kres,Hres,omega,Amtrx,P0):
    p = sp.symbols("p",real=True)
    P = np.array([x + y * p for x,y in zip(P0,kres)])
    mu_sq = sp.Poly(np.prod(P**np.mod(kres,2)))
    non_zero_indices = np.nonzero(kres)[0]
    dmu_dp_times_mu=sp.Poly(0.,p)
    for i in non_zero_indices:
        dmu_dp_times_mu+=0.5*kres[i]*np.prod([P[j] for j in non_zero_indices if j!=i])
    Fn = defaultdict(sp.Poly(0.,p))
    dFn_dp = defaultdict(sp.Poly(0.,p))
    eyeN = np.eye(Hres.N,dtype=int)
    for term in Hres.terms:
        k1 = term.k - term.kbar
        n = k1[non_zero_indices[0]] // kres[non_zero_indices[0]]
        pwrs = (term.k + term.kbar - np.mod(k1,2)) // 2
        Fn[n] += sp.Poly(term.C * np.prod(P**pwrs),p)
        dFn_dp[n] += np.sum([term.C * kres[i] * pwrs[i] * np.prod(P**(pwrs-eyeN[i])) for i in non_zero_indices])

    Minv = kres@Amtrx@kres
    Omega = kres@(omega + Amtrx@P0)
    keys = np.array(list(Fn.keys()))
    keys = keys[keys>0]
    rhs = 0
    lhs = sp.Poly(p*Minv + Omega,p)
    for n in keys:
        assert n<3, "need to implement polynomials with n>2!"
        if np.mod(n,2):
            rhs += 2 * dmu_dp_times_mu * Fn[n] + 2 * dFn_dp[n] * mu_sq
        else:
            lhs += 2 * dFn_dp[n]
    if rhs==0:
        return lhs
    return lhs * lhs * mu_sq - rhs * rhs
def get_allowed_p_range(kres,P0):
    non_zero_indices = np.nonzero(kres)[0]
    pmin,pmax = -np.inf,np.inf
    for nix in non_zero_indices:
        P0i = P0[nix]
        ki = kres[nix]
        pcrit = - P0i/ki
        if pcrit>0 and pcrit<pmax:
            pmax = pcrit
        elif pcrit<0 and pcrit > pmin:
            pmin = pcrit
    return (pmin,pmax)


def Hres_to_exprn(kres,Hres,p,phi,P0):
    cvar_symbols = Hres.cvar_symbols
    non_zero_indices = np.nonzero(kres)[0]
    P = np.array([a + b * p for a,b in zip(P0,kres)])
    exprn = 0.
    for term in Hres.terms:
        k1 = term.k - term.kbar
        n = k1[non_zero_indices[0]] // kres[non_zero_indices[0]]
        pwrs = (term.k + term.kbar)/sp.S(2)
        exprn+=term.C * np.prod(P**pwrs) * sp.cos(n*phi)
    return exprn


import sympy as sp



def replace_small_floats(expr, threshold):
    """
    Replace all floats in a SymPy expression that are smaller than the given threshold with zero.

    Args:
    expr (sympy.Expr): The SymPy expression to modify.
    threshold (float): The threshold value below which floats are replaced with zero.

    Returns:
    sympy.Expr: The modified expression.
    """
    # Define the replacement function that checks and replaces small floats
    def zero_small_floats(x):
        if isinstance(x, sp.Float) and abs(x) < threshold:
            return sp.Float(0)
        return x

    # Replace using a lambda to apply the function to each Float in the expression
    return expr.replace(lambda x: isinstance(x, sp.Float), zero_small_floats)
