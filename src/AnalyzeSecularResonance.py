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
