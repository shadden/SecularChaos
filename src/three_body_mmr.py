import rebound as rb
import celmech as cm
import sympy as sp
import numpy as np
from celmech.poisson_series import PoissonSeries, DFTerm_as_PSterms
from celmech.poisson_series import PoissonSeries_to_GeneratingFunctionSeries
from celmech.poisson_series import get_N_planet_poisson_series_symbols
from celmech.disturbing_function import list_resonance_terms

def term_order(k,nu):
    return np.sum(np.abs(k[2:])) + 2 * np.sum(nu)

def pair_mmr_lists(i1,i2,i3,jvec,sgn,inner_mmrs,outer_mmrs,max_order):
    """
    Pair combinations of disturbing function arguemnts that give rise to
    three-body MMR terms up to a given maximum order

    Parameters
    ----------
    i1 : int
        innermost planet index
    i2 : int
        middle planet index
    i3 : int
        outermost planet index
    jvec: ndarray
        1d array of coefficients multiplying desired three-body MMR
    sgn : int
        Integer ±1 depending on whether inner and outer MMR arguments
        are added or subtracted.
    inner_mmrs : list
        List of two-planet MMR terms, (k,nu), for the inner pair
    outer_mmrs : list
        List of two-planet MMR terms, (k,nu), for the outer pair
    max_order : int
        Maximum order of terms, in eccentricity and inclination,
        to retain in the three-body combinations of terms.  

    Returns
    -------
    list
        List of parings of individual terms in the format used by
        :func:`get_three_body_mmr_terms`
    """
    pairs = []
    l=(0,0)
    for k_inner,nu_inner in inner_mmrs:
        q_inner = np.array((0,*k_inner[:2]))
        order_inner = term_order(k_inner,nu_inner)
        res_inner = (i1,i2,k_inner,nu_inner,l)
        for k_outer,nu_outer in outer_mmrs:
            q_outer = np.array((*k_outer[:2],0))
            q_vec = q_outer + sgn * q_inner
            order_outer = term_order(k_outer,nu_outer)
            if order_inner + order_outer <= max_order+2:
                res_outer = (i2,i3,k_outer,nu_outer,l)

                pairs.append((res_inner,res_outer))
    return pairs

def get_three_body_mmr_terms(pham,res_pairs_list):
    # Frequency vector
    omega = pham.flow_func(*pham.state.values)[:pham.N_dof:3]
    omega = omega.reshape(-1)

    # Vector of freq. derivs, dn_i/dLambda_i
    domega = np.diag(pham.jacobian_func(*pham.state.values)[:pham.N_dof:3,pham.N_dof::3])

    # Initialize Poisson series to hold all the terms resulting from three-body interactions
    Npl  = pham.N-1
    symbol_kwargs = get_N_planet_poisson_series_symbols(Npl)
    h2_three_body_series = PoissonSeries(2*Npl,Npl,**symbol_kwargs)

    for res_in,res_out in res_pairs_list:
        i1_in,i2_in,k_in,nu_in,l_in = res_in
        i1_out,i2_out,k_out,nu_out,l_out = res_out
        terms_in  = DFTerm_as_PSterms(pham,i1_in,i2_in,k_in,nu_in,l_in)
        terms_out = DFTerm_as_PSterms(pham,i1_out,i2_out,k_out,nu_out,l_out)
        h1_in  = PoissonSeries.from_PSTerms(terms_in)
        h1_out = PoissonSeries.from_PSTerms(terms_out) 
        chi1_in  = PoissonSeries_to_GeneratingFunctionSeries(h1_in,omega,domega)
        chi1_out = PoissonSeries_to_GeneratingFunctionSeries(h1_out,omega,domega)

        # Add cross-terms to three-body hamiltonian series
        h2_three_body_series += 0.5 * (chi1_in.Lie_deriv(h1_out) + chi1_out.Lie_deriv(h1_in))

    return h2_three_body_series

def get_three_body_mmr_terms_minus(pham,res_pairs_list):
    r"""
    Generate 3-body resonance terms with arguments, :math:`k\cdot\lambda` that
    arise as the differences of cosine arguments of two-body terms. I.e.,
    :math:`k = k_\mathm{in} - k_\mathm{out}`

    Parameters
    ----------
    pham : celmech.PoincareHamiltonian
        Object representing the planetary system for which to compute 3-body MMR
        terms
    res_pairs_list : list
        A list specifying the pairs of two-body MMR tersm from which to
        construct the 3-body amplitude. The format of the list is [
        ((i1_in,i2_in,k_in,nu_in,l_in),(i1_out,i2_out,k_out,nu_out,l_out)),...]
    Returns
    -------
    celmech.poisson_series.PoissonSeries
        Poisson series repersentation of the the 3-body resonance terms.
    """
    # Frequency vector
    omega = pham.flow_func(*pham.state.values)[:pham.N_dof:3]
    omega = omega.reshape(-1)

    # Vector of freq. derivs, dn_i/dLambda_i
    domega =  - 3 * omega / np.array([L.xreplace(pham.H_params) for L in pham.Lambda0s[1:]])
    #domega = np.diag(pham.jacobian_func(*pham.state.values)[:pham.N_dof:3,pham.N_dof::3])

    # Initialize Poisson series to hold all the terms resulting from three-body interactions
    Npl  = pham.N-1
    symbol_kwargs = get_N_planet_poisson_series_symbols(Npl)
    h2_three_body_series = PoissonSeries(2*Npl,Npl,**symbol_kwargs)

    for res_in,res_out in res_pairs_list:
        i1_in,i2_in,k_in,nu_in,l_in = res_in
        i1_out,i2_out,k_out,nu_out,l_out = res_out
        h1_in  = DFTerm_as_PSterms(pham,i1_in,i2_in,k_in,nu_in,l_in)[0].as_series()
        h1_out = DFTerm_as_PSterms(pham,i1_out,i2_out,k_out,nu_out,l_out)[0].as_series()
        chi1_in  = PoissonSeries_to_GeneratingFunctionSeries(h1_in.conj,omega,domega)
        chi1_out = PoissonSeries_to_GeneratingFunctionSeries(h1_out.conj,omega,domega)

        # Add cross-terms to three-body hamiltonian series
        X = 0.5 * (chi1_in.Lie_deriv(h1_out) + chi1_out.Lie_deriv(h1_in))
        h2_three_body_series += X
        h2_three_body_series += X.conj

    return h2_three_body_series

def get_three_body_mmr_terms_plus(pham,res_pairs_list):
    r"""
    Generate 3-body resonance terms with arguments, :math:`k\cdot\lambda` that
    arise as the sums of cosine arguments of two-body terms. I.e., :math:`k =
    k_\mathm{in} + k_\mathm{out}`

    Parameters
    ----------
    pham : celmech.PoincareHamiltonian
        Object representing the planetary system for which to compute 3-body MMR
        terms
    res_pairs_list : list
        A list specifying the pairs of two-body MMR tersm from which to
        construct the 3-body amplitude. The format of the list is [
        ((i1_in,i2_in,k_in,nu_in,l_in),(i1_out,i2_out,k_out,nu_out,l_out)),...]

    Returns
    -------
    celmech.poisson_series.PoissonSeries
        Poisson series repersentation of the the 3-body resonance terms.
    """
    # Frequency vector
    omega = pham.flow_func(*pham.state.values)[:pham.N_dof:3]
    omega = omega.reshape(-1)

    # Vector of freq. derivs, dn_i/dLambda_i
    domega = np.diag(pham.jacobian_func(*pham.state.values)[:pham.N_dof:3,pham.N_dof::3])

    # Initialize Poisson series to hold all the terms resulting from three-body interactions
    Npl  = pham.N-1
    symbol_kwargs = get_N_planet_poisson_series_symbols(Npl)
    h2_three_body_series = PoissonSeries(2*Npl,Npl,**symbol_kwargs)

    for res_in,res_out in res_pairs_list:
        i1_in,i2_in,k_in,nu_in,l_in = res_in
        i1_out,i2_out,k_out,nu_out,l_out = res_out
        h1_in  = DFTerm_as_PSterms(pham,i1_in,i2_in,k_in,nu_in,l_in)[0].as_series()
        h1_out = DFTerm_as_PSterms(pham,i1_out,i2_out,k_out,nu_out,l_out)[0].as_series()
        chi1_in  = PoissonSeries_to_GeneratingFunctionSeries(h1_in,omega,domega)
        chi1_out = PoissonSeries_to_GeneratingFunctionSeries(h1_out,omega,domega)

        # Add cross-terms to three-body hamiltonian series
        X = 0.5 * (chi1_in.Lie_deriv(h1_out) + chi1_out.Lie_deriv(h1_in))
        h2_three_body_series += X
        h2_three_body_series += X.conj

    return h2_three_body_series


def three_body_mmr_periods_and_kvec(j1,k1,j2,k2,pm,Delta1):
    """
    Return orbital periods for three planets in a three-body MMR generated by
    the interactions of the j1:(j1-k1) and the j2:(j2 - k2) MMRs between the
    inner and outer pair, respectively.

    Parameters
    ----------
    j1 : int
        specifies inner MMR
    k1 : int
        specifies inner MMR order
    j2 : int
        specifies outer MMR
    k2 : int
        specifies outer MMR order
    pm : int
        Should be 1 or -1 to specify the three-body MMR as the sum or difference
        of k-vecotrs
    Delta1 : float
        distance of inner pair from exact MMR.
    
    Returns
    -------
    ndarray
        Array of orbital periods of planets.
    """
    kvec1 = np.array((k1-j1,j1,0))
    kvec2 = np.array((0,k2-j2,j2))
    kvec = kvec1 + pm * kvec2
    n1 = 2*np.pi
    n2 = n1 / (j1 * (Delta1 + 1) / (j1-k1)) 
    n3 = -1 * (kvec[0] * n1 + kvec[1] * n2) / kvec[2]
    return 2*np.pi / np.array((n1,n2,n3)), kvec

def three_body_mmr_n_and_dn(j1,k1,j2,k2,pm,masses,Delta1, n1 = 2*np.pi, GM=1):
    """
    Return orbital periods for three planets in a three-body MMR generated by
    the interactions of the j1:(j1-k1) and the j2:(j2 - k2) MMRs between the
    inner and outer pair, respectively.

    Parameters
    ----------
    j1 : int
        specifies inner MMR
    k1 : int
        specifies inner MMR order
    j2 : int
        specifies outer MMR
    k2 : int
        specifies outer MMR order
    pm : int
        Should be 1 or -1 to specify the three-body MMR as the sum or difference
        of k-vecotrs
    Delta1 : float
        distance of inner pair from exact MMR.
    
    Returns
    -------
    ndarray
        Array of orbital periods of planets.
    """
    kvec1 = np.array((k1-j1,j1,0))
    kvec2 = np.array((0,k2-j2,j2))
    kvec = kvec1 + pm * kvec2
    n2 = n1 / (j1 * (Delta1 + 1) / (j1-k1)) 
    n3 = -1 * (kvec[0] * n1 + kvec[1] * n2) / kvec[2]
    nvec = np.array([n1,n2,n3])
    a = (GM)**(1/3) * nvec**(-2/3)
    Lambda = masses * np.sqrt(GM * a)
    dn_vec = -3 * nvec / Lambda
    return nvec, dn_vec, Lambda

def Q_factors(j1,k1,j2,k2,n_vec,dn_vec):
    omega_in  = np.array((k1-j1,j1,0))@n_vec
    omega_out = np.array((0,k2-j2,j2))@n_vec
    c1 = (omega_in + omega_out)/(omega_in * omega_out)
    c2 = -1 * j1 * (j2-k2) * (dn_vec[1]) * (omega_in*omega_in + omega_out*omega_out)/(omega_in * omega_out)/(omega_in * omega_out)
    return c1,c2
    
def term_degree(ps_term):
    return np.sum(ps_term.k) + np.sum(ps_term.kbar)

def three_body_mmr(j1,k1,j2,k2,pm,Delta1,masses,zs,max_deg = None):
    if not max_deg:
        max_deg = k1 + k2 - 2
    periods,kvec = three_body_mmr_periods_and_kvec(j1,k1,j2,k2,pm,Delta1)
    P1,P2,P3 = periods
    omega = 2*np.pi/periods
    es = np.abs(zs)
    pomegas = np.angle(zs)
    sim = rb.Simulation()
    sim.add(m=1)
    for i in range(3):
        e,pomega = es[i],pomegas[i]
        sim.add(m=masses[i],P = periods[i], e = es[i],pomega = pomegas[i])
    sim.move_to_com()
    pvars = cm.Poincare.from_Simulation(sim)
    pham = cm.PoincareHamiltonian(pvars) 
    planets = pvars.particles[1:]
    dn_dLambda = -3*np.array([p.n/p.Lambda for p in planets])
    mmr1_terms = list_resonance_terms(j1,k1,min_order=k1,max_order=k1,inclinations=False)
    mmr2_terms = list_resonance_terms(j2,k2,min_order=k2,max_order=k2,inclinations=False)
    lst = pair_mmr_lists(1,2,3,kvec,pm,mmr1_terms,mmr2_terms,max_deg)
    full_series = get_three_body_mmr_terms(pham,lst)
    cond = lambda term: np.all(term.q==kvec) and np.sum(term.p)==0 and term_degree(term)<=max_deg
    select_terms = [term for term in full_series.terms if cond(term)]
    select_series = PoissonSeries.from_PSTerms(select_terms)
    A = kvec**2 @ dn_dLambda
    xvec = np.array([p.x for p in planets] + [p.y for p in planets])
    eps = 2 * np.abs(select_series(xvec,np.zeros(3),np.zeros(3)))
    dP  = 2 * np.sqrt(abs(eps / A))
    k_d_ln_omega = kvec * (dn_dLambda/omega)

    xlow = P1/P2 * (1 + (k_d_ln_omega[1] - k_d_ln_omega[0]) * dP)
    xhi  = P1/P2 * (1 - (k_d_ln_omega[1] - k_d_ln_omega[0]) * dP) 
    
    ylow = P2/P3 * (1 + (k_d_ln_omega[2] - k_d_ln_omega[1]) * dP)
    yhi  = P2/P3 * (1 - (k_d_ln_omega[2] - k_d_ln_omega[1]) * dP) 

    return np.array(((xlow,xhi),(ylow,yhi))), select_series, A, sim
    


def in_mmrs_Q(ms,Pjk_in,Pjk_out,state_values,GM = 1 ,safety_factor = 1.):
    
    Lambdas = state_values.T[3*3 + np.array((0,3,6))]
    etas = state_values.T[1 + np.array((0,3,6))]
    kappas = state_values.T[1 + 3*3 + np.array((0,3,6))]
    xs = (kappas-1j*etas)/np.sqrt(2)

    in_inner_res_Q={}
    for jk,Pjk in Pjk_in.items():
        j,k = jk
        jvec = np.array((k-j,j,0))
        n,dn,_ = three_body_mmr_n_and_dn(j,k,j,k,-1,ms,0.0,n1=2*np.pi,GM=GM)
        Minv = jvec**2 @ dn
        Ajk = np.array([Pjk([x],[],[]) for x in xs.T])
        dI = safety_factor * 2 * np.sqrt(2 * np.abs(Ajk / Minv))
        n_res_plus = np.transpose([n + jvec * dn * X for X in dI])        
        n_res_minus = np.transpose([n - jvec * dn * X for X in dI])
        Pratio_plus = n_res_plus[0]/n_res_plus[1]        
        Pratio_minus = n_res_minus[0]/n_res_minus[1]
        Pratio_obs = (Lambdas[1]/Lambdas[0])**3 * (ms[0]/ms[1])**3
        in_inner_res_Q[jk] = np.logical_and(Pratio_obs<Pratio_plus,Pratio_obs>Pratio_minus)

    in_outer_res_Q={}
    for jk,Pjk in Pjk_out.items():
        j,k = jk
        jvec = np.array((0,k-j,j))
        n,dn,_ = three_body_mmr_n_and_dn(j,k,j,k,-1,ms,0.0,n1=2*np.pi,GM=GM)
        Minv = jvec**2 @ dn
        Ajk = np.array([Pjk([x],[],[]) for x in xs.T])
        dI = safety_factor * 2 * np.sqrt(2 * np.abs(Ajk / Minv))
        n_res_plus = np.transpose([n + jvec * dn * X for X in dI])        
        n_res_minus = np.transpose([n - jvec * dn * X for X in dI])
        Pratio_plus = n_res_plus[1]/n_res_plus[2]        
        Pratio_minus = n_res_minus[1]/n_res_minus[2]
        Pratio_obs = (Lambdas[2]/Lambdas[1])**3 * (ms[1]/ms[2])**3
        in_outer_res_Q[jk] = np.logical_and(Pratio_obs<Pratio_plus,Pratio_obs>Pratio_minus)
    return in_inner_res_Q,in_outer_res_Q