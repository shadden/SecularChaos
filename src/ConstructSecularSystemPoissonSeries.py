import celmech as cm
from celmech.poisson_series import *
from celmech.disturbing_function import list_secular_terms
from celmech.disturbing_function import list_resonance_terms
from utils import *

def SecondOrderSecularHamiltonianFromSimulation(sim,max_degree,eccentricities=True,inclinations=True):
    pvars = cm.Poincare.from_Simulation(sim)
    pham = cm.PoincareHamiltonian(pvars)
    sec_args = list_secular_terms(2,max_degree,eccentricities,inclinations)
    Npl = pham.N-1
    secular_series = PoissonSeries(2*Npl,Npl)
    chi_series = PoissonSeries(2*Npl,Npl)
    omega   = np.array([p.n for p in pham.particles[1:]])
    domega =  - 3 * omega / np.array([L.xreplace(pham.H_params) for L in pham.Lambda0s[1:]])
    # loop over inner particle 
    for i in range(1,pham.N):
        pi = pham.particles[i]

        # loop over order particle
        for j in range(i+1,pham.N):
            pj = pham.particles[j]

            # add secular terms of first order in mass
            secular_series_terms =[]
            for k,nu in sec_args:
                secular_series_terms += DFTerm_as_PSterms(pham,i,j,k,nu,(0,0))
            pair_secular_series = PoissonSeries.from_PSTerms(secular_series_terms)
            secular_series+=pair_secular_series
            
            # add secular terms of second order in mass
            J_low = int(np.floor(1 + 1/(pj.P/pi.P-1)))
            bracketing_resonances = [(d* int(J_low) + n,d) for n,d in farey_sequence(max_degree//2)]

            # loop over bracketing resonances up to order max_degree/2
            for p,q in bracketing_resonances:

                # get resonant arguments for the p:(p-q) resonance
                res_series_terms = []
                res_args = list_resonance_terms(
                    p,q,
                    max_order = max_degree//2,
                    eccentricities=eccentricities,
                    inclinations=inclinations
                )
                # construct Poisson series for resonant perturbation
                for k,nu in res_args:
                    res_series_terms += DFTerm_as_PSterms(pham,i,j,k,nu,(0,0))
                pair_res_series = PoissonSeries.from_PSTerms(res_series_terms)

                # Get first-order generating function that cancels resonant perturbation
                pair_chi_series = PoissonSeries_to_GeneratingFunctionSeries(pair_res_series,omega,domega)

                # Get second-order perturbation
                pair_so_series = 0.5 * pair_chi_series.Lie_deriv(pair_res_series)
                so_secular_terms =  [t for t in pair_so_series.terms if np.all(t.q==0) and np.all(t.p==0)]
                assert len(so_secular_terms)>0, f"No secular terms generated for the {p}:{p-q} resonance between {i} and {j}"
                pair_so_secular_series = PoissonSeries.from_PSTerms(
                   so_secular_terms
                )

                # add resonant terms to the global generating function and
                # second order secular terms to the global secular hamiltonian
                chi_series+=pair_chi_series
                secular_series += pair_so_secular_series
    return secular_series, chi_series

    