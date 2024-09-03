import rebound as rb
import celmech as cm
from celmech.lie_transformations import FirstOrderGeneratingFunction
from ConstructSecularSystemPoissonSeries import SecondOrderSecularHamiltonianFromSimulation
import numpy as np
from celmech.poisson_series import PSTerm
import sys
sys.path.append("../src/")
I = int(sys.argv[1])
datadir = "/fs/lustre/cita/hadden/07_secular/SecularChaos/data/"
fi_name_template = "kep-11_fAMD_{:.2f}_id{:03d}.bin"

def get_sims_data(sa,func,times=None):
    sim0 = sa[0]
    data0 = func(sim0)
    if times:
        iterator = sa.getSimulations(times)
        Ntimes = len(times)
    else:
        iterator = sa
        Ntimes = len(sa)
    data = np.zeros((Ntimes,*data0.shape),dtype = data0.dtype)
    times_done = np.zeros(Ntimes)
    for i,sim in enumerate(iterator):
        sim.integrator_synchronize()
        times_done[i] = sim.t
        data[i] = func(sim)
    return times_done,data

def get_mean_xy_vars(sim,chi):
    pvars = cm.Poincare.from_Simulation(sim)
    chi.state.values = pvars.values
    chi.state.t = sim.t
    chi.osculating_to_mean()
    return np.array([p.x for p in chi.state.particles[1:]] + [p.y for p in chi.state.particles[1:]])
def get_osculating_xy_vars(sim):
    pvars = cm.Poincare.from_Simulation(sim)
    return np.array([p.x for p in pvars.particles[1:]] + [p.y for p in pvars.particles[1:]])

f_AMDs = [0.33,0.49,0.71,1.04,1.51,2.21,3.22,4.70,6.85,10]

for f_AMD in f_AMDs:
    print("Running f_AMD = {:.2f}".format(f_AMD))
    sa = rb.Simulationarchive(datadir+fi_name_template.format(f_AMD,I))
    sim0 = sa[0]
    pvars = cm.Poincare.from_Simulation(sim0)
    pham = cm.PoincareHamiltonian(pvars)
    chi = FirstOrderGeneratingFunction(pvars)
    for i in range(1,chi.N-1):
        p1 = chi.particles[i]
        chi.add_zeroth_order_term(i,i+1)
        for j in range(i+1,chi.N):
            p2 = chi.particles[j]
            PR = p2.P/p1.P
            J = int(np.floor(1 + 1/(PR-1)))
            if J>1:
                chi.add_MMR_terms(J-1,1,indexIn=i,indexOut=j)
            chi.add_MMR_terms(J,1,indexIn=i,indexOut=j)
            chi.add_MMR_terms(J+1,1,indexIn=i,indexOut=j)
            chi.add_MMR_terms(J+2,1,indexIn=i,indexOut=j)

    #times_xy, mean_xy_data = get_sims_data(sa,lambda x: get_mean_xy_vars(x,chi))
    times_xy, osc_xy_data = get_sims_data(sa,lambda x: get_mean_xy_vars(x,chi))

    pvars = cm.Poincare.from_Simulation(sim0)
    chi.state.values = pvars.values
    chi.state.t = sim0.t
    chi.osculating_to_mean()
    sim0_mean = chi.state.to_Simulation()
    sec_series,chi_series = SecondOrderSecularHamiltonianFromSimulation(sim0_mean,2)
    Se = np.zeros((sec_series.N//2,sec_series.N//2))
    SI = np.zeros((sec_series.N//2,sec_series.N//2))
    eyeN = np.eye(sec_series.N,dtype=int)
    zeroM = np.zeros(sec_series.M,dtype=int)
    for i in range(sec_series.N//2):
        for j in range(sec_series.N//2):
            key = sec_series._PSTerm_to_key(
                PSTerm(
                    1,
                    eyeN[i],
                    eyeN[j],
                    zeroM,
                    zeroM
                )
            )
            Se[i,j] = np.real(sec_series[key])
            # inclinatino
            key = sec_series._PSTerm_to_key(
                PSTerm(
                    1,
                    eyeN[sec_series.N//2 + i],
                    eyeN[sec_series.N//2 + j],
                    zeroM,
                    zeroM
                )
            )
            SI[i,j] = np.real(sec_series[key])

    ge,Te = np.linalg.eigh(Se)
    gI,TI = np.linalg.eigh(SI)
    # u = Te.T @ np.transpose(mean_xy_data[:,:6])
    # v = TI.T @ np.transpose(mean_xy_data[:,6:])
    u = Te.T @ np.transpose(osc_xy_data[:,:6])
    v = TI.T @ np.transpose(osc_xy_data[:,6:])

    save_file = datadir+"fAMD_osc_element_secular_data_{:.2f}_id{:03d}".format(f_AMD,I)
    print(f"Saving data to {save_file}")
    np.savez_compressed(save_file,u=u,v=v,xy=osc_xy_data,time=times_xy,Se=Se,SI=SI,Te=Te,TI=TI)