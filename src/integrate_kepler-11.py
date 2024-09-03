import rebound as rb
import numpy as np
import celmech as cm
import sys
I = int(sys.argv[1])
init_file = "/fs/lustre/cita/hadden/07_secular/SecularChaos/median_kepler-11-config.bin"
fAMD = 1.51
sim_file = "/fs/lustre/cita/hadden/07_secular/SecularChaos/kepler-11-ics/kep-11_fAMD_{:.2f}_id{:03d}.bin".format(fAMD,I)
from celmech.secular import SecularSystemSimulation
import numpy as np
from utils import farey_sequence
def run_secular_system_simulation(sec_sim,times,out_file):
    r"""
    Integrate the input secular simulation and get
    output at the specified times. Results returned
    as a dictionary with the trajectory as stored as
    ``qp`` along with the times, energy, and AMD.
    """
    Nout = len(times)
    qp0 = sec_sim.state_to_qp_vec()
    qp_solution = np.zeros((Nout, qp0.shape[0]))
    energy = np.zeros(Nout)
    amd = np.zeros(Nout)
    times_done = np.zeros(Nout)
    for i,t in enumerate(times):
        sec_sim.integrate(t)
        times_done[i] = sec_sim.t
        qp_solution[i] = sec_sim.state_to_qp_vec()
        energy[i] = sec_sim.calculate_energy()
        amd[i] = sec_sim.calculate_AMD()
        np.savez_compressed(out_file,
                            times=times_done[:i+1],
                            energy=energy[:i+1],
                            amd=amd[:i+1],
                            qp=qp_solution[:i+1]
                        )

from SampleAMD import get_critical_AMD
sim = rb.Simulation(init_file)
pvars = cm.Poincare.from_Simulation(sim)
pham = cm.PoincareHamiltonian(pvars)
res_data = {}
for i in range(1,pham.N):
    pi = pham.particles[i]
    # loop over inner particle
    for j in range(i+1,pham.N):
        pj = pham.particles[j]
        # add secular terms of second order in mass
        J_low = int(np.floor(1 + 1/(pj.P/pi.P-1)))
        res_data[(i,j)] = [(d* int(J_low) + n,d) for n,d in farey_sequence(4//2)]
sec_sim = SecularSystemSimulation.from_Simulation(
    sim,
    method='RK',
    dtFraction=0.05,
    rk_kwargs={'rk_method':'GL6'},
    resonances_to_include=res_data
)
sec_sim._integrator.atol = 1e-11 * np.sqrt(get_critical_AMD(pvars))

sim_ics = rb.Simulation(sim_file)
pvars_ic = cm.Poincare.from_Simulation(sim_ics)
sec_sim.state.values = pvars_ic.values

times = np.linspace(0,1.e9,5_000)

outfile = "/fs/lustre/cita/hadden/07_secular/SecularChaos/data/kep-11_fAMD_{:.2f}_id{:03d}_secular_soln.bin".format(fAMD,I)
run_secular_system_simulation(sec_sim,times,outfile)
