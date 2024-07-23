import rebound as rb
from matplotlib import pyplot as plt
import celmech as cm
from celmech.secular import LaplaceLagrangeSystem
import numpy as np
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft

sa = rb.Simulationarchive("../data/Kepler-11_000.bin")
sim = sa[0]
llsys = LaplaceLagrangeSystem.from_Simulation(sim)
for i in range(1,llsys.N):
    pi = llsys.particles[i]
    for j in range(i+1,llsys.N):
        pj = llsys.particles[j]
        J = 1 + 1/(pj.P/pi.P - 1)
        if J>2:
            llsys.add_first_order_resonance_term(i,j,int(np.floor(J)))
            llsys.add_first_order_resonance_term(i,j,int(np.ceil(J)))

Te,De = llsys.diagonalize_eccentricity()
ge = -1*np.diag(De)
Tsec_max = np.max(2*np.pi / ge)
times = np.linspace(0,Tsec_max,512)
x = np.zeros((sim.N-1,512),dtype = np.complex128)
for i,sim in enumerate(sa.getSimulations(times,mode='close')):
    sim.integrator_synchronize()
    pvars = cm.Poincare.from_Simulation(sim)
    for j,p in enumerate(pvars.particles[1:]):
        x[j,i] = p.x
    