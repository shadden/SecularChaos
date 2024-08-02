import numpy as np
import sys
indx = int(sys.argv[1])

fistring = "/fs/lustre/cita/hadden/07_secular/SecularChaos/data/kep-11_fAMD_3.22_id{:03d}_secular_soln.bin.npz"
results = np.load(fistring.format(indx))
time = results['times']
qp_data = results['qp']
energy = results['energy']
AMD = results['amd']
Npl = 6

eta,rho,kappa,sigma = [qp_data[:,i*Npl:(i+1)*Npl] for i in range(4)]
x = np.sqrt(0.5) * (kappa - 1j * eta)
y = np.sqrt(0.5) * (sigma - 1j * rho)

import pickle
with open("/fs/lustre/cita/hadden/07_secular/SecularChaos/data/kep-11_secular_info.bin","rb") as fi:
    A,omega,K4_pseries,Te,TI = pickle.load(fi)
u = Te.T @ x.T
v = TI.T @ y.T
from GroupResonancesPoissonSeries import group_resonances_poisson_series
resonances = group_resonances_poisson_series(K4_pseries)

uv_data = np.transpose(np.vstack((u,v)))

counts = {key:0 for key in resonances.keys() if key!='secular'}
res_times = {key:[] for key in resonances.keys() if key!='secular'}
for kres,Hres in resonances.items():
    if kres=='secular':
        continue
    M = 1/(kres@A@kres)
    for t,uv in zip(time,uv_data):
        P = np.abs(uv)**2
        Omega = kres @ (omega + A @ P)
        pres = -1*Omega*M
        Pres = P + np.array(kres) * pres
        if np.all(Pres>=0):
            eps = np.abs(Hres(np.sqrt(Pres),[],[]))
            dp_res = 2 * np.sqrt(np.abs(M*eps))
            if dp_res > np.abs(pres):
                counts[kres] +=  1
                res_times[kres] +=  [t]

with open("/fs/lustre/cita/hadden/07_secular/SecularChaos/data/kep-11_fAMD_3.22_id{:03d}_secular_resonance_data.bin","wb") as fi:
    pickle.dump((counts,res_times,uv_data),fi)