import time

import fenics as fe
import sys, os
import numpy as np
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
from common_PDE import EquSolver, ModelSS
noise_level = 0.05
from core.noise import NoiseGaussianIID
DATA_DIR = './DATA/'
equ_nx = np.load('equ_nx.npy')
domain_equ = Domain1D(n=equ_nx, mesh_type='CG', mesh_order=1)
k=1
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )
measurement_points = np.load(DATA_DIR + "measurement_points_1D.npy")
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'CG', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain_equ.function_space)
alpha = 0.01
equ_solver = EquSolver(domain_equ=domain_equ, alpha=alpha, \
                           points=np.array([measurement_points]).T, m=truth_fun)
d = np.load(DATA_DIR + "measurement_noise_1D" + "_" + str(noise_level) + ".npy")
d_clean = np.load(DATA_DIR + "measurement_clean_1D.npy")
noise_level_ = noise_level*abs(max(d_clean))
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)
model = ModelSS(
        d=d_clean, domain_equ=domain_equ, prior=prior_measure,
        noise=noise, equ_solver=equ_solver
    )

def Phi(u_vec):
    model.update_m(m_vec=u_vec,update_sol=True)
    ans=model.loss_residual()
    return ans
############################################################################
t1=time.time()
samplesize=3*10**6
beta=0.02
Sample=np.zeros((samplesize,equ_nx+1))
u_vec=prior_measure.mean_vec
trace=np.zeros((samplesize))
accept=0
for i in range(samplesize):
    Sample[i,:]=u_vec
    v_vec=np.sqrt(1-beta**2)*u_vec+beta*prior_measure.generate_sample()
    a_u_v=min(1.0,np.exp(Phi(u_vec)-Phi(v_vec)))
    if (a_u_v>np.random.rand(1)):
        u_vec=v_vec
        accept=accept+1
    trace[i]=u_vec[30]
    if (i%100 ==0):
        print(i)
print(accept/samplesize)
np.save(DATA_DIR+"pCNsamples",Sample)
np.save(DATA_DIR+"pCNtrace",trace)
t2=time.time()
print(t2-t1)










