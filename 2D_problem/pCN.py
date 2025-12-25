import time

import fenics as fe
import matplotlib.pyplot as plt
import sys, os
import numpy as np
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2
from core.misc import load_expre, smoothing
from common_PDE import EquSolver, ModelDarcyFlow
noise_level = 0.05
from core.noise import NoiseGaussianIID
DATA_DIR = './DATA/'
equ_nx = 20
dim=(equ_nx+1)**2
domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
k=1
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )
measurement_points = np.load(DATA_DIR + "measurement_points_2D.npy")
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'CG', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain_equ.function_space)
f_expre = load_expre(DATA_DIR + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f_expre, degree=2), domain_equ.function_space)
equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, points=measurement_points)
d = np.load(DATA_DIR + "measurement_noise_2D" + "_" + str(noise_level) + ".npy")
d_clean = np.load(DATA_DIR + "measurement_clean_2D.npy")
noise_level_ = noise_level*abs(max(np.absolute(d_clean)))
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)
model = ModelDarcyFlow(
        d=d_clean, domain_equ=domain_equ, prior=prior_measure,
        noise=noise, equ_solver=equ_solver
    )
def Phi(u_vec):
    model.update_m(m_vec=u_vec,update_sol=True)
    ans=model.loss_residual()
    return ans
############################################################################
samplesize=3*10**6
beta=0.01
u_vec=prior_measure.mean_vec
print(u_vec.shape)
trace=np.zeros((samplesize))
print(Phi(u_vec))
p_num=5000
pi=int(samplesize/p_num)
samples_5000=np.zeros((p_num,dim))
for j in range(pi):
    t1=time.time()
    accept = 0
    u_vec=samples_5000[-1,:]
    print('Phi_now=',Phi(u_vec))
    for i in range(p_num):
        v_vec=np.sqrt(1-beta**2)*u_vec+beta*prior_measure.generate_sample()
        a_u_v=min(1.0,np.exp(Phi(u_vec)-Phi(v_vec)))
        if (a_u_v>np.random.rand(1)):
            u_vec=v_vec
            accept=accept+1
        if (i%100 ==0):
            print(i)
        samples_5000[i, :] = u_vec
    np.save('pCN/'+'pcn'+str(j)+'.npy',samples_5000)
    print(accept / p_num)
    t2=time.time()
    print('time',t2-t1)











