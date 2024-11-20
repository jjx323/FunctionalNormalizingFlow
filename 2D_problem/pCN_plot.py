import fenics as fe
import matplotlib.pyplot as plt
import sys, os
import numpy as np
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2
noise_level = 0.05
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

samplesize=3*10**6
p_num=5000
pi=int(samplesize/p_num)
mean=np.zeros((dim))
for i in range(pi-20):
    print(i)
    pcn=np.load('pCN/'+'pcn'+str(i+20)+'.npy')
    mean_i=np.percentile(pcn.T,50,axis=1)
    mean=mean+mean_i
mean=mean/pi
np.save('pCN/mean',mean)
ymean_fun=fe.Function(domain_equ.function_space)
ymean_fun.vector()[:]=mean

fig2=fe.plot(ymean_fun)
plt.colorbar(fig2)
plt.savefig('PIC/pcnmean')
plt.close()


fig = fe.plot(truth_fun, label='Truth')
plt.colorbar(fig)
plt.title("Truth")
plt.savefig('PIC/truth')
plt.close()

cov_total=np.zeros((dim,dim))
for j in range(pi-20):
    print(j)
    cov = np.zeros((dim, dim))
    vec_total=np.load('pCN/'+'pcn'+str(j+20)+'.npy')
    for i in range(p_num):
        vec = vec_total[i,:] - mean
        vec = np.reshape(vec, (1, dim))
        M = vec.T @ vec
        covi=M
        cov=cov+covi
    cov=cov/p_num
    cov_total=cov_total+cov
cov_total=cov_total/pi
np.save('pCN/cov_total',cov_total)