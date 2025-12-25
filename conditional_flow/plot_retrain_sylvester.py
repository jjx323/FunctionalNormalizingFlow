import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import fenics as fe
plt.rcParams['font.family'] = 'Times New Roman'
import math
torch.manual_seed(8)
from core.model import Domain2D
from core.probability import GaussianElliptic2
from core.misc import trans2spnumpy, spnumpy2sptorch,sptorch2spnumpy
DATA_DIR = './DATA/'
nump=5001
equ_nx = 20
dim=(equ_nx+1)**2
domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
## setting the prior measure
k=1
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=(0.1/k), a_fun=(1.0/k), theta=1.0, boundary='Neumann'
    )
M=prior_measure.M
K=prior_measure.K
def csr_torch(matrix):
    Mcoo = matrix.tocoo()
    Mpt = torch.sparse.FloatTensor(torch.LongTensor([Mcoo.row.tolist(), Mcoo.col.tolist()]),
                                   torch.FloatTensor(Mcoo.data.astype(float)))
    Mpt = Mpt.to(torch.float32)
    return Mpt
M_torch=csr_torch(M)
Psi_r=np.load('./RESULT/'+'eig_vec_'+str(dim)+'.npy')
Psi_r_torch=torch.from_numpy(Psi_r)
Psi_r_torch=Psi_r_torch.to(torch.float32)
m=20
Psi_r_torch=Psi_r_torch[:,:m]
class sylvesterFlow(nn.Module):
    def __init__(self):
        super().__init__()
        q=0.1
        self.R1 = nn.Parameter(q*torch.rand(m,m))
        self.R2 = nn.Parameter(q*torch.rand(m,m))
        self.dia1_=nn.Parameter(q*torch.rand(m))
        self.b = nn.Parameter(q*torch.rand(m,1))
        self.h=nn.Tanh()
        self.h_prime = lambda z: (1 - self.h(z) ** 2)
        self.name='sylvesterFlow'
    def constrained_r1(self):
        r_=torch.triu(self.R1, diagonal=1)
        diag_=self.h(self.dia1_)+0.1**5
        diag=torch.diag(diag_)
        r=diag+r_
        return r,diag_
    def constrained_r2(self):
        r_=torch.triu(self.R2, diagonal=1)
        diag_=torch.ones(m)
        diag=torch.diag(diag_)
        r=diag+r_
        return r,diag_
    def forward(self, z):
        r1,diag1=self.constrained_r1()
        r2, diag2 = self.constrained_r2()
        z = z.to(torch.float32)
        hidden_units = r2@Psi_r_torch.T@(M_torch@z.T) + self.b
        x = z + (Psi_r_torch@r1@self.h(hidden_units)).T
        psi = self.h_prime(hidden_units)
        ans=(diag1*diag2)
        ans=torch.reshape(ans,(m,1))
        w=ans*psi
        w1=w+1
        det=torch.prod(w1,dim=0)
        log_det = torch.log(det)
        return x, log_det
class NormalizingFlow(nn.Module):
    def __init__(self, flow_length):
        super().__init__()
        self.layers = nn.Sequential(
            *(sylvesterFlow() for _ in range(flow_length)))
    def forward(self, z):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(z)
            log_jacobians += log_jacobian
        return z, log_jacobians

##############################################################
flow_length=5
flow = NormalizingFlow(flow_length)
flow.load_state_dict(torch.load('retrain/TEST_retrain_sylvester_'+str(nump)))
from prior import prior_sample_torch
x=prior_sample_torch(num=500,prior=prior_measure)
y,sss=flow.forward(x)
y=y.detach().numpy()
yyy1=np.percentile(y.T,2.5,axis=1)
yyy2=np.percentile(y.T,97.5,axis=1)
ymean=np.percentile(y.T,50,axis=1)
mean_fun=fe.Function(domain_equ.function_space)
mean_fun.vector()[:]=ymean
yyy1_fun=fe.Function(domain_equ.function_space)
yyy1_fun.vector()[:]=yyy1
yyy2_fun=fe.Function(domain_equ.function_space)
yyy2_fun.vector()[:]=yyy2
truth_fun=fe.Function(domain_equ.function_space)
truth_fun.vector()[:]=np.array(np.load(DATA_DIR + 'm_data.npy')[nump,:])
##################################################################################################
plt.figure(figsize=(18, 5))
plt.subplot(1, 5, 1)
fig = fe.plot(truth_fun, label='truth')
plt.colorbar(fig)
plt.title("Truth")
plt.subplot(1, 5, 2)
fig = fe.plot(mean_fun, label='mean')
plt.colorbar(fig)
plt.legend()
plt.title('Mean')
plt.subplot(1, 5, 3)
fig = fe.plot(yyy2_fun, label='up 95')
plt.colorbar(fig)
plt.legend()
plt.title('up')
plt.subplot(1, 5, 4)
fig = fe.plot(yyy1_fun, label='down 95')
plt.colorbar(fig)
plt.legend()
plt.title('down')
flow_samples=y
plt.subplot(1,5,5)
mean = np.mean(flow_samples, axis=0)
print(mean.shape)
# print(pCN.shape)
diag = np.zeros((dim))
num = flow_samples.shape[0]
print(num)
for i in range(num):
        vec = flow_samples[i, :] - mean
        vec = np.reshape(vec, (1, dim))
        M = vec.T @ vec
        diagi = np.diagonal(M)
        diag = diagi + diag
diag = diag / num
fun = fe.Function(domain_equ.function_space)
fun.vector()[:] = np.array(np.sqrt(diag))
fig = fe.plot(fun)
plt.title('covariance')
plt.colorbar(fig)
plt.show()
plt.subplot(1, 1, 1)
fig = fe.plot(mean_fun, label='Truth')
cbar=plt.colorbar(fig)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
cbar.locator = MaxNLocator(nbins=6)
cbar.ax.tick_params(labelsize=16)
plt.legend()
plt.show()
fig = fe.plot(mean_fun, label='mean')
plt.colorbar(fig)
plt.legend()
plt.title('Mean')
plt.show()
mean = np.mean(flow_samples, axis=0)
fun.vector()[:] = np.array(np.sqrt(diag))
fig = fe.plot(fun)
plt.title('Point-wise Variance Field')
plt.colorbar(fig)
plt.show()
cov=np.zeros((dim,dim))
num = y.shape[0]
for i in range(num):
        vec = y[i, :] - ymean
        vec = np.reshape(vec, (1, dim))
        M = vec.T @ vec
        covi=M
        cov=cov+covi
cov=cov/(num)
q=80
cov2=np.diagonal(cov,offset=q)
plt.plot(cov2,'r--',label='Sylvester')
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # x轴主要刻度的数量
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
plt.xticks(fontsize=16)  # x轴刻度的字体大小
plt.yticks(fontsize=16)
plt.legend(loc=1,fontsize='large')
plt.show()


