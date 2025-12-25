import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
from matplotlib.ticker import MaxNLocator
torch.manual_seed(1)
###############################################################################
from core.model import Domain1D
from core.probability import GaussianElliptic2
equ_nx = np.load('equ_nx.npy')
dim=equ_nx+1
k=1
domain_equ = Domain1D(n=equ_nx, mesh_type='CG', mesh_order=1)
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
Psi_r=np.load('./RESULT/'+'eig_vec_L2'+'_'+str(dim)+'.npy')
Psi_r_torch=torch.from_numpy(Psi_r)
Psi_r_torch=Psi_r_torch.to(torch.float32)
ww=Psi_r_torch@torch.rand(20)
################################################################################
m=20
Psi_r_torch=Psi_r_torch[:,:m]
################################################################################
from common_flows_dis_inv import (PlanarFlow,HouseholderFlow,project_transformFlow,
                          sylvesterFlow,NormalizingFlow,model_train)
path='model_dir/'
a=2
if (a==1):
    model_style = PlanarFlow
    model_load = 'PlanarFlow.zip'
if (a==2):
    model_style = HouseholderFlow
    model_load = 'HouseholderFlow.zip'
if (a==3):
    model_style = project_transformFlow
    model_load = 'project_transformFlow.zip'
if (a==4):
    model_style = sylvesterFlow
    model_load = 'sylvesterFlow.zip'
##############################################################
flow_length=24
flow = NormalizingFlow(model_style,flow_length,dim)
flow.load_state_dict(torch.load(path+model_load))
from prior import prior_sample_torch
x=prior_sample_torch(num=500,prior=prior_measure)
y,sss=flow.forward(x)
y=y.detach().numpy()
flow_samples=y
mean = np.mean(flow_samples, axis=0)
cov=np.zeros((dim,dim))
num = flow_samples.shape[0]
for i in range(num):
        vec = flow_samples[i, :] - mean
        vec = np.reshape(vec, (1, dim))
        M = vec.T @ vec
        covi=M
        cov=cov+covi
cov=cov/num
np.save('cov',cov)
cov_pcn=np.load('DATA/cov.npy')
for q in (0,10,20):
    cov1=np.diagonal(cov,offset=q)
    cov2=np.diagonal(cov_pcn,offset=q)
    plt.tight_layout(pad=1.5, w_pad=0.3, h_pad=0.3)
    plt.plot(cov2,label='pCN')
    plt.plot(cov1,'r--',label='Householder')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc=1,fontsize='large')
    plt.savefig('PIC/'+str(q)+'_cov')
    plt.close()
    d=cov1-cov2
    dd=d@d/np.sum(cov2**2)
    print('N_g=',q,',','relative error=',dd)
    a=np.sum(cov_pcn**2)
    b=np.sum((cov-cov_pcn)**2)
    print('total relative error=',b/a)




