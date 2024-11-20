import torch
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
from core.model import Domain2D
plt.rcParams['font.family'] = 'Times New Roman'
from matplotlib.ticker import MaxNLocator
torch.manual_seed(1)
from prior import prior_sample_torch
from core.probability import GaussianElliptic2
equ_nx = 30
dim=(equ_nx+1)**2
domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
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
from common_flows_dis_inv import (PlanarFlow,HouseholderFlow,project_transformFlow,
                          sylvesterFlow,NormalizingFlow)
path='model_dir/'
a=4
if (a==1):
    model_style = PlanarFlow
    length=32
    model_load = 'PlanarFlow.zip'
if (a==2):
    model_style = HouseholderFlow
    length = 32
    model_load = 'HouseholderFlow.zip'
if (a==3):
    model_style = project_transformFlow
    length = 5
    model_load = 'project_transformFlow.zip'
if (a==4):
    model_style = sylvesterFlow
    length = 5
    model_load = 'sylvesterFlow.zip'
flow_length=length
flow = NormalizingFlow(model_style,flow_length,dim=dim)
flow.load_state_dict(torch.load(path+model_load))
x=prior_sample_torch(num=2000,prior=prior_measure)
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
np.save('mean',ymean)
truth_fun_shape = fe.Expression('exp(-20*(x[0]-0.3)*(x[0]-0.3)-20*(x[1]-0.3)*(x[1]-0.3))+exp(-20*(x[0]-0.7)*(x[0]-0.7)-20*(x[1]-0.7)*(x[1]-0.7))', degree=1)
truth_fun = fe.interpolate(truth_fun_shape, domain_equ.function_space)
truth_fun_vec=truth_fun.vector()[:]
flow_samples=y
mean = np.mean(flow_samples, axis=0)
cov_flow=np.zeros((dim,dim))
num = flow_samples.shape[0]
for i in range(num):
        vec = flow_samples[i, :] - mean
        vec = np.reshape(vec, (1, dim))
        M = vec.T @ vec
        covi=M
        cov_flow=cov_flow+covi
cov_pcn=np.load('pCN/cov_total.npy')
cov_flow=cov_flow/num
for q in (0,40,80):
    cov1=np.diagonal(cov_flow,offset=q)
    cov2=np.diagonal(cov_pcn,offset=q)
    plt.plot(cov2,label='pCN')
    plt.plot(cov1,'r--',label='Sylvester')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc=1,fontsize=16)
    plt.savefig('PIC/cov_'+str(q))
    plt.close()

    d=cov1-cov2
    dd=d@d/np.sum(cov2**2,)
    print('relative error of cov with N_g='+ str(q)+': ', dd)
    a=np.sum(cov_pcn**2)
    b=np.sum((cov_flow-cov_pcn)**2)
print('total relative error: ', b/a)