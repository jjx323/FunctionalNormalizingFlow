import torch
import numpy as np
from core.model import Domain2D
torch.manual_seed(1)
import scipy
from core.probability import GaussianElliptic2
equ_nx = 20
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
from common_flows import (PlanarFlow,HouseholderFlow,project_transformFlow,
                          sylvesterFlow,NormalizingFlow,model_train)
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
flow = NormalizingFlow(model_style,flow_length)
flow.load_state_dict(torch.load(path+model_load))
from prior import prior_sample_torch
x=prior_sample_torch(num=1000,prior=prior_measure)
y,sss=flow.forward(x)
y=y.detach().numpy()
yyy1=np.percentile(y.T,2.5,axis=1)
yyy2=np.percentile(y.T,97.5,axis=1)
ymean=np.percentile(y.T,50,axis=1)
import fenics as fe
mean_fun=fe.Function(domain_equ.function_space)
mean_fun.vector()[:]=ymean
yyy1_fun=fe.Function(domain_equ.function_space)
yyy1_fun.vector()[:]=yyy1
yyy2_fun=fe.Function(domain_equ.function_space)
yyy2_fun.vector()[:]=yyy2
truth_fun_shape = fe.Expression('exp(-20*(x[0]-0.3)*(x[0]-0.3)-20*(x[1]-0.3)*(x[1]-0.3))+exp(-20*(x[0]-0.7)*(x[0]-0.7)-20*(x[1]-0.7)*(x[1]-0.7))', degree=1)
truth_fun = fe.interpolate(truth_fun_shape, domain_equ.function_space)
truth_fun_vec=truth_fun.vector()[:]
pCN=np.load('pCNdir.npy')[100000:,:]
s=pCN[:,50]
def ESS(trace):
        trace = trace - np.mean(trace)
        corr_full = scipy.signal.correlate(trace, trace, "full")
        corr_plus = corr_full[int(corr_full.size / 2):]
        corr_plus /= corr_plus[0]
        corr_plus_even = corr_plus[::2]
        corr_plus_odd = corr_plus[1::2]
        if len(corr_plus_even) != len(corr_plus_odd):
            corr_plus_even = np.delete(corr_plus_even, -1)
        tem_corr = corr_plus_even + corr_plus_odd
        for i in range(len(tem_corr)):
            if tem_corr[i] < 0:
                tem_idx = i
                break
        tem_corr = tem_corr[0:tem_idx]
        ESS_divisor = 2 * sum(tem_corr) - 1
        n = len(trace)
        ESS = n / ESS_divisor
        return ESS
ess_pcn=ESS(s)
s1=y[:,50]
ess_flow=ESS(s1)
print('ESS of pCN with '+str(len(s))+' samples:',ess_pcn)
print('ESS of flow with '+str(len(s1))+' samples:' ,ess_flow)