import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)
import fenics as fe
import scipy
###############################################################################
from core.model import Domain1D
from core.probability import GaussianElliptic2
equ_nx = 100
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
a=3
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
flow_length=5
flow = NormalizingFlow(model_style,flow_length,dim=dim)
flow.load_state_dict(torch.load(path+model_load))
from prior import prior_sample_torch
x=prior_sample_torch(num=2000,prior=prior_measure)
y,sss=flow.forward(x)
y=y.detach().numpy()

pCN=y

s=np.load('DATA/pCNtrace.npy')[100000:]
def ESS(trace):
        trace = trace - np.mean(trace)
        corr_full = scipy.signal.correlate(trace, trace, "full")
        corr_plus = corr_full[int(corr_full.size / 2):]  # delete the minus lag part
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
        # print("ESS_divisor=", ESS_divisor, ", ESS=", ESS)
        # corr_FFT = corr_plus
        return ESS
#s=y[:,98]
ess_pcn=ESS(s)
s1=y[:,50]
ess_flow=ESS(s1)
print('ESS of pCN with '+str(len(s))+' samples:',ess_pcn)
print('ESS of flow with '+str(len(s1))+' samples:' ,ess_flow)




