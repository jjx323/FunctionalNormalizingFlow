import torch
import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
from matplotlib.ticker import MaxNLocator
torch.manual_seed(1)
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
m=20
Psi_r_torch=Psi_r_torch[:,:m]
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
flow_length=5
flow = NormalizingFlow(model_style,flow_length,dim=dim)
flow.load_state_dict(torch.load(path+model_load))
from prior import prior_sample_torch
x=prior_sample_torch(num=1000,prior=prior_measure)
y,sss=flow.forward(x)
y=y.detach().numpy()
yyy1=np.percentile(y.T,2.5,axis=1)
yyy2=np.percentile(y.T,97.5,axis=1)
y_mean=np.percentile(y.T,50,axis=1)
x_x=1-np.arange(0,dim,1)/(dim-1)
plt.fill_between(x_x,yyy1,yyy2,alpha=0.2,color='blue')
DATA_DIR = './DATA/'
truth_fun = fe.Expression ( 'exp(-50*(x[0]-0.3)*(x[0]-0.3))-exp(-50*(x[0]-0.7)*(x[0]-0.7))', degree=1)
truth_fun = fe.interpolate(truth_fun, domain_equ.function_space)
f_vec=truth_fun.vector()[:]
####################################################################################################
plt.plot(x_x,f_vec,label = 'Truth')
plt.plot(x_x,y_mean,'r--',label = 'Mean')
plt.legend(loc=1,fontsize='large')
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # x轴主要刻度的数量
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
plt.xticks(fontsize=16)  # x轴刻度的字体大小
plt.yticks(fontsize=16)
plt.savefig('PIC/flow_samples')
plt.close()
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
cov_flow=cov_flow/num

fig1=plt.imshow(cov_flow)
plt.colorbar(fig1)
plt.savefig('PIC/cov_flow')
plt.close()

cov_pcn=np.load('DATA/cov.npy')
fig2=plt.imshow(cov_pcn)
plt.colorbar(fig2)
plt.savefig('PIC/cov_pcn')
plt.close()

delta=cov_pcn-cov_flow
fig3=plt.imshow(delta)
plt.colorbar(fig3)
plt.savefig('PIC/delta')
plt.close()