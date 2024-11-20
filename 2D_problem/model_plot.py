import torch
import numpy as np
import matplotlib.pyplot as plt
from core.model import Domain2D
torch.manual_seed(1)
from core.probability import GaussianElliptic2
from prior import prior_sample_torch
import fenics as fe
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
################################################################################
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
##############################################################
##############################################################
flow_length=length
flow = NormalizingFlow(model_style,flow_length,dim=dim)
flow.load_state_dict(torch.load(path+model_load))
x=prior_sample_torch(num=200,prior=prior_measure)
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
truth_fun_shape = fe.Expression('exp(-20*(x[0]-0.3)*(x[0]-0.3)-20*(x[1]-0.3)*(x[1]-0.3))+exp(-20*(x[0]-0.7)*(x[0]-0.7)-20*(x[1]-0.7)*(x[1]-0.7))', degree=1)
truth_fun = fe.interpolate(truth_fun_shape, domain_equ.function_space)
truth_fun_vec=truth_fun.vector()[:]


mean = np.mean(y, axis=0)
print(mean.shape)
# print(pCN.shape)
diag = np.zeros((dim))
num = y.shape[0]
for i in range(num):
        vec = y[i, :] - mean
        vec = np.reshape(vec, (1, dim))
        M = vec.T @ vec
        diagi = np.diag(M)
        diag = diagi + diag
diag = diag / num

fig = fe.plot(truth_fun, label='Truth')
plt.colorbar(fig)
plt.title("Truth")
plt.savefig('PIC/truth')
plt.close()

fun = fe.Function(domain_equ.function_space)
fun.vector()[:] = np.array(np.sqrt(diag))
fig = fe.plot(fun)
plt.title('Point-wise Variance Field')
plt.colorbar(fig)
plt.savefig('PIC/Point_wise_Variance_Field')
plt.close()

fig = fe.plot(mean_fun, label='Estimate')
plt.colorbar(fig)
plt.legend()
plt.title('Mean')
plt.savefig('PIC/flow_mean')
plt.close()
