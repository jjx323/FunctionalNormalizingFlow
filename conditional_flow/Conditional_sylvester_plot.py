import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)
from core.model import Domain2D
from prior import prior_sample_torch
from matplotlib.ticker import MaxNLocator
import fenics as fe
plt.rcParams['font.family'] = 'Times New Roman'
from core.probability import GaussianElliptic2
from core.misc import trans2spnumpy, spnumpy2sptorch,sptorch2spnumpy
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
    Mpt = torch.sparse_coo_tensor(torch.LongTensor([Mcoo.row.tolist(), Mcoo.col.tolist()]),
                                   torch.FloatTensor(Mcoo.data.astype(float)))
    Mpt = Mpt.to(torch.float32)
    return Mpt
M_torch=csr_torch(M)
Psi_r=np.load('./RESULT/'+'eig_vec_'+str(dim)+'.npy')
Psi_r_torch=torch.from_numpy(Psi_r)
Psi_r_torch=Psi_r_torch.to(torch.float32)
ww=Psi_r_torch@torch.rand(20)
m=20
Psi_r_torch=Psi_r_torch[:,:m]
#"""

d_dim=20
eignum=20
class sylvesterFlow(nn.Module):
    def __init__(self):
        super().__init__()
        q=0.1

        self.layerR1 = torch.nn.Linear(d_dim, 10)
        self.layerR2 = torch.nn.Linear(10, 10)
        self.layerR3 = torch.nn.Linear(10, 10)
        self.layerR4 = torch.nn.Linear(10, 10)
        self.layerR5 = torch.nn.Linear(10, 20 * 20)

        self.layerR_1 = torch.nn.Linear(d_dim, 10)
        self.layerR_2 = torch.nn.Linear(10, 10)
        self.layerR_3 = torch.nn.Linear(10, 10)
        self.layerR_4 = torch.nn.Linear(10, 10)
        self.layerR_5 = torch.nn.Linear(10, 20 * 20)

        self.layerda1 = torch.nn.Linear(d_dim, 10)
        self.layerda2 = torch.nn.Linear(10, 10)
        self.layerda3 = torch.nn.Linear(10, 10)
        self.layerda4 = torch.nn.Linear(10, 10)
        self.layerda5 = torch.nn.Linear(10, 20)

        self.layerb1 = torch.nn.Linear(d_dim, 10)
        self.layerb2 = torch.nn.Linear(10, 10)
        self.layerb3 = torch.nn.Linear(10, 10)
        self.layerb4 = torch.nn.Linear(10, 10)
        self.layerb5 = torch.nn.Linear(10, 20)
        self.h=nn.Tanh()
        self.h1=nn.Tanh()
        self.h_prime = lambda z: (1 - self.h(z) ** 2)
        self.name='sylvesterFlow'

    def constrained_r1(self,data):
        R = self.h1(self.layerR1(data))
        R = self.h1(self.layerR2(R))
        R = self.h1(self.layerR3(R))
        R = self.h1(self.layerR4(R))
        R = self.h1(self.layerR5(R))
        R=torch.reshape(R,(20,20))

        dia_ = self.h1(self.layerda1(data))
        dia_ = self.h1(self.layerda2(dia_))
        dia_ = self.h1(self.layerda3(dia_))
        dia_ = self.h1(self.layerda4(dia_))
        dia_ = self.h1(self.layerda5(dia_))

        r_ = torch.triu(R, diagonal=1)
        diag_ = self.h(dia_) + 0.1 ** 5
        diag = torch.diag(diag_)
        r = diag + r_
        return r, diag_
    def constrained_r2(self,data):
        R = self.h1(self.layerR_1(data))
        R = self.h1(self.layerR_2(R))
        R = self.h1(self.layerR_3(R))
        R = self.h1(self.layerR_4(R))
        R = self.h1(self.layerR_5(R))
        R=torch.reshape(R,(20,20))

        r_=torch.triu(R, diagonal=1)
        diag_=torch.ones(m)
        diag=torch.diag(diag_)
        print(m)
        r=diag+r_
        return r,diag_

    def forward(self, z, data):
        r1,diag1=self.constrained_r1(data)
        r2, diag2 = self.constrained_r2(data)
        z = z.to(torch.float32)
        b_net = self.h1(self.layerb1(data))
        b_net = self.h1(self.layerb2(b_net))
        b_net = self.h1(self.layerb3(b_net))
        b_net = self.h1(self.layerb4(b_net))
        b_net = self.h1(self.layerb5(b_net))
        b_net=torch.reshape(b_net,(20,1))
        hidden_units = r2@Psi_r_torch.T@(M_torch@z.T) + b_net
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
    def forward(self, z,data):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(z,data)
            log_jacobians += log_jacobian
        return z, log_jacobians
##############################################################
DATA_DIR = './DATA/'
m_vec = np.load(DATA_DIR + 'm_data.npy')
d = np.load(DATA_DIR + 'noise_data_before_reshape_0.05.npy')
project_adj=np.load(DATA_DIR + 'project_adj.npy')/400
mean=np.load(DATA_DIR + 'pro_mean.npy')
var=np.load(DATA_DIR + 'pro_var.npy')
project_adj=(project_adj-mean)/np.sqrt(var)
d_torch = torch.from_numpy(d)
project_adj_torch=torch.from_numpy(project_adj)
d_torch = d_torch.to(torch.float32)
project_adj_torch=project_adj_torch.to(torch.float32)
for i in range(1):
    mm=i+0
    m_vec_i=m_vec[mm,:]
    d_clean_torch_i=d_torch[mm,:]
    project_adj_torch_i=project_adj_torch[mm,:]
    flow_length=5
    flow = NormalizingFlow(flow_length)
    flow.load_state_dict(torch.load('conditional_sylvester'))
    x=prior_sample_torch(num=100,prior=prior_measure)
    y,sss=flow.forward(x,project_adj_torch_i)
    y=y.detach().numpy()
    yyy1=np.percentile(y.T,2.5,axis=1)
    yyy2=np.percentile(y.T,97.5,axis=1)
    y_mean=np.percentile(y.T,50,axis=1)
    plt.subplot(1, 5, 1)
    plt.title('truth')
    map_fun = fe.Function(domain_equ.function_space)
    map_fun.vector()[:] = m_vec_i
    fig1 = fe.plot(map_fun)
    plt.colorbar(fig1)
    plt.subplot(1, 5, 2)
    plt.title('estimate_mean')
    mean_fun=fe.Function(domain_equ.function_space)
    mean_fun.vector()[:]=y_mean
    fig2=fe.plot(mean_fun)
    plt.colorbar(fig2)
    plt.subplot(1, 5, 3)
    plt.title('down')
    mean_fun=fe.Function(domain_equ.function_space)
    mean_fun.vector()[:]=yyy1
    fig3=fe.plot(mean_fun)
    plt.colorbar(fig3)
    plt.subplot(1, 5, 4)
    plt.title('up')
    mean_fun=fe.Function(domain_equ.function_space)
    mean_fun.vector()[:]=yyy2
    fig4=fe.plot(mean_fun)
    plt.colorbar(fig4)
    plt.subplot(1, 5, 5)
    pCN=y
    mean = np.mean(pCN, axis=0)
    print(mean.shape)
    diag = np.zeros((dim))
    num = y.shape[0]
    for i in range(num):
        vec = pCN[i, :] - mean
        vec = np.reshape(vec, (1, dim))
        M = vec.T @ vec
        diagi = np.diag(M)
        diag = diagi + diag
    diag = diag / num
    fun = fe.Function(domain_equ.function_space)
    fun.vector()[:] = np.array(np.sqrt(diag))
    fig = fe.plot(fun)
    plt.title('covariance')
    plt.colorbar(fig)
    plt.show()
##################################################################################
    plt.subplot(1, 1, 1)
    fig = fe.plot(mean_fun, label='Truth')
    cbar = plt.colorbar(fig)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar.locator = MaxNLocator(nbins=6)
    cbar.ax.tick_params(labelsize=16)
    plt.legend()
    plt.show()
    plt.subplot(1, 1, 1)
    plt.title('Mean')
    mean_fun=fe.Function(domain_equ.function_space)
    mean_fun.vector()[:]=y_mean
    fig2=fe.plot(mean_fun)
    plt.colorbar(fig2)
    plt.show()
    plt.subplot(1, 1, 1)
    pCN=y
    mean = np.mean(pCN, axis=0)
    print(mean.shape)
    diag = np.zeros((dim))
    num = y.shape[0]
    for i in range(num):
        vec = pCN[i, :] - mean
        vec = np.reshape(vec, (1, dim))
        M = vec.T @ vec
        diagi = np.diag(M)
        diag = diagi + diag
    diag = diag / num
    fun = fe.Function(domain_equ.function_space)
    fun.vector()[:] = np.array(np.sqrt(diag))
    fig = fe.plot(fun)
    plt.title('Point-wise Variance Field')
    plt.colorbar(fig)
    plt.show()



