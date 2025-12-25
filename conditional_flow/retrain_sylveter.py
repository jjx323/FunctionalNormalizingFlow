import torch
import fenics as fe
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
torch.manual_seed(88)
from core.model import Domain2D
from core.probability import GaussianElliptic2
from Darcyflow_post import PostFun
from common_PDE import EquSolver, ModelDarcyFlow
from core.noise import NoiseGaussianIID
from core.misc import load_expre, smoothing
from core.misc import trans2spnumpy, spnumpy2sptorch,sptorch2spnumpy
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
############################################################
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
class sylvesterFlow(nn.Module):
    def __init__(self,init_value_R1,init_value_R2,init_value_dia,init_value_b):
        super().__init__()
        init_value_R1 = torch.from_numpy(init_value_R1)
        init_value_R2 = torch.from_numpy(init_value_R2)
        init_value_dia = torch.from_numpy(init_value_dia)
        init_value_b = torch.from_numpy(init_value_b)
        init_value_b = torch.reshape(init_value_b, (m, 1))
        q=0.1
        self.R1 = nn.Parameter(init_value_R1)
        self.R2 = nn.Parameter(init_value_R2)
        self.dia1_=nn.Parameter(init_value_dia)
        self.b = nn.Parameter(init_value_b)
        self.h=nn.Tanh()
        self.h_prime = lambda z: (1 - self.h(z) ** 2)
        self.name='sylvesterFlow'
    def constrained_r1(self):
        r_=torch.triu(self.R1, diagonal=1)
        diag_=self.h(self.dia1_)+0.1**15
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
from prior import negtive_log_prior
from prior import prior_sample_torch
def train(flow, optimizer, nb_epochs, log_density, batch_size):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        z0=prior_sample_torch(num=batch_size,prior=prior_measure)
        zk, log_jacobian = flow(z0)
        zk=zk.to(torch.float64)
        flow_log_density = negtive_log_prior(z0) - log_jacobian
        exact_log_density = log_density(zk)
        # Compute the loss
        reverse_kl_divergence = (flow_log_density - exact_log_density).mean()
        optimizer.zero_grad()
        loss = reverse_kl_divergence
        print(loss)
        x = torch.tensor(0.0, requires_grad=True)
        if (math.isnan(reverse_kl_divergence)):
            reverse_kl_divergence = x
        if ((reverse_kl_divergence.detach().numpy()) > 200000):
            reverse_kl_divergence = x
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), max_norm=0.5, norm_type=2)  # 使用第二种裁剪方式。
        optimizer.step()
        training_loss.append(loss.item())
    return training_loss
flow_length=5
truth_fun_shape = fe.Expression('exp(-20*(x[0]-0.3)*(x[0]-0.3)-20*(x[1]-0.3)*(x[1]-0.3))+exp(-20*(x[0]-0.7)*(x[0]-0.7)-20*(x[1]-0.7)*(x[1]-0.7))', degree=1)
truth_fun = fe.interpolate(truth_fun_shape, domain_equ.function_space)
truth_fun_vec=truth_fun.vector()[:]
DATA_DIR = './DATA/'
d_total = np.load(DATA_DIR + 'noise_data_before_reshape_0.05.npy')
noise_level=0.05
measurement_points = np.load(DATA_DIR + "measurement_points_2D.npy")
f_expre = load_expre(DATA_DIR + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f_expre, degree=2), domain_equ.function_space)
from sylvester_get_initial import getinitial
for r in range(2):
    num=r+5000
    d=d_total[num,:]
    truth_fun_vec=np.load(DATA_DIR + 'm_data.npy')[num,:]
    truth_fun.vector()[:]=truth_fun_vec
    noise_level_ = noise_level * max(np.absolute(d))
    noise = NoiseGaussianIID(dim=len(measurement_points))
    noise.set_parameters(variance=noise_level_ ** 2)
    equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, points=measurement_points)
    model = ModelDarcyFlow(
        d=d, domain_equ=domain_equ, prior=prior_measure,
        noise=noise, equ_solver=equ_solver
    )
    post = PostFun.apply
    def Darcyflow_negtive_log_post(x):
        return post(x, model)
    aaaaa=Darcyflow_negtive_log_post
    device = 'cuda';  index = 1
    getinitial(num)
    R1net = np.load('sylvester DATA/R1net'+str(num)+'.npy')
    R2net = np.load('sylvester DATA/R2net' + str(num) + '.npy')
    Dnet = np.load('sylvester DATA/DIAnet'+str(num)+'.npy')
    Bnet = np.load('sylvester DATA/Bnet'+str(num)+'.npy')
    class NormalizingFlow_100(nn.Module):
        def __init__(self, flow_length):
            super().__init__()
            self.layers = nn.Sequential(
                *(sylvesterFlow(R1net[i, :, :],R2net[i, :, :], Dnet[i, :], Bnet[i, :]) for i in range(flow_length)))
        def forward(self, z):
            log_jacobians = 0
            for layer in self.layers:
                z, log_jacobian = layer(z)
                log_jacobians += log_jacobian
            return z, log_jacobians

    flow_length = 5
    flow = NormalizingFlow_100(flow_length)
    exact_log_density = lambda z: - aaaaa(z)
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.005,weight_decay=0.001)
    loss = train(flow, optimizer, 1000, exact_log_density, 20)
    torch.save(flow.state_dict(), 'retrain/'+'TEST_retrain_sylvester_'+str(num))
#"""
