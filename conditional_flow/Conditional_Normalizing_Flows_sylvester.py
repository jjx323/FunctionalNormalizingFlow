import torch
import numpy as np
import math
torch.manual_seed(88)
###############################################################################
from core.model import Domain2D
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

import fenics as fe
import matplotlib.pyplot as plt
import sys, os
import numpy as np
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import torch
import torch.nn as nn
from tqdm import tqdm
from core.misc import load_expre, smoothing
from common_PDE import EquSolver, ModelDarcyFlow
noise_level = 0.05
from core.noise import NoiseGaussianIID
DATA_DIR = './DATA/'
measurement_points = np.load(DATA_DIR + "measurement_points_2D.npy")
f_expre = load_expre(DATA_DIR + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f_expre, degree=2), domain_equ.function_space)
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
from prior import prior_sample_torch
from prior import negtive_log_prior
def train(flow, optimizer, nb_epochs, log_density, batch_size,data):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        z0=prior_sample_torch(num=batch_size,prior=prior_measure)
        zk, log_jacobian = flow(z0,data)
        zk=zk.to(torch.float64)
        flow_log_density = negtive_log_prior(z0) - log_jacobian
        exact_log_density = log_density(zk)
        reverse_kl_divergence = (flow_log_density - exact_log_density).mean()
        optimizer.zero_grad()
        loss = reverse_kl_divergence
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())
    return training_loss
def Train_conditional(flow,optimizer,nb_epochs, log_density, KL_num,data,truth_fun,project_adj):
    total_datasize=data.shape[0]
    training_loss = []
    truth_fun_torch=torch.from_numpy(truth_fun)
    batch_size=10
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data,truth_fun_torch,project_adj), batch_size=batch_size,
                                               shuffle=True)
    for eps in tqdm(range(nb_epochs)):
        total_kl=0
        ss=total_datasize/batch_size
        q=1
        for data_i,fun_torch_i,project_adj_i in train_loader:
            print(q)
            q=q+1
            fun_numpy_i=np.array(fun_torch_i)
            datasize=data_i.shape[0]
            kl = 0
            for j in range(datasize):
                data_i_j=data_i[j,:]
                project_adj_i_j=project_adj_i[j,:]
                fun_numpy_i_j_vec=fun_numpy_i[j,:]
                data_i_j = data_i_j.to(torch.float32)
                project_adj_i_j=project_adj_i_j.to(torch.float32)
                data_i_j_numpy=np.array(data_i_j)
                truth_fun_i_j = fe.Function(domain_equ.function_space)
                truth_fun_i_j.vector()[:] = fun_numpy_i_j_vec
                target_log_density=log_density(data_i_j_numpy,truth_fun_i_j)
                z0 = prior_sample_torch(num=KL_num, prior=prior_measure)
                zk, log_jacobian = flow(z0, project_adj_i_j)
                zk = zk.to(torch.float64)
                # Evaluate the exact and approximated densities
                flow_log_density = negtive_log_prior(z0) - log_jacobian
                exact_log_density = target_log_density(zk)
                # Compute the loss
                reverse_kl_divergence_i = (flow_log_density - exact_log_density).mean()
                x = torch.tensor(0.0, requires_grad=True)
                if (math.isnan(reverse_kl_divergence_i)):
                    reverse_kl_divergence_i = x
                K=200000.0
                x1=torch.tensor(K, requires_grad=True)
                if ((reverse_kl_divergence_i.detach().numpy())>K):
                    reverse_kl_divergence_i = x1
                kl=kl+reverse_kl_divergence_i
            optimizer.zero_grad()
            loss = kl / datasize
            print('loss',loss)
            total_kl=total_kl+loss
            #print('mean_kl', loss)
            loss.backward()
            loss_numpy=loss.detach().numpy()
            training_loss.append(loss_numpy)
            nn.utils.clip_grad_norm_(flow.parameters(), max_norm=0.5, norm_type=2)
            optimizer.step()
        total_kl=total_kl/ss
        print('total+kl',total_kl)
    return training_loss

from Darcyflow_post import PostFun
post=PostFun.apply
def Darcyflow_negtive_log_post(x,model):
    return post(x,model)
def Exact_log_density(data,truth_fun):
    equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, \
                           points=measurement_points )
    noise_level_ = noise_level * max(np.absolute(data))
    noise1 = NoiseGaussianIID(dim=len(measurement_points))
    noise1.set_parameters(variance=noise_level_ ** 2)
    model = ModelDarcyFlow(
        d=data, domain_equ=domain_equ, prior=prior_measure,
        noise=noise1, equ_solver=equ_solver
    )
    exact_log_density = lambda z: - Darcyflow_negtive_log_post(z, model)
    return exact_log_density

if __name__ == "__main__":
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
    flow_length=5
    Loss=np.zeros((1,1))
    ww=5000
    i=1
    d_clean_i=d[i:i+ww,:]
    m_vec_i=m_vec[i:i+ww,:]
    d_clean_torch_i=d_torch[i:i+ww,:]
    project_adj_torch_i=project_adj_torch[i:i+ww,:]
    ##########################################################################
    flow = NormalizingFlow(flow_length)
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.001,weight_decay=1e-4)
    loss = Train_conditional(flow, optimizer, 10, Exact_log_density, 20, d_clean_torch_i,m_vec_i,project_adj_torch_i)
    loss=np.array(loss)
    Loss=np.append(Loss,loss)
    torch.save(flow.state_dict(), 'conditional_sylvester')
    Loss=np.array(Loss)
    np.save(DATA_DIR + 'Loss', Loss)
    plt.plot(Loss[1:])
    plt.show()


    











