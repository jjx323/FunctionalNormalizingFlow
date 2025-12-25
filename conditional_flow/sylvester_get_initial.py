import torch
import numpy as np
torch.manual_seed(88)
from core.model import Domain2D
from core.probability import GaussianElliptic2
from core.misc import trans2spnumpy, spnumpy2sptorch,sptorch2spnumpy
import fenics as fe
import matplotlib.pyplot as plt
import sys, os
import numpy as np
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
import scipy.sparse.linalg as spsl
import torch
import torch.nn as nn
from tqdm import tqdm
import random
from core.misc import load_expre, smoothing
from common_PDE import EquSolver, ModelDarcyFlow
noise_level = 0.05
from core.noise import NoiseGaussianIID
DATA_DIR = './DATA/'
## loading coordinates of the measurement points
measurement_points = np.load(DATA_DIR + "measurement_points_2D.npy")
## setting the forward problem
f_expre = load_expre(DATA_DIR + 'f_2D.txt')

## setting the Model
##############################################################################
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

#eig_values=np.load('./eigvec/'+'eig_vals.npy')
Psi_r=np.load('./RESULT/'+'eig_vec_'+str(dim)+'.npy')
Psi_r_torch=torch.from_numpy(Psi_r)
Psi_r_torch=Psi_r_torch.to(torch.float32)

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
        return r, diag_, R, dia_
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
        return r, diag_, R

    def forward(self, z, data):
        r1, diag1, R1_, dia1_=self.constrained_r1(data)
        r2, diag2, R2_ = self.constrained_r2(data)
        z = z.to(torch.float32)
        b_net = self.h1(self.layerb1(data))
        b_net = self.h1(self.layerb2(b_net))
        b_net = self.h1(self.layerb3(b_net))
        b_net = self.h1(self.layerb4(b_net))
        b_net = self.h1(self.layerb5(b_net))
        b_=b_net
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
        return x, log_det, R1_,R2_,dia1_,b_
class NormalizingFlow(nn.Module):
    def __init__(self, flow_length):
        super().__init__()

        self.layers = nn.Sequential(
            *(sylvesterFlow() for _ in range(flow_length)))

    def forward(self, z,data):
        R1net = []
        R2net = []
        DIAnet=[]
        Bnet=[]
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian,R1_,R2_,dia_,b_ = layer(z,data)
            R1_np=R1_.detach().numpy()
            R2_np = R2_.detach().numpy()
            dia_np = dia_.detach().numpy()
            b_np = b_.detach().numpy()
            R1net.append(R1_np)
            R2net.append(R2_np)
            DIAnet.append(dia_np)
            Bnet.append(b_np)
            log_jacobians += log_jacobian
        return z, log_jacobians,R1net,R2net,DIAnet,Bnet


project_adj=np.load(DATA_DIR + 'project_adj.npy')/400

mean=np.load(DATA_DIR + 'pro_mean.npy')
var=np.load(DATA_DIR + 'pro_var.npy')
project_adj=(project_adj-mean)/np.sqrt(var)

project_adj_torch=torch.from_numpy(project_adj)
project_adj_torch=project_adj_torch.to(torch.float32)
def getinitial(s):
    mm = s
    project_adj_torch_i = project_adj_torch[mm, :]
    flow_length = 5
    flow = NormalizingFlow(flow_length)
    flow.load_state_dict(torch.load('conditional_sylvester'))
    from prior import prior_sample_torch
    x = prior_sample_torch(num=100, prior=prior_measure)
    y, sss, Rnet1, Rnet2, DIAnet1, Bnet1 = flow.forward(x, project_adj_torch_i)
    Rnet_np1 = np.array(Rnet1)
    Rnet_np2 = np.array(Rnet2)
    DIAnet_np1 = np.array(DIAnet1)
    Bnet_np1 = np.array(Bnet1)

    np.save('sylvester DATA/R1net'+str(mm), Rnet_np1)
    np.save('sylvester DATA/R2net'+str(mm), Rnet_np2)
    np.save('sylvester DATA/DIAnet'+str(mm), DIAnet_np1)
    np.save('sylvester DATA/Bnet'+str(mm), Bnet_np1)



