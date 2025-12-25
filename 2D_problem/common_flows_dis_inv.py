import numpy as np
import torch
import torch.nn as nn
from core.model import Domain2D
from core.probability import GaussianElliptic2
from prior import prior_sample_torch,negtive_log_prior
from tqdm import tqdm

def csr_torch(matrix):
    Mcoo = matrix.tocoo()
    Mpt = torch.sparse.FloatTensor(torch.LongTensor([Mcoo.row.tolist(), Mcoo.col.tolist()]),
                                   torch.FloatTensor(Mcoo.data.astype(float)))
    Mpt = Mpt.to(torch.float32)
    return Mpt

m=20

class PlanarFlow(nn.Module):
    def __init__(self,dim):
        super().__init__()

        k = 1
        equ_nx = int(np.sqrt(dim) - 1)
        Psi_r = np.load('./RESULT/' + 'eig_vec' + '_' + str(dim) + '.npy')
        Psi_r_torch = torch.from_numpy(Psi_r)
        Psi_r_torch = Psi_r_torch.to(torch.float32)
        m = 20
        self.Psi_r_torch = Psi_r_torch[:, :m]

        domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
        prior_measure = GaussianElliptic2(
            domain=domain_equ, alpha=(0.1 / k), a_fun=(1.0 / k), theta=1.0, boundary='Neumann'
        )
        M = prior_measure.M
        self.M_torch = csr_torch(M)

        q=0.1
        self.u = nn.Parameter(q*torch.rand(m))
        self.w = nn.Parameter(q*torch.rand(m))
        self.b = nn.Parameter(q*torch.rand(1))
        self.h = nn.Tanh()
        self.name='PlanarFlow'
        self.h_prime = lambda z: (1 - self.h(z) ** 2)
    def constrained_u(self):
        u_fun1=self.Psi_r_torch@self.u
        w_fun=self.Psi_r_torch@self.w
        wu_fun_inner=w_fun@(self.M_torch@u_fun1)
        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        return u_fun1+ (m(wu_fun_inner) - wu_fun_inner) * (w_fun / (w_fun@(self.M_torch@w_fun) + 1e-15))
    def forward(self, z):
        u_fun=self.constrained_u()
        w_fun = self.Psi_r_torch @ self.w
        z = z.to(torch.float32)
        hidden_units = w_fun@(self.M_torch@z.T) + self.b
        x = z + u_fun.unsqueeze(0) * self.h(hidden_units).unsqueeze(-1)
        psi = self.h_prime(hidden_units).unsqueeze(0) * w_fun.unsqueeze(-1)
        log_det = torch.log((1 + u_fun@(self.M_torch@psi)).abs() + 1e-15)
        return x, log_det
class HouseholderFlow(nn.Module):
    def __init__(self,dim):
        super().__init__()
        k = 1
        equ_nx = int(np.sqrt(dim) - 1)
        Psi_r = np.load('./RESULT/' + 'eig_vec' + '_' + str(dim) + '.npy')
        Psi_r_torch = torch.from_numpy(Psi_r)
        Psi_r_torch = Psi_r_torch.to(torch.float32)
        m = 20
        self.Psi_r_torch = Psi_r_torch[:, :m]

        domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
        prior_measure = GaussianElliptic2(
            domain=domain_equ, alpha=(0.1 / k), a_fun=(1.0 / k), theta=1.0, boundary='Neumann'
        )
        M = prior_measure.M
        self.M_torch = csr_torch(M)

        q=0.1
        self.v = nn.Parameter(q*torch.rand(m))
        self.b = nn.Parameter(q * torch.rand(1))
        self.name='HouseholderFlow'
    def constrained_v(self):
        v_fun1=self.Psi_r_torch@self.v
        v_fun_inner=v_fun1@(self.M_torch@v_fun1)
        v_fun_L2=v_fun1/torch.sqrt(v_fun_inner)
        return v_fun_L2
    def forward(self, z):
        v_fun=self.constrained_v()
        z = z.to(torch.float32)
        aa=-0.5*(z@(self.M_torch@v_fun.T)+self.b)
        Fk = v_fun.unsqueeze(0)*aa.unsqueeze(-1)
        x = z + Fk
        log_det = 0.5
        return x, log_det
class project_transformFlow(nn.Module):
    def __init__(self,dim):
        super().__init__()
        k = 1
        equ_nx = int(np.sqrt(dim) - 1)
        Psi_r = np.load('./RESULT/' + 'eig_vec' + '_' + str(dim) + '.npy')
        Psi_r_torch = torch.from_numpy(Psi_r)
        Psi_r_torch = Psi_r_torch.to(torch.float32)
        m = 20
        self.Psi_r_torch = Psi_r_torch[:, :m]

        domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
        prior_measure = GaussianElliptic2(
            domain=domain_equ, alpha=(0.1 / k), a_fun=(1.0 / k), theta=1.0, boundary='Neumann'
        )
        M = prior_measure.M
        self.M_torch = csr_torch(M)

        q=0.1
        self.R = nn.Parameter(q*torch.rand(m,m))
        self.dia_=nn.Parameter(q*torch.rand(m))
        self.b = nn.Parameter(q*torch.rand(m,1))
        self.h=nn.Tanh()
        self.name='project_transformFlow'
    def constrained_r(self):
        r_=torch.triu(self.R, diagonal=1)
        diag_=self.h(self.dia_)+0.1**5
        diag=torch.diag(diag_)
        r=diag+r_
        return r,diag_
    def forward(self, z):
        r,diag=self.constrained_r()
        z = z.to(torch.float32)
        hidden_units = self.Psi_r_torch.T@(self.M_torch@z.T) + self.b
        x = z + (self.Psi_r_torch@(r@hidden_units)).T
        diag1=diag+1
        log_det = torch.log(torch.prod(diag1))
        return x, log_det
class sylvesterFlow(nn.Module):
    def __init__(self,dim):
        super().__init__()
        k=1
        equ_nx = int(np.sqrt(dim) - 1)
        Psi_r = np.load('./RESULT/' + 'eig_vec' + '_' + str(dim) + '.npy')
        Psi_r_torch = torch.from_numpy(Psi_r)
        Psi_r_torch = Psi_r_torch.to(torch.float32)
        m = 20
        self.Psi_r_torch = Psi_r_torch[:, :m]

        domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
        prior_measure = GaussianElliptic2(
            domain=domain_equ, alpha=(0.1 / k), a_fun=(1.0 / k), theta=1.0, boundary='Neumann'
        )
        M = prior_measure.M
        self.M_torch = csr_torch(M)

        q=0.1
        self.R1 = nn.Parameter(q*torch.rand(m,m))
        self.R2 = nn.Parameter(q*torch.rand(m,m))
        self.dia1_=nn.Parameter(q*torch.rand(m))
        self.b = nn.Parameter(q*torch.rand(m,1))
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

        hidden_units = r2@self.Psi_r_torch.T@(self.M_torch@z.T) + self.b
        x = z + (self.Psi_r_torch@r1@self.h(hidden_units)).T
        psi = self.h_prime(hidden_units)
        ans=(diag1*diag2)
        ans=torch.reshape(ans,(m,1))
        w=ans*psi
        w1=w+1
        det=torch.prod(w1,dim=0)
        log_det = torch.log(det)
        return x, log_det

class NormalizingFlow(nn.Module):
    def __init__(self,flow, flow_length,dim):
        super().__init__()
        self.layers = nn.Sequential(
            *(flow(dim) for _ in range(flow_length)))
    def forward(self, z):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(z)
            log_jacobians += log_jacobian
        return z, log_jacobians
