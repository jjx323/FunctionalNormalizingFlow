import torch
import numpy as np
from core.model import Domain2D
import matplotlib.pyplot as plt
torch.manual_seed(11)
from core.probability import GaussianElliptic2
## domain for solving PDE
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
from common_flows import (PlanarFlow,HouseholderFlow,project_transformFlow,
                          sylvesterFlow,NormalizingFlow,model_train)
from post import Darcyflow_negtive_log_post
from torch.optim.lr_scheduler import StepLR
path='model_dir/'
if __name__ == "__main__":
    exact_log_density = lambda z: - Darcyflow_negtive_log_post(z)
    flow_style=sylvesterFlow
    for flow_length in [5]:
        flow = NormalizingFlow(flow_style,flow_length)
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
        loss = model_train(flow, optimizer, 5000, exact_log_density, 30,scheduler)
        if (flow_style==PlanarFlow):
            name='PlanarFlow5.zip'
        if (flow_style==HouseholderFlow):
            name='HouseholderFlow.zip'
        if (flow_style==project_transformFlow):
            name='project_transformFlow.zip'
        if (flow_style==sylvesterFlow):
            name='sylvesterFlow.zip'
        torch.save(flow.state_dict(), path+name)
plt.plot(loss[1:])
plt.savefig('PIC/loss')
plt.close()







