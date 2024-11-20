import torch
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)
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
m=20
Psi_r_torch=Psi_r_torch[:,:m]
################################################################################
from common_flows_dis_inv import (PlanarFlow,HouseholderFlow,project_transformFlow,
                          sylvesterFlow,NormalizingFlow,model_train)
################################################################################
from post import post
from torch.optim.lr_scheduler import StepLR
path='model_dir/'
if __name__ == "__main__":
    exact_log_density = lambda z: - post(z)
    flow_style=project_transformFlow
    for flow_length in [5]:
        flow = NormalizingFlow(flow_style,flow_length,dim=dim)
        #flow.load_state_dict(torch.load('TEST'))
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.01,weight_decay=0.001)
        scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
        loss = model_train(flow, optimizer, 5000, exact_log_density, 30,scheduler)
        if (flow_style==PlanarFlow):
            name='PlanarFlow.zip'
        if (flow_style==HouseholderFlow):
            name='HouseholderFlow1.zip'
        if (flow_style==project_transformFlow):
            name='project_transformFlow.zip'
        if (flow_style==sylvesterFlow):
            name='sylvesterFlow.zip'

        torch.save(flow.state_dict(), path+name)
plt.plot(loss[1:])
plt.savefig('PIC/loss')
plt.close()







