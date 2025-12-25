import torch
import numpy as np
from prior import prior_sample_torch
import fenics as fe
from core.model import Domain2D
from core.probability import GaussianElliptic2
###########################################################
for equ_nx in (15,20,30,40,50,70):
    dim=(equ_nx+1)**2
    k=1
    domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
    prior_measure = GaussianElliptic2(
        domain=domain_equ, alpha=(0.1 / k), a_fun=(1.0 / k), theta=1.0, boundary='Neumann'
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
    Psi_r=np.load('./RESULT/'+'eig_vec'+'_'+str(dim)+'.npy')
    Psi_r_torch=torch.from_numpy(Psi_r)
    Psi_r_torch=Psi_r_torch.to(torch.float32)
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
        length = 24
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
    num=500
    x=prior_sample_torch(num=num,prior=prior_measure)
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
    flow_samples=y
    mean = np.mean(flow_samples, axis=0)
    x=np.array([0.5,0.5])
    y=np.array([0.5,0.5])
    coordinates1 = np.array([x])
    mean1 = [mean_fun(point) for point in coordinates1]
    mean1=mean1[0]
    coordinates2 = np.array([y])
    mean2 = [mean_fun(point) for point in coordinates2]
    mean2=mean2[0]
    c=0
    for i in range(num):
        fun=fe.Function(domain_equ.function_space)
        fun.vector()[:]=flow_samples[i, :]
        f1=[fun(point) for point in coordinates1]
        f2=[fun(point) for point in coordinates2]
        f1=f1[0]
        f2=f2[0]
        delta1=f1-mean1
        delta2=f2-mean2
        c_x_y=delta1*delta2
        c=c+c_x_y
    c=c/num
    print('discreate level=',dim,'c(x,y)=',c,'where x=',x,'y=',y)
    truth_fun = fe.Expression('exp(-20*(x[0]-0.3)*(x[0]-0.3)-20*(x[1]-0.3)*(x[1]-0.3))+exp(-20*(x[0]-0.7)*(x[0]-0.7)-20*(x[1]-0.7)*(x[1]-0.7))', degree=1)
    truth_fun = fe.interpolate(truth_fun, domain_equ.function_space)
    f_vec=truth_fun.vector()[:]
    L2=(f_vec-mean)@(M@(f_vec-mean))
    print('discreate level=',dim,'L2_error=',L2)
    print('##################')





