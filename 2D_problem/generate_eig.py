import torch
import numpy as np
torch.manual_seed(1)
from core.misc import trans2spnumpy, construct_measurement_matrix
import fenics as fe
from core.model import Domain2D
m=20
A=[]
for i in range(10):
    for j in range(10):
        w=i**2+j**2
        a=np.array([i,j,w])
        A.append(a)
A=np.array(A)
A = A[np.argsort(A[:, 2])]
A_2=A[:m,:]
tip=np.save('RESULT/tip.npy',A_2)
tip=np.load('RESULT/tip.npy')
print(tip)
for equ_nx in (10,15,20,30,40,50,70,90,100):
    dim=(equ_nx+1)**2
    eigs=np.zeros((dim,20))
    domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)
    u = fe.TrialFunction(domain_equ.function_space)
    v = fe.TestFunction(domain_equ.function_space)
    bb = fe.inner(u, v) * fe.dx
    M_ = fe.assemble(bb)
    M = trans2spnumpy(M_)
    for w in range(20):
        tip1 = tip[w, 0]
        tip2 = tip[w, 1]
        print(tip1,tip2)
        fun=fe.Expression('cos('+str(tip1)+'*pi*x[0])*cos('+str(tip2)+'*pi*x[1])', degree=1)
        eig = fe.interpolate(fun, domain_equ.function_space)
        y=eig.vector()[:]
        y=y/np.sqrt(y@(M@y))
        eigs[:,w]=y
    np.save('RESULT/eig_vec_'+str(dim)+'.npy',eigs)




