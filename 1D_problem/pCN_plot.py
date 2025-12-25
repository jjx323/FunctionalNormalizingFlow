import fenics as fe
import matplotlib.pyplot as plt
import sys, os
import numpy as np
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
plt.rcParams['font.family'] = 'Times New Roman'
from matplotlib.ticker import MaxNLocator
noise_level = 0.05
from core.noise import NoiseGaussianIID
DATA_DIR = './DATA/'
equ_nx = np.load('equ_nx.npy')
domain_equ = Domain1D(n=equ_nx, mesh_type='CG', mesh_order=1)
f = fe.Function(domain_equ.function_space)
f.vector()[:] = np.load(DATA_DIR + 'truth_vec.npy')
f_vec=f.vector()[:]
equ_nx = np.load('equ_nx.npy')
dim=equ_nx+1
####################################################################################################
pCN=np.load(DATA_DIR+"pCNsamples.npy")[100000:,:]
print(pCN.shape)
y=pCN
yyy1=np.percentile(y.T,2.5,axis=1)
yyy2=np.percentile(y.T,97.5,axis=1)
y_mean=np.percentile(y.T,50,axis=1)
x_x1=1-np.arange(0,dim,1)/(dim-1)
x_x2=1-np.arange(0,10001,1)/(10001-1)
plt.fill_between(x_x1,yyy1,yyy2,alpha=0.2,color='blue')
plt.plot(x_x2,f_vec,label = 'Truth')
plt.plot(x_x1,y_mean,'r--',label = 'Mean')
plt.legend(loc=1,fontsize='large')
#plt.title('Estimate')
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('PIC/pcn')
plt.close()
pCN=y
mean = np.mean(pCN, axis=0)
np.save('mean',mean)
print(mean.shape)
cov=np.zeros((dim,dim))
num = pCN.shape[0]
for i in range(num):
        vec = pCN[i, :] - mean
        vec = np.reshape(vec, (1, dim))
        M = vec.T @ vec
        covi=M
        cov=cov+covi
cov=cov/num
print(cov)
np.save('DATA/cov',cov)
fig2=plt.imshow(cov)
plt.colorbar(fig2)
plt.savefig('PIC/cov')
plt.close()











