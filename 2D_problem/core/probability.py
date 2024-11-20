import numpy as np
import fenics as fe
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from core.misc import trans2spnumpy, construct_measurement_matrix
np.random.seed(11)
tip=np.load('RESULT/tip.npy')
#############################################################################
class GaussianElliptic2(object):
    # '''
    # prior Gaussian probability measure N(m, C)
    # C^{-1/2}: an elliptic operator -\alpha\nabla(\cdot\Theta\nabla \cdot) + a(x) Id
    #
    # Ref: A computational framework for infinite-dimensional Bayesian inverse problems
    # part I: The linearized case, with application to global seismic inversion,
    # SIAM J. Sci. Comput., 2013
    # '''
    def __init__(self, domain, alpha=1.0, a_fun=fe.Constant(1.0), theta=1.0, 
                 mean_fun=None, tensor=False, boundary='Neumann', bc_val=0.0,
                 use_LU=True):
        
        """
        boundary (string): 'Neumann' or 'Dirichlet'
        mean_fun (fenics.Function or None): None(set the mean function to zero)

        use_LU (True or False): 
                take LU decomposition of the sparse matrix K and M, then Mx=b and Kx=b 
                are all solved by lu.solve directly that may be faster than spsolve. 
                (splu in scipy may take longer time than spsolve, however, if we need 
                 to generate many samples, we only need to run splu once)
        """
        assert type(alpha) == type(1.0) or type(alpha) == type(np.array(1.0)) \
            or type(alpha) == type(1)
        assert boundary == 'Neumann' or boundary == 'Dirichlet', \
                "boundary must be 'Neumann' or 'Dirichlet'"
        
        self.domain = domain
        self.function_space_dim = self.domain.function_space.dim()
        self._alpha = alpha
        if type(a_fun) == type(1.0) or type(a_fun) == type(np.array(1.0)) or type(a_fun) == type(1):
            a_fun = fe.Constant(a_fun)
        self._a_fun = fe.interpolate(a_fun, domain.function_space)
        self._tensor = tensor

        if self._tensor == False:
            assert type(theta) == type(1.0) or type(theta) == type(np.array(1.0)) or type(theta) == type(1)
            self._theta = fe.interpolate(fe.Constant(theta), self.domain.function_space)
        elif self._tensor == True:
            self._theta = fe.as_matrix(((fe.interpolate(theta[0], self.domain.function_space), \
                                         fe.interpolate(theta[1], self.domain.function_space)), \
                                        (fe.interpolate(theta[2], self.domain.function_space), \
                                         fe.interpolate(theta[3], self.domain.function_space))))
        if mean_fun is None:
            self.mean_fun = fe.interpolate(fe.Expression("0.0", degree=2), self.domain.function_space)
        else:
            self.mean_fun = fe.interpolate(mean_fun, self.domain.function_space)
        self._mean_vec = self.mean_fun.vector()[:]

        u = fe.TrialFunction(self.domain.function_space)
        v = fe.TestFunction(self.domain.function_space)
        aa = fe.Constant(self._alpha)*fe.inner(self._theta*fe.grad(u), fe.grad(v))*fe.dx \
            + fe.inner(self._a_fun*u, v)*fe.dx
        self.K_ = fe.assemble(aa)
        bb = fe.inner(u, v)*fe.dx
        self.M_ = fe.assemble(bb)

        self.bc = boundary
        self.bc_val = bc_val
        self.boundary(self.M_)
        self.boundary(self.K_)
        self.index_boundary = None
        if self.bc == "Dirichlet":
            self.boundary_index()
        self.K = trans2spnumpy(self.K_)
        self.M = trans2spnumpy(self.M_)
        lamped_elements = np.array(np.sum(self.M, axis=1)).flatten()
        self.M_lamped_half = sps.diags(np.sqrt(lamped_elements))
        self.M_lamped_half_ = fe.assemble(fe.inner(u, v)*fe.dx)
        self.M_lamped_half_.zero()
        v = fe.Vector()
        self.M_.init_vector(v, 1)
        v[:] = np.sqrt(lamped_elements)
        self.M_lamped_half_.set_diagonal(v)
        self.n_eqx=self.M.shape[0]
        self.eig_vec = np.load('RESULT/eig_vec_' + str(self.n_eqx) + '.npy')
        eig_val=np.zeros(20)

        for i in range(20):
            eig_val[i]=1/(a_fun+self._alpha*(tip[i,2]*(np.pi)**2))
        self.eig_val=eig_val

        self.temp0 = fe.Function(self.domain.function_space)
        self.temp1 = fe.Function(self.domain.function_space)
        self.temp2 = fe.Function(self.domain.function_space)
        
        ## using LU decomposition to accelerate computation
        self.use_LU = use_LU
        self.luM, self.luK = None, None
        if self.use_LU == True:
            self.luM = spsl.splu(self.M.tocsc()) 
            self.luK = spsl.splu(self.K.tocsc())
            
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, al):
        assert type(al) == type(1.0) or type(al) == type(np.array(1.0)) or type(al) == type(1)
        self._alpha = al
        self.generate_K()
        
    @property
    def a_fun(self):
        return self._a_fun
        
    @a_fun.setter
    def a_fun(self, al):
        self._a_fun = fe.interpolate(al, self.domain.function_space)
        self.generate_K()
        
    @property
    def mean_vec(self):
        self._mean_vec = np.array(self.mean_fun.vector()[:])
        return self._mean_vec
    
    @mean_vec.setter 
    def mean_vec(self, mean):
        self._mean_vec = np.array(mean)
        self.mean_fun.vector()[:] = self._mean_vec
    
    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, th):
        if self._tensor == False:
            assert type(th) == type(1.0) or type(th) == type(np.array(1.0)) or type(th) == type(1)
            self._theta = th
        elif self._tensor == True:
            self._theta = fe.as_matrix(((fe.interpolate(th[0], self.domain.function_space), \
                                         fe.interpolate(th[1], self.domain.function_space)), \
                                        (fe.interpolate(th[2], self.domain.function_space), \
                                         fe.interpolate(th[3], self.domain.function_space))))
        self.generate_K()
        
    def update_mean_fun(self, mean_fun_vec):
        self.mean_fun.vector()[:] = mean_fun_vec
        
    def boundary(self, b):
        if self.bc == 'Dirichlet':
            def boundary(x, on_boundary):
                return on_boundary
            bc = fe.DirichletBC(self.domain.function_space, fe.Constant(self.bc_val), boundary)
            bc.apply(b)
    
    def boundary_index(self):
        a = fe.Function(self.domain.function_space)
        a.vector()[:] = 1.0
        v_ = fe.TestFunction(self.domain.function_space)
        aa = fe.assemble(a*v_*fe.dx)
        bb = fe.assemble(a*v_*fe.dx)
        
        def boundary(x, on_boundary):
            return on_boundary
        bc = fe.DirichletBC(self.domain.function_space, fe.Constant(self.bc_val), boundary)
        
        bc.apply(aa)
        self.index_boundary = (aa[:] != bb[:])
    
    def boundary_vec(self, b):
        if self.bc == 'Dirichlet':
            b[self.index_boundary] = self.bc_val
        return b
        
    def generate_K(self):
        u = fe.TrialFunction(self.domain.function_space)
        v = fe.TestFunction(self.domain.function_space)
        theta = self._theta
        a = fe.Constant(self._alpha)*fe.inner(theta*fe.grad(u), fe.grad(v))*fe.dx \
            + fe.Constant(self._alpha)*fe.inner(self._a_fun*u, v)*fe.dx
        self.K_ = fe.assemble(a)
        self.boundary(self.K_)
        self.K = trans2spnumpy(self.K_)
        if self.use_LU == True:
            self.luK = spsl.splu(self.K)
        return self.K
    
    def generate_M(self):
        u = fe.TrialFunction(self.domain.function_space)
        v = fe.TestFunction(self.domain.function_space)
        a = fe.inner(u, v)*fe.dx
        self.M_ = fe.assemble(a)
        self.boundary(self.M_)
        self.M = trans2spnumpy(self.M_)
        if self.use_LU == True:
            self.luM = spsl.splu(self.M)
        return self.M
    
    def generate_sample(self, method='numpy'):
        r = np.random.randn(20)
        fix = r * self.eig_val
        sample = (self.eig_vec @ fix)
        sample = self.mean_fun.vector()[:] + sample
        return np.array(sample)

    def generate_sample_zero_mean(self, method='numpy'):
        
        assert self.K is not None 
        assert self.M_lamped_half is not None
        
        fun = fe.Function(self.domain.function_space)
        
        if method == 'numpy':
            n = np.random.normal(0, 1, (self.function_space_dim,))
            b = self.M_lamped_half@n
            #b = self.M_half@n

            self.boundary_vec(b)
            if self.use_LU == False:
                fun_vec = spsl.spsolve(self.K, b)
            elif self.use_LU == True:
                fun_vec = self.luK.solve(b)
            else:
                raise NotImplementedError("use_LU must be True or False")
            return np.array(fun_vec)
        elif method == 'FEniCS':
            n_ = fe.Vector()
            self.M_lamped_half_.init_vector(n_, 1)
            n_.set_local(np.random.normal(0, 1, (self.function_space_dim,)))
            # fe.solve(self.K_, fun.vector(), self.M_half_*n_, 'cg', 'ilu')
            fe.solve(self.K_, fun.vector(), self.M_lamped_half_*n_)
            return np.array(fun.vector()[:])
        else:
            assert False, "method must be 'FEniCS' or 'numpy'"

    def evaluate_Cxf(self, f):
        tmp = self.evaluate_SqrtCxf(f)
        ans = self.evaluate_SqrtCxf(tmp)
        return np.array(ans)
    
    def evaluate_SqrtCxf(self, f):
        tmp = self.luK.solve(self.M@f)
        return np.array(tmp)
            
    def evaluate_CM_inner(self, u_vec, v_vec, method='numpy'):
        """
        evaluate (C^{-1/2}u, C^{-1/2}v)
        """
        
        assert type(u_vec) == np.ndarray
        assert type(v_vec) == np.ndarray

        if method == 'numpy':
            temp1 = u_vec - self.mean_fun.vector()[:]
            temp2 = v_vec - self.mean_fun.vector()[:]
            if self.use_LU == False:
                return temp1@(self.K.T)@spsl.spsolve(self.M, self.K@temp2)
            elif self.use_LU == True:
                val = self.luM.solve(self.K@temp2)
                val = temp1@(self.K.T)@val
                return val
            else:
                raise NotImplementedError("use_LU must be True or False")
        elif method == 'FEniCS':
            self.temp0.vector()[:] = v_vec - self.mean_fun.vector()[:]
            fe.solve(self.M_, self.temp1.vector(), (self.K_*self.temp0.vector()))
            self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
            self.temp0.vector()[:] = u_vec - self.mean_fun.vector()[:]
            return (self.temp0.vector()).inner(self.temp2.vector())
        else:
            assert False, "method must be 'FEniCS' or 'numpy'"
    
    def evaluate_grad(self, u_vec, method='numpy'):
        # '''
        # calculate the gradient vector at u_vec
        # the input vector should be in $\mathbb{R}_{M}^{n}$
        # the output vector is in $v1\in\mathbb{R}_{M}^{n}$
        # '''
        assert type(u_vec) is np.ndarray
        
        if method == 'numpy':
            res = u_vec - self.mean_fun.vector()[:]
            if self.use_LU == False:
                grad_vec = (self.K.T)@spsl.spsolve(self.M, self.K@res)
                grad_vec = spsl.spsolve(self.M, grad_vec)
            elif self.use_LU == True:
                grad_vec = self.luM.solve(self.K@res)
                grad_vec = (self.K.T)@grad_vec 
                grad_vec = self.luM.solve(grad_vec)
            else:
                raise NotImplementedError("use_LU must be True or False")
            return grad_vec
        elif method == 'FEniCS':
            self.temp0.vector()[:] = u_vec - self.mean_fun.vector()[:]
            fe.solve(self.M_, self.temp1.vector(), (self.K_*self.temp0.vector()))
            self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
            fe.solve(self.M_, self.temp1.vector(), self.temp2.vector())
            return self.temp1.vector()[:]
        else:
            assert False, "method must be 'FEniCS' or 'numpy'"
        
    def evaluate_hessian_vec(self, u_vec, method='numpy'):
        # '''
        # evaluate HessianMatrix*u_vec
        # the input vector should be in $\mathbb{R}_{M}^{n}$,
        # the output vector is in $\mathbb{R}_{M}^{n}$
        # '''
        assert type(u_vec) is np.ndarray
        
        if method == 'numpy':
            if self.use_LU == False:
                temp = (self.K.T)@spsl.spsolve(self.M, self.K@u_vec)
                temp = spsl.spsolve(self.M, temp)
            elif self.use_LU == True:
                temp = self.luM.solve(self.K@u_vec)
                temp = (self.K.T)@temp
                temp = self.luM.solve(temp)
            else:
                raise NotImplementedError("use_LU must be True or False")
            return np.array(temp)
        elif method == 'FEniCS':
            self.temp0.vector()[:] = self.K@u_vec
            fe.solve(self.M_, self.temp1.vector(), self.temp0.vector())
            self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
            fe.solve(self.M_, self.temp1.vector(), self.temp2.vector())
            return self.temp1.vector()[:]
        else:
            assert False, "method must be 'FEniCS' or 'numpy'"
        
    def precondition(self, m_vec):
        # temp = spsl.spsolve(self.K, self.M@spsl.spsolve((self.K).T, self.M@m_vec))
        ## Usually, algorithms need a symmetric matrix.
        ## Here, we drop the last M in prior, e.g., 
        ## For GaussianElliptic2, we calculate K^{-1}MK^{-1}m_vec instead of K^{-1}MK^{-1}M m_vec 
        # temp = spsl.spsolve(self.K, self.M@spsl.spsolve(self.K, self.M@m_vec))
        if self.use_LU == False:
            temp = spsl.spsolve(self.K, self.M@spsl.spsolve(self.K, m_vec))
        elif self.use_LU == True:
            temp = self.luK.solve(m_vec)
            temp = self.M@temp
            temp = self.luK.solve(temp)
        else:
            raise NotImplementedError("use_LU must be True or False")
        return np.array(temp)
    
    def precondition_inv(self, m_vec):
        if self.use_LU == False:
            temp = self.K@spsl.spsolve(self.M, self.K@m_vec)
        elif self.use_LU == True:
            temp = self.luM.solve(self.K@m_vec)
            temp = self.K@temp
        else:
            raise NotImplementedError("use_LU must be True or False")
        # temp = spsl.spsolve(self.M, temp)
        return np.array(temp)

    def pointwise_variance_field(self, xx, yy, method="numpy"):
        # '''
        # This function evaluate the pointwise variance field in a finite element discretization
        #
        # Parameters
        # ----------
        # xx : list
        #     [(x_1,y_1), \cdots, (x_N, y_N)]
        # yy : list
        #     [(x_1,y_1), \cdots, (x_M, y_M)]
        #
        # Returns: variance field c(xx, yy), a matrix NxM
        # -------
        # None.
        #
        # '''
        #
        SN = construct_measurement_matrix(np.array(xx), self.domain.function_space)
        SM = construct_measurement_matrix(np.array(xx), self.domain.function_space)
        SM = np.array((SM.T).todense())
        
        if method == "FEniCS":
            raise NotImplementedError
        elif method == "numpy":
            if self.use_LU == False:
                val = SN@spsl.spsolve(self.K, self.M@spsl.spsolve(self.K, SM))
            elif self.use_LU == True:
                val = self.luK.solve(SM)
                val = self.M@val
                val = self.luK.solve(val)
                val = SN@val 
            else:
                raise NotImplementedError("use_LU must be True or False")
                
            if type(val) == type(self.M):
                val = val.todense()
            return np.array(val)
        else:
            assert False, "method must be numpy or FEniCS (FEniCS has not yet been implemented)"


############################################################################################