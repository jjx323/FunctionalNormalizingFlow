# Code for Functional Normalizing Flows
## Overview
The program relies on FEniCS (Version 2019.1.0) and PyTorch (Version 1.10.0). These codes provide implementations of the method, named Functionl Normalizing Flow, proposed in the paper **https://arxiv.org/abs/2411.13277**

1.The whole code is divided into three parts, namely 1D problem, 2D problem and conditional flow, which correspond to the 1D problem, the 2D problem in Section 4,  and the numerical experiment results of conditional functional normalizing flow in Section 5. Some files with identical names appear in multiple folders. While these files share similar functionalities, their specific implementations may vary across different contexts.

1. Directory **core** contains the main functions and classes that are useful for implementing the algorithms. Specifically,
- **probability.py**: This file contains classes of GaussianElliptic2[The Gaussian measure implemented by finite element methods based on solving elliptic differential equations used for generating samples and also contains the functionality of evaluate the gradient and Hessian operators];
- **noise.py**: This file contains the class NoiseGaussianIID.
- **model.py**: This file contains the class Domain and two classes Domain2D and Domain1D inherit from the parent class Domain; contains the parent class ModelBase of the model classes employed in specific examples, the class ModelBase incorporates the components of the domain, prior, equation solver, and noise. 
- **linear_eq_solver.py**: contains the function cg_my which is our implementation of the conjugate gradient algorithm for solving linear equations. 
- **eigensystem.py**: This file contains the function double_pass which is our implementation of an algorithm for calculating eigensystem. 
- **approximate_sample.py**: This file contains the class LaplaceApproximate, which can be used to compute the Laplace approximation of the posterior measures. 
- **optimizer.py**: This file contains the class OptimBase[incorporate an implementation of armijo_line_search can be employed for each optimizer]; the class GradientDescent[an implementation of the gradient descent algorithm]; the class NewtonCG[an implementation of the Newton conjugate gradient algorithm]. 
- **sample.py**: This file contains the class pCN, which is a type of discrete invariant Markov chain Monte Carlo sampling algorithm. 
- **Plotting.py**: This file contains some functions that can draw functions generated by FEniCS. 
- **misc.py**: This file contains functions of trans2spnumpy, trans2sptorch, spnumpy2sptorch, sptorch2spnumpy, and sptensor2cude, which will be useful for transferring sparse matrixes to different forms required for doing calculations in numpy, pytorch, and FEniCS. This file also contains the function construct_measurement_matrix, which will be used for generating a sparse matrix S. The matrix S times a function generated by FEniCS to get the values at the measurement points.
- **commen_flows.py** This file contains four different flow models, namely functional Householder flow, functional projected transformation flow, functional planar flow, functional Sylvester flow.
- **commen_flows_dis_inv.py** This file contains four different flow models: functional Householder flow, functional projected transformation flow, functional planar flow, and functional Sylvester flow. Each model accepts an input dimension 'dim', facilitating experiments across various discrete settings. This folder is dedicated to investigating the model’s discrete invariance property.
- **commen_PDEs.py** This file contains a collection of solvers tailored to the specific partial differential equation model under consideration.
- **ESS.py** This file enables the calculation and visualization of effective sample size (ESS) for samples generated by various algorithms.
- **cov.py** This file can show the covariance function obtained by different algorithms.
- **generate_eig.py** This file generates eigenfunctions corresponding to a priori measures of varying discrete dimensions.
- **pCN.py** This file can be used to conduct pCN algorithm.
- **pCN_plot.py** This file can be used to plot the results of the pCN algorithm.
- **prior.py** This file contains some code of the priori measure.


## 1D problem
1.This folder contains the code corresponding to the Simple Smooth Problem. We will briefly introduce the role of some files and explain how to run this code.

- **discreate_invariance.py** This file contains the code for discrete invariance experiments.
- **experiment.py** This file contains the code for training functional normalizing flow.
- **generate_data.py** This file contains the code to generate the real function and the measurement data under the influence of different noise.
- **model_plot.py** This file contains the code to plot the result of functional normalizing flow.
- **post.py** This file contains some code about posteriori measure.
### Workflows

To obtain the corresponding results, follow these steps:

Run **generate_eig.py** to generate the eigenfunctions of priori measure of different discrete dimensions.

Run **generate_data.py** to generate the real function and the measurement data under the influence of different noise.

Run **experiment.py** to train functional normalizing flow.

Run **pCN.py** to conduct pCN algorithm.

Run **pCN_plot.py** to plot the results of the pCN algorithm.

Run **model_plot.py** to plot the result of functional normalizing flow.

Run **discrete_invariance.py** to do discrete invariance experiment.

Run **ESS.py** to calculate the effective sample size (ESS) for samples generated by various algorithms.

Run **cov.py** to show the covariance function obtained by different algorithms.

## 2D problem
1.This folder contains the code corresponding to the Darcy flow problem. We will briefly introduce the role of some files and explain how to run this code.

- **discreate_invariance.py** This file contains the code for discrete invariance experiments.
- **experiment.py** This file contains the code for training functional normalizing flow.
- **generate_data.py** This file contains the code to generate the real function and the measurement data under the influence of different noise.
- **model_plot.py** This file contains the code to plot the result of functional normalizing flow.
- **post.py** This file contains some code about posteriori measure.
### Workflows

To obtain the corresponding results, follow these steps:

Run **generate_eig.py** to generate the eigenfunctions of priori measure of different discrete dimensions.

Run **generate_data.py** to generate the real function and the measurement data under the influence of different noise.

Run **experiment.py** to train functional normalizing flow.

Run **pCN.py** to conduct pCN algorithm.

Run **pCN_plot.py** to plot the results of the pCN algorithm.

Run **model_plot.py** to plot the result of functional normalizing flow.

Run **discrete_invariance.py** to do discrete invariance experiment.

Run **ESS.py** to calculate the effective sample size (ESS) for samples generated by various algorithms.

Run **cov.py** to show the covariance function obtained by different algorithms.

## conditional flow
1.This folder contains the corresponding code of conditional normalizing flow. We will briefly introduce the role of some files and explain how to run this code.

- **conditional_data_generation.py** This file contains the code to generate training dataset and test dataset.
- **conditional_normalizing_flows_sylvester.py** This file contains the code for training conditional functional normalizing flow.
- **conditional_sylvester_plot.py** This file contains the code to plot the results of conditional functional normalizing flow.
- **Darcyflow_post.py** This file contains some code about posteriori measure.
- **sylvester_get_initial.py** This file contains the code to get a good initial value through the trained conditional network.
- **retrain_sylvester.py** This file contains the code for further training functional normalizing flow based on the good initial we got from the conditional network.
- **plot_retrain_sylvester.py** This file contains the code to plot the results of retrained functional normalizing flow.
### Workflows

To obtain the corresponding results, follow these steps:

Run **generate_eig.py** to generate the eigenfunctions of priori measure of different discrete dimensions.

Run **conditional_data_generation.py** to generate training dataset and test dataset.

Run **conditional_normalizing_flows_sylvester.py** to train conditional functional normalizing flow.

Run **conditional_sylvester_plot.py** to plot the results of conditional functional normalizing flow.

Run **retrain_sylvester.py** to conduct retrained functional normalizing flow.

Run **plot_retrain_sylvester.py** to plot the results of retrained functional normalizing flow.
