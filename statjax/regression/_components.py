# -*- coding: utf-8 -*-
# Author:shihua
# Designer:shihua
# Coder:shihua
# Email:15021408795@163.com
# License: PiggyStudio
# Copyright (c) 2026 PiggyStudio. All rights reserved.



'''
Module Introduction
-------------------

This is a collection of concrete regression, support vector regression, binary-classification and multiclass-classification implementation classes. Each algorithm is mounted as an independent Component under BaseRegression, preserving the overall Base / Component / Algo architecture

- Design mode:

    (1) Combination mode

- Key points:

    (1) JAX technology stack (thin QR decomposition, triangular back substitution, thin SVD decomposition, singular value shrinkage, soft thresholding operator, Gauss-Seidel coordinate sweep, rank-1 residual update, IRLS, Fisher scoring, column-pivoted QR, Armijo backtracking, normal log-CDF, JIT compilation where shapes permit)

    (2) Reference-category Softmax, row-major parameter unfolding, per-sample Cholesky whitening, Kronecker-product design matrices and jointly coupled multinomial Newton systems

    (3) Scale-constrained Cholesky covariance parameters, class-specific negative-orthant contrasts, GHK truncated-normal simulation, fixed-common-random-number MSL and automatic-differentiation BFGS for correlated multinomial Probit utilities

    (4) RBF kernel matrices, compact SVR dual variables, KKT working-set selection, exact piecewise-quadratic pair subproblems, O(n) gradient updates and fully compiled lax.while_loop SMO iteration

- Main functions:

    (1) Linear regression, regularized regression and sparse regression

    (2) Binary Logistic and Probit probability inference and classification

    (3) Multinomial Logistic and multinomial Probit K-class probability inference and argmax classification

    (4) Exact small and medium dense RBF support vector regression with sparse support-vector inference

Usage examples
--------------
.. code-block:: python
    :linenos:

    from statjax.regression import HouseholderOLSComponent,SVDRidgeComponent,CDLassoComponent,QRLogisticComponent,QRProbitComponent,QRMultinomialLogisticComponent,GHKMultinomialProbitComponent,SMOSVRComponent

    ### Train the OLS model and solve the optimal regression coefficients
    ols_instance = HouseholderOLSComponent()
    x_hat,metrics = ols_instance.train(A=A_train,b=b_train)
    ### Batch prediction on the test set
    y_pred = ols_instance.reasoning(A_test=A_test)

    ### Train the ridge regression model and solve the L2 regularized optimal regression coefficients
    ridge_instance = SVDRidgeComponent(lam=1.0)
    w_hat,metrics = ridge_instance.train(A=A_train,b=b_train)
    ### Batch prediction on the test set
    y_pred = ridge_instance.reasoning(A_test=A_test)

    ### Train the LASSO regression model and solve the L1 regularized sparse optimal regression coefficients
    lasso_instance = CDLassoComponent(lam=0.1)
    w_hat,metrics = lasso_instance.train(A=A_train,b=b_train)
    ### Batch prediction on the test set (the intercept is automatically added back)
    y_pred = lasso_instance.reasoning(A_test=A_test)

    ### Train binary Logistic regression on labels containing only 0 and 1
    logistic_instance = QRLogisticComponent(l2=1e-6)
    beta_hat,metrics = logistic_instance.train(A=A_binary_train,b=b_binary_train)
    ### Return [P(y=0|x),P(y=1|x)] and threshold-based binary classes
    probabilities = logistic_instance.predict_proba(A_test=A_binary_test)
    y_pred = logistic_instance.reasoning(A_test=A_binary_test)

    ### Train Probit on a caller-prepared design matrix containing any intercept column
    probit_instance = QRProbitComponent()
    beta_hat,metrics = probit_instance.train(A=A_probit_train,b=b_binary_train)
    probabilities = probit_instance.predict_proba(A_test=A_probit_test)
    y_pred = probit_instance.reasoning(A_test=A_probit_test)

    ### Train multinomial Logistic regression with explicit K and integer labels 0,...,K-1
    multinomial_instance = QRMultinomialLogisticComponent(n_classes=3,l2=1e-4)
    theta_hat,metrics = multinomial_instance.train(A=A_multiclass_train,b=b_multiclass_train)
    ### Return K-class probabilities in label order and argmax class predictions
    probabilities = multinomial_instance.predict_proba(A_test=A_multiclass_test)
    y_pred = multinomial_instance.reasoning(A_test=A_multiclass_test)

    ### Multinomial Probit uses A(n,D,p), D=J-1 and the final class as reference
    mnp_instance = GHKMultinomialProbitComponent(num_draws=128)
    theta_hat,metrics = mnp_instance.train(A=A_mnp_train,b=b_mnp_train)
    ### Inference draws are independent of the fixed training draws
    probabilities = mnp_instance.predict_proba(A_test=A_mnp_test)
    y_pred = mnp_instance.reasoning(A_test=A_mnp_test)

    ### RBF SVR returns the complete compact dual vector and continuous predictions
    svr_instance = SMOSVRComponent(C=10.0,epsilon=0.05,gamma=0.5)
    beta_hat,metrics = svr_instance.train(A=A_svr_train,b=b_svr_train)
    ### The support vectors and dual coefficients are retained for sparse inference
    y_pred = svr_instance.reasoning(A_test=A_svr_test)

Class Description
-----------------
(1)TestRegression: This is a regression test placeholder component implementation class, the main function is to verify the framework link, the main technical inheritance

(2)HouseholderOLSComponent: This is an OLS linear regression component implementation class based on Householder QR decomposition, the main function is to solve the least squares optimal regression coefficients and batch prediction, the main technical JAX technology stack (thin QR decomposition, triangular back substitution, JIT compilation)

(3)SVDRidgeComponent: This is a ridge regression component implementation class based on SVD decomposition, the main function is to solve the L2 regularized optimal regression coefficients and batch prediction, the main technical JAX technology stack (thin SVD decomposition, singular value shrinkage, JIT compilation)

(4)CDLassoComponent: This is a LASSO regression component implementation class based on coordinate descent, the main function is to solve the L1 regularized sparse optimal regression coefficients and batch prediction, the main technical JAX technology stack (soft thresholding operator, Gauss-Seidel coordinate sweep, rank-1 residual update, lax.fori_loop compiled control flow, JIT compilation)

(5)QRLogisticComponent: This is a binary Logistic regression component based on IRLS and column-pivoted Householder QR decomposition, the main functions are stable L2-regularized training, numerical-rank truncation, Armijo-protected updates, two-class probability inference and threshold-based classification

(6)QRProbitComponent: This is a binary Probit regression component based on Fisher scoring and column-pivoted Householder QR decomposition, the main functions are stable unregularized maximum-likelihood training, explicit numerical-rank failure, Armijo-protected updates, normal-CDF probability inference and threshold-based classification

(7)QRMultinomialLogisticComponent: This is a multinomial Logistic regression component based on reference-category Softmax, jointly coupled IRLS, per-sample Cholesky whitening and handwritten column-pivoted Householder QR, the main functions are L2-regularized training, numerical-rank truncation, Armijo-protected updates, K-class probability inference and argmax classification

(8)GHKMultinomialProbitComponent: This is a correlated multinomial Probit component based on scale-constrained Cholesky covariance parameters, GHK choice-probability simulation, fixed-draw MSL and BFGS, with independent inference draws, J-class probabilities and argmax classification

(9)SMOSVRComponent: This is an RBF support vector regression component based on compact-dual second-order SMO, exact pair subproblems, KKT convergence checks, sparse support-vector inference and a fully compiled lax.while_loop

References
----------
- [Ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)

- [Householder transformation](https://en.wikipedia.org/wiki/Householder_transformation)

- [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition)

- [Ridge regression](https://en.wikipedia.org/wiki/Ridge_regression)

- [Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)

- [Lasso (statistics)](https://en.wikipedia.org/wiki/Lasso_(statistics))

- [Coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent)

- [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)

- [Multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)

- [Probit model](https://en.wikipedia.org/wiki/Probit_model)

- [Multinomial probit](https://en.wikipedia.org/wiki/Discrete_choice#Multinomial_probit)

- [BFGS algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)

- [Iteratively reweighted least squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)

- [Backtracking line search](https://en.wikipedia.org/wiki/Backtracking_line_search)

- [Support vector regression](https://en.wikipedia.org/wiki/Support_vector_machine#Regression)

- [Sequential minimal optimization](https://en.wikipedia.org/wiki/Sequential_minimal_optimization)

- [Radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)

- [Karush-Kuhn-Tucker conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)

- StatJAX001 OLS Design Document `"StatJAX001 OLS SH V003"<https://github.com/redblue0216/statjax>`_

- StatJAX002 Ridge Regression Design Document `"StatJAX002 Ridge SH V002"<https://github.com/redblue0216/statjax>`_

- StatJAX003 LASSO Design Document `"StatJAX003 LASSO SH V001"<https://github.com/redblue0216/statjax>`_

- StatJAX004 Logistic Design Document `"StatJAX004 Logistic SH V001"<https://github.com/redblue0216/statjax>`_

- StatJAX005 Probit Design Document `"StatJAX005 Probit SH V001"<https://github.com/redblue0216/statjax>`_

- StatJAX006 Multinomial Logistic Design Document `"StatJAX006 Multinomial Logistic SH V001"<https://github.com/redblue0216/statjax>`_

- StatJAX007 Multinomial Probit Design Document `"StatJAX007 Multinomial Probit SH V001"<https://github.com/redblue0216/statjax>`_

- StatJAX008 Support Vector Regression Design Document `"StatJAX008 SVM SH V001"<https://github.com/redblue0216/statjax>`_
'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
from functools import lru_cache
from typing import NamedTuple
from statjax.regression._base import BaseRegression 
### Algorithm package
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import qr,solve_triangular
from jax.scipy.optimize import minimize
from jax.scipy.special import log_ndtr,logsumexp,ndtr,ndtri
from jax.scipy.stats import norm
from jax.numpy.linalg import lstsq
### Project Package - Algorithm Components



### Global precision configuration: enable float64 before any JAX array creation, matching LAPACK precision
jax.config.update("jax_enable_x64",True)



####### Classes and Functions ###################################################################################################################################################
###
### class:TestRegression
### ------This is a regression test placeholder component implementation class, the main function is to verify the framework link, the main technical inheritance
###
### class:HouseholderOLSComponent
### ------This is an OLS linear regression component implementation class based on Householder QR decomposition, the main function is to solve the least squares optimal regression coefficients and batch prediction, the main technical JAX technology stack (thin QR decomposition, triangular back substitution, JIT compilation)
###
### class:SVDRidgeComponent
### ------This is a ridge regression component implementation class based on SVD decomposition, the main function is to solve the L2 regularized optimal regression coefficients and batch prediction, the main technical JAX technology stack (thin SVD decomposition, singular value shrinkage, JIT compilation)
###
### class:CDLassoComponent
### ------This is a LASSO regression component implementation class based on coordinate descent, the main function is to solve the L1 regularized sparse optimal regression coefficients and batch prediction, the main technical JAX technology stack (soft thresholding operator, Gauss-Seidel coordinate sweep, rank-1 residual update, lax.fori_loop compiled control flow, JIT compilation)
###
### class:QRLogisticComponent
### ------This is a binary Logistic regression component based on IRLS and column-pivoted Householder QR decomposition, with numerical-rank truncation, Armijo backtracking, two-class probability inference and threshold-based classification
###
### class:QRProbitComponent
### ------This is a binary Probit regression component based on Fisher scoring and column-pivoted Householder QR decomposition, with explicit numerical-rank failure, Armijo step halving, normal-CDF probability inference and threshold-based classification
###
### class:QRMultinomialLogisticComponent
### ------This is a multinomial Logistic regression component based on reference-category Softmax, jointly coupled IRLS, per-sample Cholesky whitening and handwritten column-pivoted Householder QR, with numerical-rank truncation, Armijo backtracking, K-class probability inference and argmax classification
###
### class:GHKMultinomialProbitComponent
### ------This is a correlated multinomial Probit regression component based on scale-constrained Cholesky covariance parameters, class contrasts, GHK, fixed-draw MSL and BFGS, with independent probability inference and argmax classification
###
### class:SMOSVRComponent
### ------This is an RBF support vector regression component based on compact-dual second-order SMO, exact pair subproblems, KKT diagnostics and fully compiled device-side iteration
###
################################################################################################################################################################################



####### component class #################################################################################################################################################
#########################################################################################################################################################################



class TestRegression(BaseRegression):
    '''Class Introduction:

        This is a regression test placeholder component implementation class, the main function is to verify the framework link, the main technical inheritance

    :example:
        >>> from statjax.regression import TestRegression
        >>> test_regression_instance = TestRegression(n_steps_ahead=1)
        >>> train_result = test_regression_instance.train()
        >>> reasoning_result = test_regression_instance.reasoning()

    :reference:
        nothing
    '''


    def __init__(self,n_steps_ahead):
        '''Attribute Function:

            Define an initialization method, mainly used to initialize the properties of the class
        
        :parameters:
            - n_steps_ahead (int) - Number of extra-sample inference steps
        ''' 

        self.n_steps_ahead = n_steps_ahead

    
    def train(self):
        '''Method Function:

            Define a training method

        :parameters:
            nothing

        :return:
            nothing
        '''

        result = "TestRegression train"
        return result
    

    def reasoning(self):
        '''Method Function:

            Define a method of reasoning

        :parameters:
            nothing

        :return:
            nothing
        '''

        result = "TestRegression reasoning"
        return result
    

    def _info(self):
        '''Method Function:

            Define an internal method to obtain basic meta information

        :parameters:
            nothing

        :return:
            nothing
        '''

        return "TestRegression component"
    


@jax.jit
def _predict_ols(A_test,x_hat):
    '''Method Function:

        Define a JIT-compiled batch prediction function, the main function is to accelerate batch inference of linear regression, the main technical JAX just-in-time compilation. The inference function is a minimalist linear computing logic. It inputs a test feature matrix of any dimension and implements batch prediction through matrix multiplication:

        $y_pred = A_test · x_hat$
        y_pred = A_test·x̂

    :parameters:
        - A_test (jnp.ndarray) - (m,p) Test set feature design matrix, m is the number of test samples
        - x_hat (jnp.ndarray) - (p,) Trained optimal regression coefficients

    :return:
        - y_pred (jnp.ndarray) - (m,) Predicted values of test samples
    '''

    ### Linear regression prediction formula: y_pred = A_test · x_hat
    y_pred = A_test @ x_hat

    return y_pred



class HouseholderOLSComponent(BaseRegression):
    '''Class Introduction:

        This is an OLS linear regression component implementation class based on Householder QR decomposition. The main function is to solve the least squares optimal regression coefficients and perform batch prediction. The main technical JAX technology stack (thin QR decomposition, triangular back substitution, JIT compilation)

    Mathematical principle
    ----------------------
    (1) OLS mathematical definition: Ordinary least squares (OLS) is a classic algorithm for solving overdetermined linear equations. The core goal is to minimize the 2-norm residual between the observed values and the fitted values. The standard mathematical definition of the problem is:

        $min_{x in R^p} ||A*x - b||_2^2$
        min ‖A·x − b‖₂² , A ∈ R^(n×p) is the feature design matrix (n is the number of samples, p is the number of features, industrial scenarios satisfy n >> p and the matrix is column full rank), b ∈ R^n is the sample observation label vector, x ∈ R^p is the linear regression coefficient vector to be solved

    (2) Comparison of mainstream solutions (why Householder QR is chosen):

        - Normal equation method: directly solves the closed-form solution $A^T*A*x = A^T*b$ (Aᵀ·A·x = Aᵀ·b). The core defect is $cond(A^T*A) = cond(A)^2$ (cond(Aᵀ·A) = cond(A)²), which squares the condition number of the original matrix. When facing ill-conditioned matrices, numerical errors accumulate sharply and the solution easily fails. The numerical stability is extremely poor

        - Gram-Schmidt orthogonalization: the classic GS orthogonalization accumulates serious column-by-column orthogonalization errors, and the stability of the modified MGS algorithm is only slightly improved. It is only suitable for classroom teaching demonstrations and cannot be used for industrial numerical computing

        - Householder QR decomposition (industrial standard): matrix decomposition based on orthogonal reflection transformation, which does not amplify the matrix condition number throughout the process. The numerical stability is the best of the three methods, and it is the underlying default implementation algorithm for solving OLS in mainstream scientific computing libraries such as LAPACK, JAX, and MATLAB

    (3) Householder transformation: an efficient orthogonal reflection transformation that can directionally eliminate the elements of the specified dimension of the matrix without changing the vector norm. The transformation matrix is defined as:

        $H = I - 2*u*u^T / u^T*u$
        H = I − 2·(u·uᵀ)/(uᵀ·u) , the matrix satisfies the two core properties of orthogonality and symmetry: Hᵀ = H, Hᵀ·H = I. The core property of orthogonal transformation is the preservation of the vector 2-norm, which is the mathematical basis for the stable solution of the least squares problem by QR decomposition

    (4) QR decomposition and OLS equivalent transformation: through multiple rounds of iterative Householder reflection transformations, the column full rank design matrix can be decomposed into a thin QR decomposition suitable for overdetermined least squares scenarios:

        $A = Q*R$
        A = Q·R , Q ∈ R^(n×p) is a column orthogonal matrix satisfying Qᵀ·Q = I_p, R ∈ R^(p×p) is an invertible upper triangular full rank square matrix

        Substituting the QR decomposition into the OLS objective function, and using the core property that orthogonal transformation does not change the 2-norm, the equivalent simplification of the problem is completed:

        $||A*x - b||_2^2 = ||Q*R*x - b||_2^2 = ||Q^T*(Q*R*x - b)||_2^2 = ||R*x - Q^T*b||_2^2$
        ‖A·x − b‖₂² = ‖Q·R·x − b‖₂² = ‖Qᵀ·(Q·R·x − b)‖₂² = ‖R·x − Qᵀ·b‖₂²

        The original complex overdetermined least squares optimization problem is finally equivalently transformed into a simple upper triangular square linear equation system solving problem:

        $R*x_hat = Q^T*b$
        R·x̂ = Qᵀ·b , the OLS optimal regression coefficients x̂ can be solved efficiently and stably through the upper triangular matrix back substitution algorithm

    Algorithm flow
    --------------
    Step 1 (Householder thin QR decomposition): construct Householder reflection matrices column by column for the column full rank design matrix A, iteratively eliminate all elements in the lower triangle of the matrix, transform the original matrix into an upper triangular form, and accumulate all reflection transformations to obtain the column orthogonal matrix Q. Only the thin QR dimensions (n×p, p×p) are retained, and redundant dimensions are discarded, which greatly saves memory and computing resources

    Step 2 (orthogonal transformation of the observation vector): calculate $Q^T*b$ (Qᵀ·b), project the original observation vector onto the space spanned by the orthogonal basis Q, complete the equivalent transformation of the overdetermined least squares problem, and simplify the complex optimization problem into a standard upper triangular equation system solution

    Step 3 (upper triangular back substitution to solve the optimal coefficients): for the simplified upper triangular equation system $R*x = Q^T*b$ (R·x = Qᵀ·b), the back substitution algorithm is used to solve forward from the last dimension of the feature, and finally the global optimal regression coefficients x̂ are obtained. The computational complexity of this step is only O(p^2), and the solution efficiency is much higher than that of general matrix solving algorithms

    JAX unique advantages
    ---------------------
    The entire training, inference and solution link is fully differentiable and JIT-compilable, and all operators support automatic differentiation. It can be directly nested in complex scenarios such as deep learning fine-tuning, iterative optimization, and hyperparameter search. This is a core capability that NumPy/SciPy does not have. The demonstration code for solving the Jacobian matrix of the OLS solution is as follows:

    .. code-block:: python

        import jax
        import jax.numpy as jnp
        from jax.scipy.linalg import qr,solve_triangular

        jax.config.update("jax_enable_x64",True)

        ### Encapsulate a differentiable OLS solution function
        def ols_solve(A,b):
            Q,R = qr(A,mode="economic")
            QT_b = Q.T @ b
            return solve_triangular(R,QT_b,lower=False)

        ### Solve the Jacobian matrix of the regression coefficients with respect to the observation label b
        jac_fun = jax.jacfwd(ols_solve,argnums=1)
        jac_matrix = jac_fun(A,b)
        print("Jacobian matrix shape of coefficients with respect to b:",jac_matrix.shape)

    Precautions
    -----------
    (1) Rank constraint: this algorithm is suitable for column full rank design matrices. If the features have multicollinearity and the matrix is rank-deficient, R will have diagonal elements close to 0, and the solution will be numerically unstable. In rank-deficient scenarios, the JAX official lstsq interface should be used to automatically truncate singular values to ensure solution stability

    (2) QR mode selection: jax.scipy.linalg.qr does not support the reduced parameter. The OLS scenario must use mode="economic" to implement thin QR decomposition. It is forbidden to use mode="full" to solve OLS. Complete QR will generate a large number of redundant dimensions and waste computing resources. For very large matrix scenarios, mode="raw" can be used without explicitly storing the Q matrix to further save memory

    (3) Precision configuration priority: jax_enable_x64 must be configured before all JAX arrays are created. Post-configuration cannot take effect, which will cause large numerical errors and affect the accuracy of QR decomposition and least squares solutions. This component has been globally enabled at the module level

    (4) Array uniformity: it is forbidden to mix NumPy arrays and JAX arrays, which will interrupt the differentiable link, trigger data copy, and lose the core capabilities of JAX automatic differentiation and JIT compilation

    (5) Reproducibility description: minor differences in the underlying random algorithms of different JAX versions will not affect the correctness of the algorithm. The core criterion is that the deviation between the QR solution and the lstsq solution is at the machine precision level (1e-12~1e-16)

    (6) Engineering adaptation difference: the handwritten QR + back substitution process is transparent and controllable, suitable for principle research, secondary development and algorithm optimization; the official lstsq is based on the same Householder QR at the bottom, with additional integrated rank detection, regularization truncation, and exception fault tolerance logic, which is more robust and suitable for direct deployment in production environments

    :example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from statjax.regression import HouseholderOLSComponent
        >>> # Generate a reproducible and verifiable OLS simulated dataset with dual fixed random seeds
        >>> key_A = jax.random.PRNGKey(0)
        >>> key_noise = jax.random.PRNGKey(1)
        >>> x_true = jnp.array([1.2,-2.1,0.7,3.3,-1.5],dtype=jnp.float64)
        >>> A_train = jax.random.normal(key_A,(100,5),dtype=jnp.float64)
        >>> b_train = A_train @ x_true + 0.05 * jax.random.normal(key_noise,(100,),dtype=jnp.float64)
        >>> # Train the OLS model and solve the optimal regression coefficients
        >>> ols_instance = HouseholderOLSComponent()
        >>> x_hat,metrics = ols_instance.train(A=A_train,b=b_train)
        >>> # Batch prediction on the test set
        >>> A_test = jax.random.normal(jax.random.PRNGKey(2),(10,5),dtype=jnp.float64)
        >>> y_pred = ols_instance.reasoning(A_test=A_test)

    :reference:
        - [Ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)
        - [Householder transformation](https://en.wikipedia.org/wiki/Householder_transformation)
        - [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition)
        - StatJAX001 OLS Design Document `"StatJAX001 OLS SH V003"<https://github.com/redblue0216/statjax>`_
    '''


    def __init__(self):
        '''Attribute Function:

            Define an initialization method, mainly used to initialize the properties of the class

        :parameters:
            nothing
        '''

        ### Optimal regression coefficients solved by training, None means the model has not been trained yet
        self.x_hat = None
        ### Training accuracy evaluation metrics (residual norm, deviation from the official lstsq benchmark)
        self.metrics = None


    def train(self,A,b):
        '''Method Function:

            Define a training method, the main function is to solve the OLS optimal regression coefficients based on Householder thin QR decomposition, including three steps: thin QR decomposition, orthogonal transformation of the observation vector, and upper triangular back substitution

        :parameters:
            - A (jnp.ndarray) - (n,p) Training set design matrix, n is the number of samples, p is the feature dimension, column full rank is required
            - b (jnp.ndarray) - (n,) Training set observation label vector

        :return:
            - x_hat (jnp.ndarray) - (p,) Trained optimal regression coefficients
            - metrics (dict) - Training accuracy evaluation metrics (residual_norm, diff_with_lstsq)
        '''

        ### Step 1: Householder thin QR decomposition (economic is the optimal OLS mode exclusive to JAX)
        Q,R = qr(A,mode="economic")
        ### Step 2: Orthogonal transformation of the observation vector, completing the equivalent simplification of the least squares problem
        QT_b = Q.T @ b
        ### Step 3: Upper triangular matrix back substitution to solve the optimal regression coefficients
        x_hat = solve_triangular(R,QT_b,lower=False)

        ### Metric 1: Overall fitting residual 2-norm (model fitting effect)
        residual_norm = jnp.linalg.norm(A @ x_hat - b)
        ### Metric 2: Compare with the JAX official lstsq benchmark algorithm to verify the correctness of the algorithm
        x_lstsq,_,_,_ = lstsq(A,b,rcond=None)
        diff_qr_lstsq = jnp.linalg.norm(x_hat - x_lstsq)

        ### Store training results in instance attributes for the reasoning method to use
        self.x_hat = x_hat
        self.metrics = {
            "residual_norm": float(residual_norm),
            "diff_with_lstsq": float(diff_qr_lstsq)
        }

        return self.x_hat,self.metrics


    def reasoning(self,A_test):
        '''Method Function:

            Define a method of reasoning, the main function is to use the trained regression coefficients to perform JIT-accelerated batch prediction on the test set

        :parameters:
            - A_test (jnp.ndarray) - (m,p) Test set feature design matrix, m is the number of test samples

        :return:
            - y_pred (jnp.ndarray) - (m,) Predicted values of test samples
        '''

        ### Check whether the model has been trained
        if self.x_hat is None:
            raise RuntimeError("HouseholderOLSComponent has not been trained yet, please call the train method first")

        ### JIT-accelerated batch prediction: y_pred = A_test · x_hat
        y_pred = _predict_ols(A_test,self.x_hat)

        return y_pred


    def _info(self):
        '''Method Function:

            Define an internal method to obtain basic meta information

        :parameters:
            nothing

        :return:
            - info (str) - Text introduction of the component
        '''

        return "HouseholderOLSComponent: OLS linear regression component based on Householder QR decomposition, implemented with the JAX technology stack"
    


@jax.jit
def _predict_ridge(A_test,w_hat):
    '''Method Function:

        Define a JIT-compiled batch prediction function, the main function is to accelerate batch inference of ridge regression, the main technical JAX just-in-time compilation. The inference function is a minimalist linear computing logic. It inputs a test feature matrix of any dimension and implements batch prediction through matrix multiplication:

        $y_pred = A_test · w_hat$
        y_pred = A_test·ŵ

    :parameters:
        - A_test (jnp.ndarray) - (m,p) Test set feature design matrix, m is the number of test samples
        - w_hat (jnp.ndarray) - (p,) Trained optimal ridge regression coefficients

    :return:
        - y_pred (jnp.ndarray) - (m,) Predicted values of test samples
    '''

    ### Linear regression prediction formula: y_pred = A_test · w_hat
    y_pred = A_test @ w_hat

    return y_pred



class SVDRidgeComponent(BaseRegression):
    '''Class Introduction:

        This is a ridge regression component implementation class based on SVD decomposition. The main function is to solve the L2 regularized optimal regression coefficients and perform batch prediction. The main technical JAX technology stack (thin SVD decomposition, singular value shrinkage, JIT compilation)

    Mathematical principle
    ----------------------
    (1) Ridge regression mathematical definition: ridge regression adds L2 regularization to the least squares objective. The standard mathematical definition of the problem is:

        $min_{w} ||X*w - y||_2^2 + lambda*||w||_2^2 , lambda >= 0$
        min ‖X·w − y‖₂² + λ·‖w‖₂² , X ∈ R^(n×p) is the feature design matrix (n is the number of samples, p is the number of features), y ∈ R^n is the sample observation label vector, w ∈ R^p is the regression coefficient vector to be solved

        Taking the derivative with respect to w and setting it to zero gives the closed-form solution:

        $w_hat_ridge = (X^T*X + lambda*I)^(-1) * X^T*y$
        ŵ_ridge = (Xᵀ·X + λ·I)⁻¹·Xᵀ·y , compared with the OLS solution (Xᵀ·X)⁻¹·Xᵀ·y, there is an additional λ·I term. When the features are highly collinear, Xᵀ·X is nearly singular with an extremely large condition number, and direct inversion explodes numerically; λ·I turns Xᵀ·X + λ·I into a positive definite matrix, guaranteeing invertibility and shrinking the coefficients. This is the core mechanism by which ridge regression solves multicollinearity

    (2) Comparison of mainstream solutions (why SVD is chosen): there are three underlying paths to solve ridge regression — Householder QR, Cholesky, and SVD. The key to choosing SVD is that it maps L2 regularization to the attenuation of each singular component, while natively handling multicollinearity and rank deficiency:

        ==================  ==============================================  ==============================================  ==============================================
        Dimension           Householder QR                                  Cholesky                                        SVD (this solution)
        ==================  ==============================================  ==============================================  ==============================================
        Regularization      Not natively supported, needs the augmented     Natively supports Xᵀ·X + λ·I                    Natively supported, lambda acts directly on
            support         matrix [X; sqrt(lambda)·I]                                                                      the singular values
        Collinearity/rank   Moderate: near-zero diagonal elements of R      Weak: Xᵀ·X loses precision, unstable at         Optimal: small singular values are suppressed
            deficiency      amplify errors                                  small lambda                                    by lambda and can be truncated
            robustness
        Complexity          O(n·p^2), smallest constant                     O(p^3), requires explicit Xᵀ·X                  O(n·p^2), larger constant
        Spectral info /     None, only w is obtained                        None                                            Strong, singular values sigma_i and the
            diagnostics                                                                                                     condition number are obtained directly
        Scenario            Full rank, n >> p, speed-oriented               Small p, no collinearity, fastest               Collinearity, rank deficiency, lambda tuning,
                                                                                                                            diagnostics required
        ==================  ==============================================  ==============================================  ==============================================

        Selection conclusion: this component targets collinear data + tunable regularization + spectral diagnostics. SVD is the only solution that natively carries L2 regularization and remains numerically stable under rank deficiency. It reduces ridge regression to a per-singular-component attenuation of 1/(sigma_i^2 + lambda), which is both robust and intuitive

    (3) SVD core properties: any real matrix can be decomposed as:

        $X = U*Sigma*V^T$
        X = U·Σ·Vᵀ , U ∈ R^(n×n) and V ∈ R^(p×p) are orthogonal matrices (Uᵀ·U = I, Vᵀ·V = I), Σ ∈ R^(n×p) is a diagonal matrix whose diagonal elements are the singular values sigma_1 >= sigma_2 >= ... >= sigma_p >= 0

        Three core properties support the ridge regression implementation:

        - Orthogonality preserves norms: the orthogonal transformations of U and V do not change the vector L2 norm. After projecting the original problem onto the singular principal axes, all directions are decoupled and independent of each other

        - Rank and condition number visualization: rank(X) = number of non-zero singular values; cond(X) = sigma_1/sigma_p. Extremely small singular values are exactly the collinear directions, which can be directly observed and truncated

        - Explicit pseudo-inverse: the Moore-Penrose pseudo-inverse of X can be written as $X^+ = V*Sigma^+*U^T$ (X⁺ = V·Σ⁺·Uᵀ) with diagonal elements 1/sigma_i, which is the stable form of OLS

    (4) SVD and ridge regression: substituting the SVD into the ridge closed-form solution, since $X^T*X = V*Sigma^T*Sigma*V^T$ (Xᵀ·X = V·Σᵀ·Σ·Vᵀ), we get:

        $w_hat = V*(Sigma^T*Sigma + lambda*I)^(-1)*Sigma^T*U^T*y = V*diag(sigma_i/(sigma_i^2+lambda))*U^T*y$
        ŵ = V·(Σᵀ·Σ + λ·I)⁻¹·Σᵀ·Uᵀ·y = V·diag(σᵢ/(σᵢ²+λ))·Uᵀ·y

        This form reveals the per-principal-axis shrinkage essence of ridge regression: for the i-th singular direction, the weight correction coefficient is d_i = sigma_i/(sigma_i^2 + lambda). When sigma_i is large (signal principal axis), d_i is approximately 1/sigma_i and is barely affected by regularization; when sigma_i is small (collinearity/noise direction), d_i is approximately sigma_i/lambda -> 0 and is strongly suppressed. The role of lambda is to attenuate the small singular value directions, which is the mechanism by which it eliminates numerical explosion in exchange for a bias-variance trade-off

        The actual shrinkage effect of the coefficients with lambda (from a parallel scan of jax.vmap in the original implementation, float32):

        ============  ==========  ======  ======  ======  ======
        lambda        0 (OLS)     0.1     1       10      100
        ============  ==========  ======  ======  ======  ======
        ||w||         3,698,203   5.43    5.41    5.25    4.29
        ============  ==========  ======  ======  ======  ======

        Key breakpoint: when lambda goes from 0 to 0.1, the coefficient norm drops by about 680,000 times (the collinear directions are suppressed); as lambda continues to increase, the shrinkage flattens out. On the same data, the OLS hold-out MSE is about 74, while ridge with lambda=1 is about 27 (an improvement of about 2.7x)

    Algorithm flow
    --------------
    .. code-block:: text

        Input X(n×p), y(n,), lambda
            |
            v
        Feature standardization (optional Z-score)
            |
            v
        SVD decomposition: X = U·Σ·Vᵀ
            |----------------------------> Numerical verification
            v                               (compare with closed-form)
        Corrected singular values: d = σ/(σ²+λ)
            |
            v
        Weighted back multiplication: w = V·(d ⊙ Uᵀ·y)
            |
            +------------> Coefficients w (p,)
            |
            v
        Inference: ŷ = X·w (single matmul)

    Step 1 (input and preprocessing): read X (n×p), y (n,), and the regularization coefficient lambda. When the feature scales differ greatly, Z-score standardization should be performed first to prevent large-scale features from being mistakenly hurt by the L2 penalty (see Precautions for scale sensitivity)

    Step 2 (thin SVD decomposition): call jnp.linalg.svd(X, full_matrices=False) to obtain the reduced U (n×min(n,p)), S (min(n,p),), Vᵀ (p×p), avoiding the memory waste of constructing the complete U (n×n)

    Step 3 (singular value shrinkage correction): compute d_i = sigma_i/(sigma_i^2 + lambda) for each singular value, vectorized as d = S / (S**2 + lam). The regularization effect of lambda is completed in this single step

    Step 4 (weighted back multiplication for w): first compute Uᵀ·y, then multiply element-wise by d, and finally left-multiply by V: w = Vt.T @ (d * (U.T @ y))

    Step 5 (inference): for new samples, y_hat = X @ w is enough. The inference degenerates to a single matrix multiplication, with no matrix decomposition overhead

    Step 6 (verification): compare with the closed-form solution (Xᵀ·X + λ·I)⁻¹·Xᵀ·y (implemented by linear equation solving, explicit inversion is forbidden) on the maximum absolute deviation, confirming that the two paths are numerically consistent

    JAX unique advantages
    ---------------------
    Using JAX instead of pure NumPy, the core benefit is that the same set of matrix code can be seamlessly connected to the JAX function transformation system. The entire training, inference and solution link is fully differentiable and JIT-compilable, and all operators support automatic differentiation:

        ==================  ==============================================  ==============================================
        Feature             Role                                            Embodiment in this solution
        ==================  ==============================================  ==============================================
        jax.jit             XLA compilation, eliminating Python scheduling  Training and inference are compiled as a whole,
                            overhead                                        suitable for batch prediction/online serving
        jax.vmap            Vectorized batch execution                      One line of parallel scanning over a lambda
                                                                            grid, directly obtaining the shrinkage curve
        jax.grad            Automatic differentiation                       Computing the gradient of the objective
                                                                            function, verifying whether the closed-form
                                                                            solution is at the extreme point
        Device independence The same code runs on CPU/GPU/TPU               Large-matrix SVD can be switched to the GPU
                                                                            backend for acceleration
        Differentiable      Tensor operations are traceable and             The matrix decomposition results can be
            pipeline        transformable                                   connected to differentiable pipelines such as
                                                                            neural networks
        ==================  ==============================================  ==============================================

        More importantly, this SVD solution logic can be combined with automatic differentiation — for example, treating lambda as a learnable parameter and training end-to-end with grad backpropagation, which requires handwritten derivatives in pure NumPy. The demonstration code for verifying that the gradient of the ridge objective function at the optimal solution is zero is as follows:

    .. code-block:: python

        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64",True)

        ### Encapsulate a differentiable SVD ridge regression solution function
        def ridge_solve(A,b,lam):
            U,S,Vt = jnp.linalg.svd(A,full_matrices=False)
            d = S / (S ** 2 + lam)
            return Vt.T @ (d * (U.T @ b))

        ### Ridge objective function: J(w) = ||X·w - y||^2 + lambda*||w||^2
        def loss(w,A,b,lam):
            return jnp.sum((A @ w - b) ** 2) + lam * jnp.sum(w ** 2)

        ### Verify that the gradient of the objective function at the optimal solution is zero (i.e. the extreme point)
        grad_loss = jax.grad(loss)
        g = grad_loss(ridge_solve(A,b,lam),A,b,lam)
        print("Gradient norm of the objective function at the optimal solution:",jnp.linalg.norm(g))

    Precautions
    -----------
    (1) Explicit inversion is forbidden: the theoretical formula (Xᵀ·X + λ·I)⁻¹ is only a mathematical shorthand. Engineering implementations always use SVD or linear equation solving (jnp.linalg.solve) to avoid floating-point precision disasters

    (2) Feature scale sensitivity: the L2 penalty is scale-sensitive, and large-scale features are penalized less. Z-score standardization must be performed before use, and the inference stage must apply the same mean/standard deviation to the test data. Standardization is a preprocessing responsibility and is not included in this component

    (3) dtype consistency: this component follows the module-level jax_enable_x64 configuration and uses float64 throughout. It is forbidden to mix NumPy arrays and JAX arrays, which will interrupt the differentiable link, trigger data copy, and lose the core capabilities of JAX automatic differentiation and JIT compilation

    (4) Choice of lambda: lambda=0 degenerates to OLS (explodes under collinearity); an excessively large lambda introduces bias. Use cross-validation or a vmap grid scan to determine it, observing whether the coefficient norm enters the flat region

    (5) full_matrices=False: when n >> p, the reduced SVD must be used, otherwise the complete U (n×n) costs O(n^2) memory, which is a serious waste

    (6) Condition number warning: sigma_1/sigma_p can be output as a collinearity diagnostic; when sigma_p is close to 0, numerical noise may still exist even with a small lambda, and truncation should be applied when necessary

    (7) jnp.linalg is the built-in path: jnp.linalg.svd of JAX has encapsulated the LAPACK/XLA implementation with good robustness; handwriting a full SVD (bidiagonalization + iteration) is unnecessary, which is consistent with the positioning of "understanding from the matrix bottom layer and landing with mature decomposition primitives"

    :example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from statjax.regression import SVDRidgeComponent
        >>> # Generate a reproducible ridge regression simulated dataset with multicollinearity (the last column is a linear combination of the previous columns plus tiny perturbations)
        >>> keys = jax.random.split(jax.random.PRNGKey(0),4)
        >>> A_base = jax.random.normal(keys[0],(100,4),dtype=jnp.float64)
        >>> A_col = A_base @ jax.random.normal(keys[1],(4,1),dtype=jnp.float64) + 1e-6 * jax.random.normal(keys[2],(100,1),dtype=jnp.float64)
        >>> A_train = jnp.concatenate([A_base,A_col],axis=1)
        >>> w_true = jnp.array([1.2,-2.1,0.7,3.3,-1.5],dtype=jnp.float64)
        >>> b_train = A_train @ w_true + 0.05 * jax.random.normal(keys[3],(100,),dtype=jnp.float64)
        >>> # Train the ridge regression model and solve the optimal regression coefficients
        >>> ridge_instance = SVDRidgeComponent(lam=1.0)
        >>> w_hat,metrics = ridge_instance.train(A=A_train,b=b_train)
        >>> # Batch prediction on the test set
        >>> A_test = jax.random.normal(jax.random.PRNGKey(2),(10,5),dtype=jnp.float64)
        >>> y_pred = ridge_instance.reasoning(A_test=A_test)

    :reference:
        - [Ridge regression](https://en.wikipedia.org/wiki/Ridge_regression)
        - [Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
        - StatJAX002 Ridge Regression Design Document `"StatJAX002 Ridge SH V002"<https://github.com/redblue0216/statjax>`_
    '''


    def __init__(self,lam=1.0):
        '''Attribute Function:

            Define an initialization method, mainly used to initialize the properties of the class

        :parameters:
            - lam (float) - L2 regularization coefficient, lambda >= 0. lambda=0 degenerates to OLS; the larger the lambda, the stronger the coefficient shrinkage
        '''

        ### L2 regularization coefficient
        self.lam = lam
        ### Optimal ridge regression coefficients solved by training, None means the model has not been trained yet
        self.w_hat = None
        ### Training accuracy evaluation metrics (residual norm, deviation from the closed-form benchmark, coefficient norm)
        self.metrics = None


    def train(self,A,b):
        '''Method Function:

            Define a training method, the main function is to solve the ridge regression optimal coefficients based on thin SVD decomposition, including three steps: thin SVD decomposition, singular value shrinkage correction, and weighted back multiplication

        :parameters:
            - A (jnp.ndarray) - (n,p) Training set design matrix, n is the number of samples, p is the feature dimension
            - b (jnp.ndarray) - (n,) Training set observation label vector

        :return:
            - w_hat (jnp.ndarray) - (p,) Trained optimal ridge regression coefficients
            - metrics (dict) - Training accuracy evaluation metrics (residual_norm, diff_with_closed_form, coef_norm)
        '''

        ### Step 1: Thin SVD decomposition (full_matrices=False avoids the redundant O(n^2) memory of the complete U matrix)
        U,S,Vt = jnp.linalg.svd(A,full_matrices=False)
        ### Step 2: Singular value shrinkage correction, the L2 regularization is completed in this step: d_i = sigma_i / (sigma_i^2 + lambda)
        d = S / (S ** 2 + self.lam)
        ### Step 3: Weighted back multiplication to solve the optimal coefficients: w_hat = V · (d ⊙ Uᵀ·b)
        w_hat = Vt.T @ (d * (U.T @ b))

        ### Metric 1: Overall fitting residual 2-norm (model fitting effect)
        residual_norm = jnp.linalg.norm(A @ w_hat - b)
        ### Metric 2: Compare with the closed-form benchmark solved by linear equation solving (explicit matrix inversion is forbidden), verifying the correctness of the SVD solution
        p = A.shape[1]
        w_closed_form = jnp.linalg.solve(A.T @ A + self.lam * jnp.eye(p,dtype=A.dtype),A.T @ b)
        diff_svd_closed_form = jnp.linalg.norm(w_hat - w_closed_form)
        ### Metric 3: Coefficient 2-norm (multicollinearity diagnostic, contrast with the exploding OLS coefficient norm)
        coef_norm = jnp.linalg.norm(w_hat)

        ### Store training results in instance attributes for the reasoning method to use
        self.w_hat = w_hat
        self.metrics = {
            "residual_norm": float(residual_norm),
            "diff_with_closed_form": float(diff_svd_closed_form),
            "coef_norm": float(coef_norm)
        }

        return self.w_hat,self.metrics


    def reasoning(self,A_test):
        '''Method Function:

            Define a method of reasoning, the main function is to use the trained ridge regression coefficients to perform JIT-accelerated batch prediction on the test set

        :parameters:
            - A_test (jnp.ndarray) - (m,p) Test set feature design matrix, m is the number of test samples

        :return:
            - y_pred (jnp.ndarray) - (m,) Predicted values of test samples
        '''

        ### Check whether the model has been trained
        if self.w_hat is None:
            raise RuntimeError("SVDRidgeComponent has not been trained yet, please call the train method first")

        ### JIT-accelerated batch prediction: y_pred = A_test · w_hat
        y_pred = _predict_ridge(A_test,self.w_hat)

        return y_pred


    def _info(self):
        '''Method Function:

            Define an internal method to obtain basic meta information

        :parameters:
            nothing

        :return:
            - info (str) - Text introduction of the component
        '''

        return "SVDRidgeComponent: ridge regression component based on SVD decomposition, implemented with the JAX technology stack"



def soft_threshold(z,lam):
    '''Method Function:

        Define the soft thresholding operator, the main function is to serve as the L1 proximal mapping, i.e. the atomic operator of the coordinate descent LASSO solver, acting element-wise: soft(z,lam) = sign(z)*max(|z|-lam,0). The update amount whose absolute value is less than or equal to lambda is directly compressed to 0, which is the mathematical source of sparsity

    :parameters:
        - z (jnp.ndarray) - Input value or array
        - lam (float) - L1 regularization coefficient, the threshold of the soft thresholding operator

    :return:
        - result (jnp.ndarray) - Soft-thresholded value or array
    '''

    ### Soft thresholding formula: soft(z,lam) = sign(z)*max(|z|-lam,0)
    result = jnp.sign(z) * jnp.maximum(jnp.abs(z) - lam,0.0)

    return result



def _standardize(X):
    '''Method Function:

        Define a column standardization function, the main function is to eliminate the influence of feature scales on the L1 penalty, the main technical Z-score standardization. A too small standard deviation is replaced by 1 as a fallback to prevent division by zero

    :parameters:
        - X (jnp.ndarray) - (n,p) Input design matrix

    :return:
        - Xs (jnp.ndarray) - (n,p) Standardized design matrix
        - mu (jnp.ndarray) - (p,) Column means
        - s (jnp.ndarray) - (p,) Column standard deviations (fallback to 1 when too small)
    '''

    ### Column means
    mu = X.mean(axis=0)
    ### Column standard deviations, too small values are replaced by 1 to prevent division by zero
    s = X.std(axis=0)
    s = jnp.where(s < 1e-8,1.0,s)
    ### Z-score standardization
    Xs = (X - mu) / s

    return Xs,mu,s



def _one_sweep(X,beta,r,col_sq,lam):
    '''Method Function:

        Define a single-round coordinate sweep function, the main function is to perform exact single-coordinate minimization for each coordinate in turn with the Gauss-Seidel loop, the main technical lax.fori_loop compiled control flow and rank-1 residual update (avoiding the expensive recomputation of X@beta)

    :parameters:
        - X (jnp.ndarray) - (n,p) Standardized design matrix
        - beta (jnp.ndarray) - (p,) Current regression coefficients in the standardized space
        - r (jnp.ndarray) - (n,) Current residual vector
        - col_sq (jnp.ndarray) - (p,) Precomputed squared column 2-norms s_j = ||x_j||_2^2
        - lam (float) - L1 regularization coefficient

    :return:
        - beta (jnp.ndarray) - (p,) Updated regression coefficients in the standardized space
        - r (jnp.ndarray) - (n,) Updated residual vector
    '''

    p = X.shape[1]

    def body(j,carry):
        '''Method Function:

            Perform one Gauss-Seidel coordinate update and its rank-1 residual correction inside lax.fori_loop
        '''

        beta,r = carry
        ### Take the j-th column
        xj = X[:,j]
        ### s_j = ||x_j||_2^2 (precomputed)
        s = col_sq[j]
        old = beta[j]
        ### z = x_j^T·r + s_j*beta_j_old, correcting the contribution of the current coordinate to the residual
        z = xj @ r + s * old
        ### Soft thresholding update: beta_j_new = soft(z_j,lam)/s_j
        new = soft_threshold(z,lam) / s
        ### Rank-1 residual update, reducing the O(n*p) matrix multiplication to O(n)
        r = r - (new - old) * xj
        beta = beta.at[j].set(new)
        return beta,r

    return lax.fori_loop(0,p,body,(beta,r))



@jax.jit
def _fit_lasso(X,y,lam,sweeps):
    '''Method Function:

        Define a JIT-compiled LASSO training function, the main function is to solve the L1 regularized optimal regression coefficients in the standardized space by coordinate descent, the main technical JAX just-in-time compilation and lax.fori_loop double-loop compiled control flow. The training chain is: column standardization -> intercept separation (the intercept is not penalized by L1) -> all-zero initialization -> column norm precomputation -> multi-round Gauss-Seidel coordinate sweeps -> scale restoration

    :parameters:
        - X (jnp.ndarray) - (n,p) Training set design matrix
        - y (jnp.ndarray) - (n,) Training set observation label vector
        - lam (float) - L1 regularization coefficient
        - sweeps (int) - Number of full coordinate sweep rounds

    :return:
        - w_hat (jnp.ndarray) - (p,) Optimal regression coefficients in the original scale
        - intercept (jnp.ndarray) - Scalar intercept in the original scale
        - beta_std (jnp.ndarray) - (p,) Optimal regression coefficients in the standardized space (for KKT verification)
        - obj (jnp.ndarray) - Scalar LASSO objective function value 0.5*||r||_2^2 + lam*||beta||_1 in the standardized space
        - r (jnp.ndarray) - (n,) Final residual vector in the standardized space (for KKT verification)
        - Xs (jnp.ndarray) - (n,p) Standardized design matrix (for KKT verification)
    '''

    ### Step 1: Column standardization, eliminating the influence of feature scales on the L1 penalty
    Xs,mu,s = _standardize(X)
    n,p = Xs.shape
    ### Step 2: Intercept separation, the intercept is not penalized by L1
    ybar = y.mean()
    yc = y - ybar
    ### Step 3: All-zero initialization (corresponding to infinite lambda) and initial residual
    beta = jnp.zeros(p,dtype=Xs.dtype)
    r = yc - Xs @ beta
    ### Precompute the squared 2-norm of each column as the denominator of the coordinate update
    col_sq = jnp.sum(Xs ** 2,axis=0)

    def sweep_loop(i,carry):
        '''Method Function:

            Apply one complete coordinate sweep while threading immutable beta and residual state through the compiled outer loop
        '''

        beta,r = carry
        return _one_sweep(Xs,beta,r,col_sq,lam)

    ### Step 4: Double lax.fori_loop compiled control flow, outer loop of sweeps rounds, inner loop of p coordinates
    beta,r = lax.fori_loop(0,sweeps,sweep_loop,(beta,r))

    ### Step 5: Scale restoration from the standardized space to the original scale
    w_hat = beta / s
    intercept = ybar - (mu / s) @ beta
    ### LASSO objective function value in the standardized space: 0.5*||r||_2^2 + lam*||beta||_1
    obj = jnp.sum(0.5 * r ** 2) + lam * jnp.sum(jnp.abs(beta))

    return w_hat,intercept,beta,obj,r,Xs



@jax.jit
def _kkt_violation(Xs,r,beta_std,lam):
    '''Method Function:

        Define a JIT-compiled KKT optimality verification function, the main function is to compute the maximum violation of the LASSO KKT optimality condition as the core criterion for the correctness of the algorithm (JAX native verification, replacing sklearn comparison). The KKT condition is $X^T*(X*beta - y) + lambda*d_||beta||_1 in 0$ (Xᵀ·(X·β−y) + λ·∂‖β‖₁ ∋ 0), where the subgradient of |beta_j| is [-1,1] when beta_j = 0

    :parameters:
        - Xs (jnp.ndarray) - (n,p) Standardized design matrix
        - r (jnp.ndarray) - (n,) Residual vector yc - Xs @ beta_std in the standardized space
        - beta_std (jnp.ndarray) - (p,) Regression coefficients in the standardized space
        - lam (float) - L1 regularization coefficient

    :return:
        - max_viol (jnp.ndarray) - Scalar maximum violation of the KKT optimality condition, approaching 0 when ideally converged
    '''

    ### g_j = x_j^T·r
    g = Xs.T @ r
    ### Branch beta > 0: |g - lam| should be 0
    cond1 = jnp.where(beta_std > 1e-10,jnp.abs(g - lam),0.0)
    ### Branch beta < 0: |g + lam| should be 0
    cond2 = jnp.where(beta_std < -1e-10,jnp.abs(g + lam),0.0)
    ### Branch beta = 0: max(0,|g| - lam) should be 0
    cond3 = jnp.where(jnp.abs(beta_std) <= 1e-10,jnp.maximum(0.0,jnp.abs(g) - lam),0.0)
    max_viol = jnp.max(jnp.maximum(jnp.maximum(cond1,cond2),cond3))

    return max_viol



@jax.jit
def _predict_lasso(A_test,w_hat,intercept):
    '''Method Function:

        Define a JIT-compiled batch prediction function, the main function is to accelerate batch inference of LASSO regression, the main technical JAX just-in-time compilation. The inference function is a minimalist linear computing logic. It inputs a test feature matrix of any dimension and implements batch prediction through matrix multiplication plus the intercept:

        $y_pred = A_test · w_hat + intercept$
        y_pred = A_test·ŵ + intercept

    :parameters:
        - A_test (jnp.ndarray) - (m,p) Test set feature design matrix, m is the number of test samples
        - w_hat (jnp.ndarray) - (p,) Trained optimal LASSO regression coefficients in the original scale
        - intercept (jnp.ndarray) - Scalar intercept in the original scale

    :return:
        - y_pred (jnp.ndarray) - (m,) Predicted values of test samples
    '''

    ### Linear regression prediction formula: y_pred = A_test · w_hat + intercept
    y_pred = A_test @ w_hat + intercept

    return y_pred



class CDLassoComponent(BaseRegression):
    '''Class Introduction:

        This is a LASSO regression component implementation class based on coordinate descent. The main function is to solve the L1 regularized sparse optimal regression coefficients and perform batch prediction. The main technical JAX technology stack (soft thresholding operator, Gauss-Seidel coordinate sweep, rank-1 residual update, lax.fori_loop compiled control flow, JIT compilation)

    Mathematical principle
    ----------------------
    (1) LASSO mathematical definition: given the design matrix X in R^(n×p) (n samples, p features) and the response vector y in R^n, LASSO solves the convex optimization problem with the L1 norm penalty:

        $min_{beta} 0.5*||y - X*beta||_2^2 + lambda*||beta||_1 , ||beta||_1 = sum_{j=1}^{p} |beta_j| , lambda >= 0$
        min ½‖y − X·β‖₂² + λ‖β‖₁ , the first term is the OLS residual sum of squares measuring the goodness of fit, the second term is the L1 penalty, and lambda is the regularization strength

        The unique value of the L1 penalty is sparsity: |beta_j| is not differentiable at 0 (its subgradient contains 0), which forces some coefficients to be exactly 0, directly implementing feature selection. This is what L2 (ridge regression) cannot do, and it is the core advantage of LASSO:

        ============  ==============================================  ====================  ==============================================
        Penalty       Solution characteristics                        Feature selection     Closed-form solution
        ============  ==============================================  ====================  ==============================================
        L1 (LASSO)    Sparse (some coefficients are exactly 0)        Yes                   No (iterative solving required)
        L2 (ridge)    Overall shrinkage (towards 0 but not 0)         No                    Yes (inverting Xᵀ·X + λ·I)
        ============  ==============================================  ====================  ==============================================

    (2) Comparison of mainstream solutions (why coordinate descent is chosen): LASSO has no closed-form solution and must be solved iteratively. There are four mainstream underlying implementations:

        ==================  ==============================================  ==============================================  ==============================================
        Method              Core matrix computation                         Characteristics                                 Scenario
        ==================  ==============================================  ==============================================  ==============================================
        Coordinate descent  Vector inner product + soft thresholding +      Simple, stable, high-dimensional friendly,      Industrial mainstay (sklearn default, this
            (CD)            rank-1 residual update                          fast convergence                                solution)
        Proximal gradient   Matrix-vector product + power iteration for     Step size tuning required, convergence prone    Learning proximal optimization frameworks
            (FISTA)         the largest eigenvalue                          to oscillation
        LARS                QR/Cholesky + active set maintenance            Solves the entire lambda path, complex code     Regularization path research
        ADMM                Linear equation solving + soft thresholding     Distributed / super-large matrices              Distributed scenarios
        ==================  ==============================================  ==============================================  ==============================================

        Selection conclusion: coordinate descent has the lowest matrix computation threshold (the inner loop only takes one column of the matrix, vector inner products, vector-scalar products, and element-wise soft thresholding, without any matrix decomposition or eigenvalue solving); it is naturally adapted to high-dimensional p >> n scenarios (the p×p Gram matrix is never explicitly constructed, so the memory overhead is small); the hyperparameters are insensitive and the convergence is stable, without the step-size tuning pain of FISTA; and it is consistent with the underlying algorithm of sklearn Lasso, which can be directly cross-validated

    (3) Coordinate descent core properties: coordinate descent solves the following separable convex problem:

        $min_{beta} f(beta) = g(beta) + sum_{j=1}^{p} h_j(beta_j)$
        min f(β) = g(β) + Σ h_j(β_j) , where g is smooth and h_j is separable convex. In LASSO, g = ½‖y − X·β‖₂² and h_j = λ|β_j|

        The core idea is: fix all other coordinates beta_{-j} each time, perform exact minimization only on the single coordinate beta_j, sweep through all p variables coordinate by coordinate in a Gauss-Seidel loop, and iterate repeatedly until convergence. For convex separable g + sum h_j, coordinate descent is guaranteed to converge to the global optimum, and it converges very fast for quadratic losses

    (4) Coordinate descent and LASSO regression: for LASSO, in the single-variable subproblem with beta_{-j} fixed, denote the residual $r = y - X*beta$ (r = y − X·β) and $s_j = ||x_j||_2^2$ (s_j = ‖x_j‖₂²), and let $z_j = x_j^T*r + s_j*beta_j_old$ (z_j = x_jᵀ·r + s_j·β_j^old). The coordinate update formula is:

        $beta_j_new = soft(z_j,lambda) / s_j$
        β_j^new = soft(z_j, λ)/s_j , where the soft thresholding operator is the L1 proximal mapping:

        $soft(z,lambda) = z - lambda (z > lambda) ; 0 (|z| <= lambda) ; z + lambda (z < -lambda)$
        soft(z,λ) = sign(z)·max(|z|−λ, 0) , taking the subgradient with respect to beta_j and requiring it to contain 0, the soft thresholding operator is naturally derived from the non-differentiability of L1. The soft thresholding compresses the update amount whose absolute value is less than or equal to lambda directly to 0, which is exactly the mathematical source of sparsity. The s_j·β_j^old term corrects the contribution of the current coordinate to the residual, ensuring that the subproblem is an exact minimization rather than a gradient step

        KKT optimality condition: beta is the LASSO solution if and only if $X^T*(X*beta - y) + lambda*d_||beta||_1 in 0$ (Xᵀ·(X·β−y) + λ·∂‖β‖₁ ∋ 0), where the subgradient of |beta_j| is [-1,1] when beta_j = 0. Coordinate descent is exactly the practical method that approaches this condition step by step, and this component uses the maximum violation of the KKT condition as the core criterion for the correctness of the algorithm (JAX native verification, replacing sklearn comparison)

    Algorithm flow
    --------------
    .. code-block:: text

        Input X(n×p), y(n,), lambda
            |
            v
        Column standardization: X = (X - mu) / sigma
            |
            v
        Intercept separation: ybar = mean(y), yc = y - ybar
            |  (the intercept is not penalized by L1)
            v
        Initialization: beta = 0, r = yc - X·beta
            |
            v
        Precompute column norm squares: s_j = ||x_j||_2^2
            |
            v
        Outer loop (sweeps rounds) ----> Inner loop (per coordinate j)
            |                                 z = x_jᵀ·r + s_j·β_j^old
            |                                 β_j^new = soft(z, λ)/s_j
            |                                 rank-1 residual update: r -= (β_new - β_old)·x_j
            |<--------------------------------
            v
        Scale restoration: w_hat = beta / sigma
            |
            v
        Output: w_hat, intercept, objective
            |
            +------------> KKT optimality verification (max violation -> 0)
            |
            v
        Inference: y_pred = A_test·w_hat + intercept (single matmul)

    Step 1 (input): read X (n×p), y (n,), and the L1 regularization coefficient lambda

    Step 2 (column standardization): the LASSO penalty is scale-sensitive, so the features must be centered and scaled to unit variance first, otherwise the penalty strength is unbalanced across features. A too small standard deviation is replaced by 1 as a fallback to prevent division by zero

    Step 3 (intercept separation): the intercept is not penalized by L1 and is represented separately by the sample mean ybar. If the constant term were shrunk by L1, a systematic bias would be introduced

    Step 4 (initialization): beta = 0 (corresponding to infinite lambda), r = yc - Xs @ beta

    Step 5 (column norm precomputation): s_j = ||x_j||_2^2 is precomputed as the denominator of the coordinate update

    Step 6 (outer loop): repeat full sweeps over all features for sweeps rounds

    Step 7 (inner loop): perform exact single-coordinate minimization for each feature j: compute $z = x_j^T*r + s_j*beta_j_old$ (z = x_jᵀ·r + s_j·β_j^old), update with soft thresholding $beta_j_new = soft(z_j,lambda)/s_j$ (β_j^new = soft(z_j, λ)/s_j)

    Step 8 (rank-1 residual update): when only one beta_j changes, the residual only needs the correction delta_beta_j·x_j, avoiding the expensive recomputation of X·beta (reducing the O(n·p) matrix multiplication to O(n))

    Step 9 (convergence): the coefficient change can be checked with ||beta_new - beta_old||_inf < tol; this component uses a fixed number of sweeps for reproducibility

    Step 10 (scale restoration): $w_hat = beta / sigma$ (ŵ = β/σ), $intercept = ybar - (mu/sigma)^T*beta$ (intercept = ȳ − (μ/σ)ᵀ·β), restoring from the standardized space to the original scale

    JAX unique advantages
    ---------------------
    Using JAX instead of pure NumPy, the core benefit is that the same set of matrix code can be seamlessly connected to the JAX function transformation system:

        (1) jax.jit just-in-time compilation: the training and inference functions are compiled by XLA into low-level machine code, and the loops are statically unrolled with lax.fori_loop without Python interpretation overhead, which is an order of magnitude faster than equivalent pure NumPy handwriting, close to C/Fortran speed

        (2) Functional purity (no side effects): all states are passed in and out as immutable arrays, which is naturally reproducible, parallelizable, cacheable, and convenient for unit testing

        (3) Automatic differentiation jax.grad: although coordinate descent does not rely on gradients, it can be used to quickly verify whether the gradient of the objective function with respect to beta is consistent with the analytical subgradient, as a supplementary means of correctness verification

        (4) lax.fori_loop / lax.while_loop compile-time control flow: handwritten iterative algorithms can be truly compiled into the computation graph instead of degenerating into successive Python calls. Ordinary for/while loops cannot operate on device-side arrays under @jax.jit

        (5) Device independence (CPU/GPU/TPU): the same code runs seamlessly on GPU/TPU, and coordinate descent is naturally suitable for column-parallel batching in the p dimension (extensible with vmap/pmap)

        (6) Strong reproducibility: jax.random.PRNGKey + fixed seeds, combined with pure functions, can be accurately reproduced in one run, meeting experimental audit requirements

        The demonstration code for verifying the consistency between the automatic differentiation gradient and the analytical subgradient at the coordinate descent solution is as follows:

    .. code-block:: python

        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64",True)

        ### Encapsulate a differentiable LASSO objective function: J(beta) = 0.5*||y - X·beta||^2 + lambda*||beta||_1
        def lasso_objective(beta,X,y,lam):
            r = y - X @ beta
            return 0.5 * jnp.sum(r ** 2) + lam * jnp.sum(jnp.abs(beta))

        ### Verify the gradient of the objective function at the coordinate descent solution (the smooth part gradient plus the L1 subgradient should satisfy the KKT condition)
        grad_obj = jax.grad(lasso_objective)
        g = grad_obj(w_hat,X,y,lam)
        print("Gradient of the objective function at the solution:",g)

    Precautions
    -----------
    (1) Column standardization is mandatory: the LASSO penalty is measured by scale, and a large disparity in feature scales will unbalance the penalty and distort the solution. This component has built-in column standardization in the training chain, and the inference stage automatically uses the restored original-scale coefficients and intercept

    (2) The intercept is not penalized: if the constant term were shrunk by L1, a systematic bias would be introduced. It is represented separately by the sample mean ybar and restored during inference. The intercept is stored in the instance attribute self.intercept during training and automatically added back during reasoning

    (3) s_j must not be 0: a constant column causes division by zero. This component uses jnp.where(s<1e-8,1,s) as a fallback, and the coefficient of such a column is naturally invalid

    (4) Fixed sweeps vs convergence: the fixed sweeps=300 is simple and reproducible, but may not converge for ill-conditioned data (increase sweeps in this case). A production version is recommended to use lax.while_loop with dynamic stopping by the coefficient change, with a maximum round limit to prevent infinite loops

    (5) Never explicitly construct the Gram matrix Xᵀ·X: in high-dimensional p >> n scenarios, the p×p matrix explodes in memory. Coordinate descent only uses the squared norm of each column and inner products, which naturally avoids this, and this is also where it outperforms the normal equation method

    (6) JAX functional constraints: arrays cannot be modified in place by Python; .at[j].set(...) (immutable update) must be used, and loops must use lax.fori_loop, otherwise @jax.jit will raise an error

    (7) Lambda scale matching: when comparing with sklearn, pay attention to the convention difference — the objective of this component is $0.5*||r||_2^2 + lambda*||beta||_1$ (½‖r‖₂² + λ‖β‖₁), while sklearn uses $1/(2n)*||r||_2^2 + alpha*||beta||_1$ ((1/2n)‖r‖₂² + α‖β‖₁), so sklearn alpha = lambda/n

    (8) Choice of lambda: use cross-validation grid search to select the optimal lambda; a too small lambda degenerates to OLS without sparsity, and a too large lambda compresses all coefficients to 0

    :example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from statjax.regression import CDLassoComponent
        >>> # Generate a reproducible LASSO simulated dataset with sparse true coefficients (only the first 5 coefficients are non-zero)
        >>> key_A = jax.random.PRNGKey(0)
        >>> key_noise = jax.random.PRNGKey(1)
        >>> w_true = jnp.zeros(30,dtype=jnp.float64)
        >>> w_true = w_true.at[:5].set(jnp.array([1.2,-0.8,0.6,-1.1,0.9],dtype=jnp.float64))
        >>> A_train = jax.random.normal(key_A,(200,30),dtype=jnp.float64)
        >>> b_train = A_train @ w_true + 0.1 * jax.random.normal(key_noise,(200,),dtype=jnp.float64)
        >>> # Train the LASSO model and solve the sparse optimal regression coefficients
        >>> lasso_instance = CDLassoComponent(lam=0.1)
        >>> w_hat,metrics = lasso_instance.train(A=A_train,b=b_train)
        >>> # Batch prediction on the test set (the intercept is automatically added back)
        >>> A_test = jax.random.normal(jax.random.PRNGKey(2),(10,30),dtype=jnp.float64)
        >>> y_pred = lasso_instance.reasoning(A_test=A_test)

    :reference:
        - [Lasso (statistics)](https://en.wikipedia.org/wiki/Lasso_(statistics))
        - [Coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent)
        - [Proximal gradient method](https://en.wikipedia.org/wiki/Proximal_gradient_method)
        - StatJAX003 LASSO Design Document `"StatJAX003 LASSO SH V001"<https://github.com/redblue0216/statjax>`_
    '''


    def __init__(self,lam=0.1,sweeps=300):
        '''Attribute Function:

            Define an initialization method, mainly used to initialize the properties of the class

        :parameters:
            - lam (float) - L1 regularization coefficient, lambda >= 0. lambda=0 degenerates to OLS; the larger the lambda, the stronger the sparsity
            - sweeps (int) - Number of full coordinate sweep rounds
        '''

        ### L1 regularization coefficient
        self.lam = lam
        ### Number of full coordinate sweep rounds
        self.sweeps = sweeps
        ### Optimal LASSO regression coefficients in the original scale solved by training, None means the model has not been trained yet
        self.w_hat = None
        ### Intercept in the original scale (separated from the L1 penalty), restored and stored during training and added back during reasoning
        self.intercept = None
        ### Training accuracy evaluation metrics (residual norm, KKT optimality violation, objective function value, number of non-zero coefficients)
        self.metrics = None


    def train(self,A,b):
        '''Method Function:

            Define a training method, the main function is to solve the LASSO sparse optimal regression coefficients based on coordinate descent, including five steps: column standardization, intercept separation, all-zero initialization and column norm precomputation, multi-round Gauss-Seidel coordinate sweeps, and scale restoration

        :parameters:
            - A (jnp.ndarray) - (n,p) Training set design matrix, n is the number of samples, p is the feature dimension
            - b (jnp.ndarray) - (n,) Training set observation label vector

        :return:
            - w_hat (jnp.ndarray) - (p,) Trained optimal LASSO regression coefficients in the original scale
            - metrics (dict) - Training accuracy evaluation metrics (residual_norm, kkt_violation, objective, n_nonzero)
        '''

        ### JIT-compiled coordinate descent training in the standardized space, returning the restored original-scale coefficients and intercept
        w_hat,intercept,beta_std,obj,r,Xs = _fit_lasso(A,b,self.lam,self.sweeps)

        ### Metric 1: Overall fitting residual 2-norm in the original scale (model fitting effect)
        residual_norm = jnp.linalg.norm(A @ w_hat + intercept - b)
        ### Metric 2: Maximum violation of the KKT optimality condition (JAX native verification, replacing sklearn comparison), the core criterion for the correctness of the algorithm
        kkt_viol = _kkt_violation(Xs,r,beta_std,self.lam)
        ### Metric 3: Number of non-zero coefficients (sparsity and feature selection effect indicator)
        n_nonzero = jnp.sum(jnp.abs(w_hat) > 1e-10)

        ### Store training results in instance attributes for the reasoning method to use
        self.w_hat = w_hat
        self.intercept = intercept
        self.metrics = {
            "residual_norm": float(residual_norm),
            "kkt_violation": float(kkt_viol),
            "objective": float(obj),
            "n_nonzero": int(n_nonzero)
        }

        return self.w_hat,self.metrics


    def reasoning(self,A_test):
        '''Method Function:

            Define a method of reasoning, the main function is to use the trained LASSO regression coefficients and intercept to perform JIT-accelerated batch prediction on the test set

        :parameters:
            - A_test (jnp.ndarray) - (m,p) Test set feature design matrix, m is the number of test samples

        :return:
            - y_pred (jnp.ndarray) - (m,) Predicted values of test samples
        '''

        ### Check whether the model has been trained
        if self.w_hat is None:
            raise RuntimeError("CDLassoComponent has not been trained yet, please call the train method first")

        ### JIT-accelerated batch prediction: y_pred = A_test · w_hat + intercept
        y_pred = _predict_lasso(A_test,self.w_hat,self.intercept)

        return y_pred


    def _info(self):
        '''Method Function:

            Define an internal method to obtain basic meta information

        :parameters:
            nothing

        :return:
            - info (str) - Text introduction of the component
        '''

        return "CDLassoComponent: LASSO regression component based on coordinate descent, implemented with the JAX technology stack"



##############################################################################################################################################################################
##############################################################################################################################################################################


def _add_intercept(X):
    '''Method Function:

        Define an intercept augmentation function, the main function is to append a constant one column after feature standardization

    :parameters:
        - X (jnp.ndarray) - (n,p) Standardized feature matrix

    :return:
        - Xb (jnp.ndarray) - (n,p+1) Augmented design matrix, with the intercept column in the last position
    '''

    ones = jnp.ones((X.shape[0],1),dtype=X.dtype)
    Xb = jnp.concatenate([X,ones],axis=1)

    return Xb



def _pivoted_qr_lstsq(A,b,rtol=1e-12):
    '''Method Function:

        Define a column-pivoted Householder QR least-squares solver, the main function is to solve min ||A*x-b||_2 with numerical rank truncation. Dynamic rank slicing intentionally keeps this function outside JIT compilation

    :parameters:
        - A (jnp.ndarray) - (m,k) Weighted and regularization-augmented least-squares matrix
        - b (jnp.ndarray) - (m,) Least-squares right-hand side
        - rtol (float) - Relative threshold applied to the absolute diagonal elements of R

    :return:
        - x (jnp.ndarray) - (k,) Least-squares solution restored to the original column order
        - rank (int) - Estimated numerical rank
    '''

    ### Column-pivoted economic QR: A[:,pivots] = Q @ R
    Q,R,pivots = qr(A,mode="economic",pivoting=True,check_finite=False)
    diagonal = jnp.abs(jnp.diag(R))
    if diagonal.size == 0 or float(diagonal[0]) == 0.0:
        return jnp.zeros(A.shape[1],dtype=A.dtype),0

    ### Retain only reliable directions and solve the leading upper triangular system
    rank = int(jnp.sum(diagonal > rtol * diagonal[0]))
    qty = Q.T @ b
    x_permuted_head = solve_triangular(R[:rank,:rank],qty[:rank],lower=False,check_finite=False)

    ### Restore the solution from pivoted column order to original column order
    x_permuted = jnp.zeros(A.shape[1],dtype=A.dtype).at[:rank].set(x_permuted_head)
    x = jnp.zeros_like(x_permuted).at[pivots].set(x_permuted)

    return x,rank



@jax.jit
def _penalized_logistic_loss(beta,Xb,y,l2):
    '''Method Function:

        Define the stable mean binary cross-entropy loss with L2 regularization. The intercept in the last parameter position is not penalized

    :parameters:
        - beta (jnp.ndarray) - (p+1,) Feature weights and intercept in the standardized space
        - Xb (jnp.ndarray) - (n,p+1) Standardized design matrix with an intercept column
        - y (jnp.ndarray) - (n,) Binary observation labels
        - l2 (float) - Non-negative L2 regularization coefficient

    :return:
        - loss (jnp.ndarray) - Scalar regularized mean binary cross-entropy loss
    '''

    eta = Xb @ beta
    ### Stable form of log(1+exp(eta))-y*eta, avoiding log(0) from saturated probabilities
    data_loss = jnp.mean(jnp.logaddexp(0.0,eta) - y * eta)
    penalty = 0.5 * l2 * jnp.sum(beta[:-1] ** 2)

    return data_loss + penalty



@jax.jit
def _logistic_gradient(beta,Xb,y,l2):
    '''Method Function:

        Define the analytic matrix gradient of the regularized Logistic loss, used for Armijo direction checks and independent automatic differentiation verification

    :parameters:
        - beta (jnp.ndarray) - (p+1,) Feature weights and intercept in the standardized space
        - Xb (jnp.ndarray) - (n,p+1) Standardized design matrix with an intercept column
        - y (jnp.ndarray) - (n,) Binary observation labels
        - l2 (float) - Non-negative L2 regularization coefficient

    :return:
        - gradient (jnp.ndarray) - (p+1,) Analytic gradient with no intercept regularization
    '''

    eta = Xb @ beta
    residual = jax.nn.sigmoid(eta) - y
    data_gradient = Xb.T @ residual / Xb.shape[0]
    regularizer = jnp.concatenate([jnp.ones(Xb.shape[1]-1,dtype=Xb.dtype),jnp.zeros(1,dtype=Xb.dtype)])

    return data_gradient + l2 * regularizer * beta



def _fit_qr_logistic(X,y,l2,max_iter,tol,weight_floor,qr_rtol,max_backtracks):
    '''Method Function:

        Define the QR-Logit training function, the main function is to solve regularized binary Logistic regression through IRLS, column-pivoted QR least squares and Armijo backtracking

    :parameters:
        - X (jnp.ndarray) - (n,p) Training feature matrix
        - y (jnp.ndarray) - (n,) Binary training labels
        - l2 (float) - Non-negative L2 regularization coefficient
        - max_iter (int) - Maximum number of IRLS iterations
        - tol (float) - Relative parameter-update convergence tolerance
        - weight_floor (float) - Lower bound for IRLS probability weights
        - qr_rtol (float) - Relative numerical-rank threshold for QR truncation
        - max_backtracks (int) - Maximum number of Armijo step-halving attempts

    :return:
        - result (dict) - Trained standardized parameters, preprocessing statistics and optimization history
    '''

    ### Estimate preprocessing statistics from training data only and handle constant columns safely
    mean = jnp.mean(X,axis=0)
    scale = jnp.std(X,axis=0)
    scale = jnp.where(scale > 1e-12,scale,1.0)
    Xb = _add_intercept((X - mean) / scale)
    n,d1 = Xb.shape

    beta = jnp.zeros(d1,dtype=jnp.float64)
    loss = float(_penalized_logistic_loss(beta,Xb,y,l2))
    loss_history = [loss]
    rank_history = []
    converged = False

    ### D masks the intercept so that only feature weights enter the L2 penalty
    penalty_mask = jnp.concatenate([jnp.ones(d1-1,dtype=X.dtype),jnp.zeros(1,dtype=X.dtype)])
    penalty_matrix = jnp.diag(penalty_mask)

    for _ in range(max_iter):
        eta = Xb @ beta
        p = jax.nn.sigmoid(eta)
        weights = jnp.clip(p * (1.0 - p),weight_floor,0.25)
        sqrt_w = jnp.sqrt(weights)

        ### Convert the Newton/IRLS step into a weighted least-squares problem
        A_weighted = Xb * sqrt_w[:,None]
        b_weighted = (y - p) / sqrt_w

        ### Mean data loss requires n*l2 in the unscaled Newton system
        reg_scale = jnp.sqrt(n * l2)
        A_reg = reg_scale * penalty_matrix
        b_reg = -reg_scale * (penalty_matrix @ beta)
        A = jnp.concatenate([A_weighted,A_reg],axis=0)
        b = jnp.concatenate([b_weighted,b_reg],axis=0)

        delta,rank = _pivoted_qr_lstsq(A,b,rtol=qr_rtol)
        rank_history.append(rank)

        if float(jnp.linalg.norm(delta)) <= tol * (1.0 + float(jnp.linalg.norm(beta))):
            converged = True
            break

        gradient = _logistic_gradient(beta,Xb,y,l2)
        directional_derivative = float(jnp.dot(gradient,delta))
        if not jnp.isfinite(directional_derivative) or directional_derivative >= 0.0:
            raise RuntimeError("QR step is not a finite descent direction; check the rank threshold, regularization and input data")

        ### Accept only a finite candidate satisfying the Armijo sufficient-decrease condition
        accepted = False
        step_size = 1.0
        for _ in range(max_backtracks):
            candidate = beta + step_size * delta
            candidate_loss = float(_penalized_logistic_loss(candidate,Xb,y,l2))
            armijo_bound = loss + 1e-4 * step_size * directional_derivative
            if jnp.isfinite(candidate_loss) and candidate_loss <= armijo_bound:
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            raise RuntimeError("Armijo backtracking failed; increase l2 or inspect the data for complete separation")

        beta = candidate
        loss = candidate_loss
        loss_history.append(loss)

        relative_update = float(jnp.linalg.norm(step_size * delta)) / (1.0 + float(jnp.linalg.norm(beta)))
        if relative_update <= tol:
            converged = True
            break

    return {
        "beta_hat": beta,
        "mean": mean,
        "scale": scale,
        "loss_history": loss_history,
        "rank_history": rank_history,
        "converged": converged,
        "n_iter": len(loss_history) - 1
    }



@jax.jit
def _predict_logistic_proba(A_test,beta_hat,mean,scale):
    '''Method Function:

        Define a JIT-compiled Logistic probability inference function, reusing training-set standardization and appending the intercept in the same position as training

    :parameters:
        - A_test (jnp.ndarray) - (m,p) Test feature matrix
        - beta_hat (jnp.ndarray) - (p+1,) Trained standardized feature weights and intercept
        - mean (jnp.ndarray) - (p,) Training-set feature means
        - scale (jnp.ndarray) - (p,) Training-set feature standard deviations

    :return:
        - probabilities (jnp.ndarray) - (m,2) Negative-class and positive-class probabilities
    '''

    Xb = _add_intercept((A_test - mean) / scale)
    positive = jax.nn.sigmoid(Xb @ beta_hat)
    probabilities = jnp.stack([1.0 - positive,positive],axis=1)

    return probabilities



class QRLogisticComponent(BaseRegression):
    '''Class Introduction:

        This is a binary Logistic regression component based on iteratively reweighted least squares (IRLS) and column-pivoted Householder QR decomposition. The main functions are stable L2-regularized training, numerical-rank diagnostics, Armijo-protected parameter updates, two-class probability inference and threshold-based classification. It is intended for small and medium dense data, teaching, algorithm verification and numerically sensitive reference workflows

    Mathematical principle
    ----------------------
    (1) Binary Logistic model: given n samples and p features, the inputs are $X in R^(n*p)$ and $y in {0,1}^n$ (X ∈ R^(n×p), y ∈ {0,1}ⁿ). Features are standardized using training-set statistics only, then a constant column is appended:

        $X_tilde = [X_scaled,1] , beta = [w,b]$
        X̃ = [X_scaled, 1] ∈ R^(n×(p+1)), β = [w, b]ᵀ, where b is the unpenalized intercept

        $eta = X_tilde*beta , p_i = P(y_i=1|x_i,beta) = sigmoid(eta_i)$
        η = X̃β, p_i = σ(η_i), σ(η) = 1/(1+e^(-η)), with Sigmoid applied element-wise

    (2) Stable regularized objective: the component minimizes mean binary cross-entropy plus an L2 penalty on feature weights only:

        $J(beta) = (1/n)*sum_i[log(1+exp(eta_i))-y_i*eta_i] + (l2/2)*||w||_2^2 , l2 >= 0$
        J(β) = (1/n)Σᵢ[log(1+e^ηᵢ)−yᵢηᵢ] + (λ/2)‖w‖₂², λ = l2 ≥ 0

        The expression log(1+exp(eta))-y*eta is equivalent to -y*log(p)-(1-y)*log(1-p), but jnp.logaddexp avoids evaluating log(0) when Sigmoid probabilities saturate. The last parameter is the intercept and is excluded from both the loss penalty and its gradient

    (3) Gradient and Hessian: let $W=diag(w_i)$ with $w_i=p_i*(1-p_i)$ (W = diag(wᵢ), wᵢ = pᵢ(1−pᵢ)), and let $D=diag(1,...,1,0)$ (D = diag(1,…,1,0)) mask the intercept:

        $g = (1/n)*X_tilde^T*(p-y) + l2*D*beta$
        g = (1/n)X̃ᵀ(p−y) + λDβ

        $H = (1/n)*X_tilde^T*W*X_tilde + l2*D$
        H = (1/n)X̃ᵀWX̃ + λD

        The Newton direction satisfies H*delta=-g. Multiplication by n gives the unscaled system actually encoded by the weighted least-squares rows:

        $(X_tilde^T*W*X_tilde+n*l2*D)*delta = X_tilde^T*(y-p)-n*l2*D*beta$
        (X̃ᵀWX̃+nλD)δ = X̃ᵀ(y−p)−nλDβ

        The n*l2 factor is essential because the data loss is averaged while the L2 penalty is not averaged

    (4) Equivalent IRLS least-squares system: define $S=W^(1/2)$ and $r_lambda=sqrt(n*l2)$ (S = W^(1/2), r_λ = √(nλ)). Each iteration constructs:

        $A = [S*X_tilde ; r_lambda*D] , c = [S^(-1)*(y-p) ; -r_lambda*D*beta]$
        A = [SX̃; √(nλ)D], c = [S⁻¹(y−p); −√(nλ)Dβ]

        and solves $min_delta ||A*delta-c||_2^2$ (min_δ ‖Aδ−c‖₂²). Its normal equations are exactly the Newton system above, but the implementation never explicitly forms X̃ᵀWX̃, AᵀA or a matrix inverse. This is also the step form of classical IRLS with working response $z_i=x_tilde_i^T*beta+(y_i-p_i)/w_i$ (zᵢ = x̃ᵢᵀβ+(yᵢ−pᵢ)/wᵢ)

    (5) Householder QR and column pivoting: a Householder reflection $H=I-2*v*v^T/(v^T*v)$ (H = I−2vvᵀ/(vᵀv)) is symmetric and orthogonal. Column-pivoted QR computes:

        $A*P = Q*R , A[:,pivots] = Q*R$
        AP = QR, equivalently A[:, pivots] = QR

        Orthogonal transformations preserve the vector 2-norm and avoid the condition-number squaring caused by explicitly constructing AᵀA. If the estimated numerical rank is r, only the reliable leading system $R_11*u=(Q^T*c)_1:r$ (R₁₁u = (Qᵀc)₁:ᵣ) is solved; remaining permuted components are zero and .at[pivots].set(...) restores original column order

        Rank r is estimated by $|R_ii| > qr_rtol*|R_00|$ (|Rᵢᵢ| > qr_rtol·|R₀₀|). This yields a basic solution determined by pivoted columns, not necessarily the SVD minimum-norm solution, and diagonal-based rank revelation is not infallible for every ill-conditioned matrix

    (6) Why IRLS + column-pivoted QR: the method is selected for numerical transparency and stability, not because it is fastest at every scale:

        ==================  ================================  ==========================================  ==================================
        Method              Main per-iteration work            Numerical/storage property               Typical scenario
        ==================  ================================  ==========================================  ==================================
        Gradient descent    O(n*p) matrix operations            Simple and memory efficient; many steps  Baseline and large simple training
        L-BFGS              Matrix-vector products              No explicit Hessian; general purpose     Medium/high-dimensional optimization
        Newton-CG           Hessian-vector products             No complete Hessian required             High-dimensional or sparse problems
        IRLS + Cholesky     Hessian construction + O(p^3)        Fast for stable positive definite H     Small dense regularized problems
        IRLS + pivoted QR   Weighted QR, about O(n*p^2), n>>p    No Hessian inverse; rank diagnostics    Dense reference and collinear data
        IRLS + SVD          Usually higher decomposition cost    Direct spectrum and rank analysis      Small ill-conditioned diagnostics
        ==================  ================================  ==========================================  ==================================

        For very high-dimensional, strongly sparse or very large data, L-BFGS, Newton-CG or coordinate methods may be more appropriate

    Algorithm flow
    --------------
    .. code-block:: text

        Input A(n×p), b(n,), l2 and optimization controls
            |
            v
        Validate finite binary inputs and compute training-only mean / scale
            |
            v
        Standardize features -> append intercept -> initialize beta = 0
            |
            v
        eta = Xb*beta -> p = sigmoid(eta) -> w = clip(p*(1-p))
            |
            v
        Build weighted data rows and sqrt(n*l2) regularization rows
            |
            v
        Column-pivoted economic Householder QR: A[:,pivots] = Q*R
            |
            v
        Estimate rank -> truncated R11 triangular solve -> restore column order
            |
            v
        Check g.T*delta < 0 -> Armijo backtracking -> beta = beta + alpha*delta
            |
            +------------> Record finite loss, numerical rank and convergence state
            |
            v
        Inference: reuse mean / scale -> append intercept -> Sigmoid
            |
            +------------> predict_proba: [1-p1,p1]; reasoning: p1 >= threshold

    Step 1 (input and precision): convert inputs to float64, require A(n,p) and b(n,) with matching non-empty samples, finite values and labels in {0,1}. The module enables jax_enable_x64 before component arrays are created

    Step 2 (training-only preprocessing): compute $mu_j=(1/n)*sum_iX_ij$ and $s_j=std(X_:,j)$ (μⱼ = (1/n)ΣᵢXᵢⱼ, sⱼ = std(X:,j)). Replace near-zero scale by 1, standardize, then append the intercept. Test data never participates in estimating these statistics

    Step 3 (initialization): set beta=0 in the standardized augmented space. All initial positive-class probabilities are 0.5 and initial unregularized data loss is log(2)

    Step 4 (IRLS weights): compute eta, p and $w_i=clip(p_i*(1-p_i),weight_floor,0.25)$ (wᵢ = clip(pᵢ(1−pᵢ), weight_floor, 1/4)). The floor is numerical protection, not part of the original statistical model

    Step 5 (weighted rows): construct $A_weighted=sqrt(W)*X_tilde$ and $c_weighted=(y-p)/sqrt(W)$ (A_weighted = √W·X̃, c_weighted = (y−p)/√W) without materializing an n×n diagonal W

    Step 6 (regularization rows): append sqrt(n*l2)*D and -sqrt(n*l2)*D*beta. This encodes both nλD in the Hessian and −nλDβ on the right-hand side while leaving the intercept unpenalized

    Step 7 (pivoted QR): call jax.scipy.linalg.qr(A,mode="economic",pivoting=True). Economic QR avoids unneeded full-Q dimensions; Householder transformations provide the stable orthogonal factorization

    Step 8 (rank truncation): use diag(R) and qr_rtol to select r, solve only R11 by upper triangular back substitution, set unreliable directions to zero and restore the original parameter order with pivots

    Step 9 (descent and Armijo): compute the analytic gradient and require $g^T*delta<0$ (gᵀδ < 0). Try alpha in {1,1/2,1/4,...} until:

        $J(beta+alpha*delta) <= J(beta)+c1*alpha*g^T*delta , c1=1e-4$
        J(β+αδ) ≤ J(β)+c₁αgᵀδ, c₁ = 10⁻⁴

        Non-finite candidates and non-descent directions are rejected explicitly rather than silently accepted

    Step 10 (update and convergence): update beta, record loss and stop when $||alpha*delta||_2/(1+||beta||_2)<=tol$ (‖αδ‖₂/(1+‖β‖₂) ≤ tol) or max_iter is reached. loss_history includes the initial loss, so n_iter=len(loss_history)-1; rank_history can contain one additional final QR evaluation without a parameter update

    Step 11 (probability inference): reuse mean and scale, append the intercept and compute $p1=sigmoid(X_tilde_new*beta)$ (p₁ = σ(X̃_newβ)). predict_proba returns [1-p1,p1] with shape (m,2), and each row sums to 1

    Step 12 (class inference): reasoning returns int32 values 0 or 1 according to p1>=threshold. The threshold changes only the decision rule, not the trained probability model

    JAX unique advantages
    ---------------------
    (1) Explicit precision and vectorized matrix computation: module-level float64 is used for repeated QR, weight division and triangular solves. Matrix products and row weighting avoid Python sample loops

    (2) Selective JIT compilation: the pure loss, analytic gradient and fixed-shape probability inference use @jax.jit. The dynamic-rank QR solver intentionally does not, because rank=int(...) synchronizes with the host and R[:rank,:rank] requires static shapes under JIT

    (3) Automatic differentiation as an independent checker: training directions come from QR rather than gradient descent, while jax.grad can independently verify the analytic gradient:

    .. code-block:: python

        import jax
        from statjax.regression._components import _penalized_logistic_loss,_logistic_gradient

        manual = _logistic_gradient(beta,Xb,y,l2)
        automatic = jax.grad(_penalized_logistic_loss,argnums=0)(beta,Xb,y,l2)
        relative_error = jax.numpy.linalg.norm(manual-automatic) / (1.0+jax.numpy.linalg.norm(automatic))

    (4) Immutable functional updates: JAX arrays use .at[pivots].set(...) to restore pivoted solutions without in-place mutation. This makes the permutation explicit and compatible with JAX array semantics

    (5) Reproducible verification: tests derive data and split keys from explicit PRNGKey values. Cross-version, backend or hardware bitwise identity is not assumed, especially when pivots tie or diagonal values approach the rank threshold

    (6) Backend migration with explicit boundaries: matrix operations can run on supported JAX backends, but pivoted QR support, performance and determinism must be verified per backend. CPU is the reference environment. Host conversions such as float(...) and int(...) prioritize clear optimization control over maximum throughput; benchmarks must use block_until_ready() for correct asynchronous timing

    Precautions
    -----------
    (1) QR is not universally fastest: each iteration performs a matrix decomposition. Assess alternative optimizers for high-dimensional, sparse or very large workloads

    (2) Never explicitly invert the Hessian: theoretical inverse notation is only mathematical shorthand. This implementation solves weighted least squares directly through QR and triangular back substitution

    (3) Never compute log(sigmoid(eta)) directly: saturated probabilities can produce log(0). The stable logaddexp formulation is required

    (4) weight_floor changes extreme-region Newton equations: persistent activation can indicate complete separation, poor feature scaling or insufficient regularization; do not hide the issue by only increasing max_iter

    (5) Match regularization scaling: mean data loss requires sqrt(n*l2) augmentation. Changing the loss to a sum requires a corresponding change to the QR system

    (6) Keep the intercept unpenalized: loss, analytic gradient and QR regularization rows must all use the same D mask; changing only one location makes the algorithm internally inconsistent

    (7) qr_rtol has no universal value: too small can retain noise directions, while too large can discard informative directions. Inspect rank_history, residuals and validation behavior

    (8) Rank-truncated QR is not an SVD minimum-norm solver: use SVD separately when a singular spectrum, systematic rank analysis or minimum-norm solution is required

    (9) Complete separation can make unregularized maximum-likelihood parameters diverge: prefer a small positive l2 and monitor parameter norms, losses and converged status

    (10) Reuse training preprocessing: inference and deployment must use the stored mean and scale in the original order standardize -> append intercept -> matrix multiply -> Sigmoid

    (11) Respect pivot conventions: this component uses A[:,pivots]=Q@R and .at[pivots].set(...). Reverify conventions before replacing the QR implementation

    (12) Dynamic rank causes synchronization: rank=int(...) and Python slicing are deliberate reference-implementation boundaries, not a maximum-throughput design

    (13) Accuracy is not a sufficient correctness test: retain gradient, QR rank/residual, monotone loss and probability checks because apparently reasonable labels can conceal mathematical errors

    (14) Production use needs additional engineering: model serialization, stronger schema checks, outlier handling, threshold calibration and genuine cross-validation are outside this component's current scope

    :example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from statjax.regression import QRLogisticComponent
        >>> # Generate reproducible correlated features and binary labels
        >>> keys = jax.random.split(jax.random.PRNGKey(42),5)
        >>> latent = jax.random.normal(keys[0],(600,4),dtype=jnp.float64)
        >>> mixing = jax.random.normal(keys[1],(4,8),dtype=jnp.float64) / 2.0
        >>> A = latent @ mixing + 0.35 * jax.random.normal(keys[2],(600,8),dtype=jnp.float64)
        >>> true_w = 0.8 * jax.random.normal(keys[3],(8,),dtype=jnp.float64)
        >>> b = jax.random.bernoulli(keys[4],jax.nn.sigmoid(0.35+A@true_w)).astype(jnp.float64)
        >>> # Split before fitting so standardization uses training data only
        >>> A_train,A_test = A[:480],A[480:]
        >>> b_train,b_test = b[:480],b[480:]
        >>> logistic_instance = QRLogisticComponent(l2=1e-6)
        >>> beta_hat,metrics = logistic_instance.train(A=A_train,b=b_train)
        >>> probabilities = logistic_instance.predict_proba(A_test=A_test)
        >>> y_pred = logistic_instance.reasoning(A_test=A_test)
        >>> accuracy = jnp.mean(y_pred == b_test.astype(jnp.int32))

    :reference:
        - [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
        - [Iteratively reweighted least squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)
        - [Householder transformation](https://en.wikipedia.org/wiki/Householder_transformation)
        - [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition)
        - [Backtracking line search](https://en.wikipedia.org/wiki/Backtracking_line_search)
        - StatJAX004 Logistic Design Document `"StatJAX004 Logistic SH V001"<https://github.com/redblue0216/statjax>`_
    '''


    def __init__(self,l2=1e-6,max_iter=50,tol=1e-8,weight_floor=1e-8,qr_rtol=1e-12,max_backtracks=20,threshold=0.5):
        '''Attribute Function:

            Define an initialization method, mainly used to initialize Logistic optimization hyperparameters and trained model state

        :parameters:
            - l2 (float) - Non-negative L2 regularization coefficient; the intercept is not penalized
            - max_iter (int) - Maximum number of IRLS iterations
            - tol (float) - Relative parameter-update convergence tolerance
            - weight_floor (float) - Positive lower bound no greater than 0.25 for IRLS weights
            - qr_rtol (float) - Positive relative numerical-rank threshold
            - max_backtracks (int) - Maximum number of Armijo backtracking attempts
            - threshold (float) - Positive-class decision threshold in [0,1]
        '''

        if not 0.0 <= l2 < float("inf"):
            raise ValueError("l2 must be finite and non-negative")
        if not isinstance(max_iter,int) or isinstance(max_iter,bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not 0.0 < tol < float("inf"):
            raise ValueError("tol must be finite and positive")
        if not 0.0 < weight_floor <= 0.25:
            raise ValueError("weight_floor must be in (0,0.25]")
        if not 0.0 < qr_rtol < float("inf"):
            raise ValueError("qr_rtol must be finite and positive")
        if not isinstance(max_backtracks,int) or isinstance(max_backtracks,bool) or max_backtracks < 1:
            raise ValueError("max_backtracks must be a positive integer")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")

        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.weight_floor = weight_floor
        self.qr_rtol = qr_rtol
        self.max_backtracks = max_backtracks
        self.threshold = threshold
        ### None means that training has not produced standardized parameters or preprocessing statistics
        self.beta_hat = None
        self.mean = None
        self.scale = None
        self.metrics = None


    def train(self,A,b):
        '''Method Function:

            Define a training method, the main function is to solve binary Logistic regression through standardized IRLS, column-pivoted QR and Armijo backtracking

        :parameters:
            - A (jnp.ndarray) - (n,p) Training feature matrix
            - b (jnp.ndarray) - (n,) Binary training labels containing only 0 and 1

        :return:
            - beta_hat (jnp.ndarray) - (p+1,) Standardized feature weights followed by the intercept
            - metrics (dict) - Loss history, QR rank history, convergence state, iteration count and regularization coefficient
        '''

        A = jnp.asarray(A,dtype=jnp.float64)
        b = jnp.asarray(b,dtype=jnp.float64)
        if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
            raise ValueError("A must have shape (n,p), b must have shape (n,), and their sample counts must match")
        if A.shape[0] == 0 or A.shape[1] == 0:
            raise ValueError("A and b must be non-empty and A must contain at least one feature")
        if not bool(jnp.all(jnp.isfinite(A))) or not bool(jnp.all(jnp.isfinite(b))):
            raise ValueError("A and b must contain only finite values")
        if not bool(jnp.all((b == 0.0) | (b == 1.0))):
            raise ValueError("b must contain only binary labels 0 and 1")

        result = _fit_qr_logistic(A,b,self.l2,self.max_iter,self.tol,self.weight_floor,self.qr_rtol,self.max_backtracks)

        self.beta_hat = result["beta_hat"]
        self.mean = result["mean"]
        self.scale = result["scale"]
        self.metrics = {
            "initial_loss": result["loss_history"][0],
            "final_loss": result["loss_history"][-1],
            "loss_history": result["loss_history"],
            "rank_history": result["rank_history"],
            "converged": result["converged"],
            "n_iter": result["n_iter"],
            "l2": self.l2
        }

        return self.beta_hat,self.metrics


    def predict_proba(self,A_test):
        '''Method Function:

            Define a probability inference method, reusing training-set preprocessing to return negative-class and positive-class probabilities

        :parameters:
            - A_test (jnp.ndarray) - (m,p) Test feature matrix with the training feature dimension

        :return:
            - probabilities (jnp.ndarray) - (m,2) Columns [P(y=0|x),P(y=1|x)]
        '''

        if self.beta_hat is None:
            raise RuntimeError("QRLogisticComponent has not been trained yet, please call the train method first")

        A_test = jnp.asarray(A_test,dtype=jnp.float64)
        if A_test.ndim != 2 or A_test.shape[1] != self.mean.shape[0]:
            raise ValueError("A_test must have shape (m,p) with the same feature dimension as training data")
        if not bool(jnp.all(jnp.isfinite(A_test))):
            raise ValueError("A_test must contain only finite values")

        return _predict_logistic_proba(A_test,self.beta_hat,self.mean,self.scale)


    def reasoning(self,A_test):
        '''Method Function:

            Define a classification inference method, the main function is to threshold positive-class probabilities into binary class predictions

        :parameters:
            - A_test (jnp.ndarray) - (m,p) Test feature matrix with the training feature dimension

        :return:
            - y_pred (jnp.ndarray) - (m,) Binary predictions with dtype int32
        '''

        probabilities = self.predict_proba(A_test)
        y_pred = (probabilities[:,1] >= self.threshold).astype(jnp.int32)

        return y_pred


    def _info(self):
        '''Method Function:

            Define an internal method to obtain basic meta information

        :parameters:
            nothing

        :return:
            - info (str) - Text introduction of the component
        '''

        return "QRLogisticComponent: binary Logistic regression component based on IRLS and column-pivoted Householder QR decomposition, implemented with the JAX technology stack"



##############################################################################################################################################################################
##############################################################################################################################################################################



### Tail protection applies only to IRLS scores and weights; the statistical likelihood always uses original eta
_PROBIT_ETA_TAIL = 30.0


@jax.jit
def _probit_loglik(beta,X,y):
    '''Method Function:

        Define the stable total Probit log-likelihood using normal log-CDF values:
        $ell(beta)=sum_i[y_i*log(Phi(eta_i))+(1-y_i)*log(Phi(-eta_i))]$ (ℓ(β) = Σᵢ[yᵢlogΦ(ηᵢ)+(1−yᵢ)logΦ(−ηᵢ)])

    :parameters:
        - beta (jnp.ndarray) - (p,) Probit regression coefficients
        - X (jnp.ndarray) - (n,p) Design matrix, including any intercept column
        - y (jnp.ndarray) - (n,) Binary observation labels

    :return:
        - loglik (jnp.ndarray) - Scalar total log-likelihood
    '''

    ### Linear predictor eta = X @ beta
    eta = X @ beta
    ### Stable symmetric log-CDF form avoids log(0), tail underflow and log(1-p) cancellation
    log_p = norm.logcdf(eta)
    log_q = norm.logcdf(-eta)
    loglik = jnp.sum(y * log_p + (1.0 - y) * log_q)

    return loglik


@jax.jit
def _probit_mean_nll(beta,X,y):
    '''Method Function:

        Define the mean negative Probit log-likelihood $-ell(beta)/n$ (−ℓ(β)/n) for evaluation and independent automatic differentiation checks

    :parameters:
        - beta (jnp.ndarray) - (p,) Probit regression coefficients
        - X (jnp.ndarray) - (n,p) Design matrix
        - y (jnp.ndarray) - (n,) Binary observation labels

    :return:
        - loss (jnp.ndarray) - Scalar mean negative log-likelihood
    '''

    ### Keep mean NLL distinct from the total log-likelihood used by Armijo acceptance
    loss = -_probit_loglik(beta,X,y) / X.shape[0]

    return loss


def _probit_score_and_weights(beta,X,y):
    '''Method Function:

        Define the per-observation score vector and Fisher weights:
        $u_i=y_i*phi_i/p_i-(1-y_i)*phi_i/q_i$ (uᵢ = yᵢφᵢ/pᵢ−(1−yᵢ)φᵢ/qᵢ) and $w_i=phi_i^2/(p_i*q_i)$ (wᵢ = φᵢ²/(pᵢqᵢ)). Tail protection applies only to these IRLS calculations

    :parameters:
        - beta (jnp.ndarray) - (p,) Probit regression coefficients
        - X (jnp.ndarray) - (n,p) Design matrix
        - y (jnp.ndarray) - (n,) Binary observation labels

    :return:
        - u (jnp.ndarray) - (n,) Score vector with respect to linear predictors
        - w (jnp.ndarray) - (n,) Diagonal Fisher information weights
    '''

    eta = X @ beta
    ### Limit extreme IRLS derivatives without replacing the original likelihood argument
    eta_safe = jnp.clip(eta,-_PROBIT_ETA_TAIL,_PROBIT_ETA_TAIL)
    log_phi = norm.logpdf(eta_safe)
    log_p = norm.logcdf(eta_safe)
    log_q = norm.logcdf(-eta_safe)

    ### Evaluate inverse Mills ratios and Fisher weights in the log domain
    lambda_pos = jnp.exp(log_phi - log_p)
    lambda_neg = jnp.exp(log_phi - log_q)
    u = y * lambda_pos - (1.0 - y) * lambda_neg
    log_w = 2.0 * log_phi - log_p - log_q
    w = jnp.exp(log_w)

    return u,w


@jax.jit
def _probit_nll_gradient(beta,X,y):
    '''Method Function:

        Define the analytic gradient of mean negative Probit log-likelihood:
        $gradient=-(X^T@u)/n$ (gradient = −Xᵀu/n). JAX automatic differentiation independently checks this expression in the unit tests

    :parameters:
        - beta (jnp.ndarray) - (p,) Probit regression coefficients
        - X (jnp.ndarray) - (n,p) Design matrix
        - y (jnp.ndarray) - (n,) Binary observation labels

    :return:
        - gradient (jnp.ndarray) - (p,) Analytic mean negative log-likelihood gradient
    '''

    u,_ = _probit_score_and_weights(beta,X,y)
    gradient = -(X.T @ u) / X.shape[0]

    return gradient


@jax.jit
def _probit_irls_qr_step(beta,X,y):
    '''Method Function:

        Define one Fisher-scoring increment using weighted least squares and column-pivoted Householder QR. With $Xw=sqrt(W)*X$ and $h=u/sqrt(w)$ (Xw = √W·X, h = u/√W), solving $min_delta||Xw@delta-h||_2^2$ (min_δ‖Xwδ−h‖₂²) is equivalent to $(X^T*W*X)@delta=X^T@u$ ((XᵀWX)δ = Xᵀu), without forming normal equations or a working response

    :parameters:
        - beta (jnp.ndarray) - (p,) Current Probit regression coefficients
        - X (jnp.ndarray) - (n,p) Design matrix
        - y (jnp.ndarray) - (n,) Binary observation labels

    :return:
        - delta (jnp.ndarray) - (p,) Fisher-scoring parameter increment
        - rank (jnp.ndarray) - Estimated numerical rank of the weighted design matrix
        - grad_inf (jnp.ndarray) - Infinity norm of the mean log-likelihood gradient
        - slope (jnp.ndarray) - Total log-likelihood directional derivative
        - qr_error (jnp.ndarray) - Relative QR reconstruction error
    '''

    ### Step 1: construct weighted rows by broadcasting, without materializing W
    u,w = _probit_score_and_weights(beta,X,y)
    sqrt_w = jnp.sqrt(w)
    Xw = X * sqrt_w[:,None]
    h = u / sqrt_w

    ### Step 2: economic column-pivoted Householder QR uses Xw[:,piv] = Q @ R
    Q,R,piv = qr(Xw,mode="economic",pivoting=True,check_finite=False)
    diag_r = jnp.abs(jnp.diag(R))
    eps = jnp.finfo(X.dtype).eps
    ### Engineering rank threshold tau = max(n,p)*eps*|R[0,0]|
    rank_tol = max(X.shape) * eps * diag_r[0]
    rank = jnp.sum(diag_r > rank_tol)

    ### Step 3: solve the permuted upper triangular system and restore original parameter order
    delta_piv = solve_triangular(R,Q.T @ h,lower=False,check_finite=False)
    delta = jnp.zeros_like(beta).at[piv].set(delta_piv)

    ### Step 4: use mean gradient for scale-free convergence and total slope for Armijo
    grad = X.T @ u
    grad_inf = jnp.linalg.norm(grad / X.shape[0],ord=jnp.inf)
    slope = grad @ delta
    qr_error = jnp.linalg.norm(Xw[:,piv] - Q @ R) / jnp.linalg.norm(Xw)

    return delta,rank,grad_inf,slope,qr_error


def _fit_qr_probit(X,y,max_iter,grad_tol,step_tol,ll_tol,max_halvings):
    '''Method Function:

        Define Probit training through Fisher scoring, column-pivoted QR and Armijo step halving. Accepted steps satisfy $ell(beta+alpha*delta)>=ell(beta)+1e-4*alpha*g^T*delta$ (ℓ(β+αδ) ≥ ℓ(β)+10⁻⁴αgᵀδ). The host loop preserves explicit rank, direction and convergence decisions

    :parameters:
        - X (jnp.ndarray) - (n,p) Validated design matrix
        - y (jnp.ndarray) - (n,) Validated binary labels
        - max_iter (int) - Maximum number of Fisher-scoring iterations
        - grad_tol (float) - Mean gradient infinity-norm convergence tolerance
        - step_tol (float) - Relative parameter-step convergence tolerance
        - ll_tol (float) - Relative log-likelihood-change convergence tolerance
        - max_halvings (int) - Maximum number of Armijo step-halving attempts

    :return:
        - result (dict) - Trained parameters, convergence state and optimization diagnostics
    '''

    ### Step 1: zero initialization gives eta=0 and p=q=1/2; history includes initial likelihood
    beta = jnp.zeros(X.shape[1],dtype=jnp.float64)
    ll = float(_probit_loglik(beta,X,y))
    loglik_history = [ll]
    converged = False
    n_iter = 0

    for iteration in range(1,max_iter + 1):
        ### Step 2: obtain the Fisher/QR increment and synchronize scalar diagnostics with the host
        delta,rank,grad_inf,slope,qr_error = _probit_irls_qr_step(beta,X,y)
        rank,grad_inf,slope,qr_error = map(float,(rank,grad_inf,slope,qr_error))

        ### Step 3: rank deficiency is an explicit modeling error, never silent coefficient truncation
        if rank < X.shape[1]:
            raise np.linalg.LinAlgError(
                f"Weighted design matrix has numerical rank {rank} < {X.shape[1]}; "
                "remove redundant columns or use an explicitly regularized model"
            )
        if qr_error > 1e-10:
            raise RuntimeError(f"QR reconstruction check failed with relative error {qr_error:.3e}")
        ### Step 4: mean gradient is the primary first-order convergence criterion
        if grad_inf < grad_tol:
            converged = True
            n_iter = iteration - 1
            break
        if not np.isfinite(slope) or slope <= 0.0:
            raise RuntimeError(f"IRLS direction is not a finite ascent direction: g^T delta={slope}")

        ### Step 5: try the full Fisher step before Armijo step halving
        accepted = False
        step = 1.0
        for _ in range(max_halvings + 1):
            candidate = beta + step * delta
            candidate_ll = float(_probit_loglik(candidate,X,y))
            ### Sufficient ascent uses total log-likelihood and total directional derivative
            if np.isfinite(candidate_ll) and candidate_ll >= ll + 1e-4 * step * slope:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise RuntimeError("Armijo step halving failed to find an acceptable Probit update")

        ### Step 6: accept the update and record monotone-likelihood and relative-change diagnostics
        step_inf = float(jnp.max(jnp.abs(step * delta)))
        beta,ll = candidate,candidate_ll
        loglik_history.append(ll)
        n_iter = iteration
        ll_change = abs(loglik_history[-1] - loglik_history[-2]) / max(1.0,abs(loglik_history[-2]))
        beta_scale = float(jnp.max(jnp.abs(beta)))

        ### Step 7: small step and likelihood change are necessary but not sufficient; recheck gradient
        if step_inf <= step_tol * (1.0 + beta_scale) and ll_change <= ll_tol:
            final_grad = float(jnp.linalg.norm(_probit_nll_gradient(beta,X,y),ord=jnp.inf))
            if final_grad < grad_tol:
                grad_inf = final_grad
                converged = True
                break

    ### Recompute the final gradient at zero-iteration convergence, iteration limit or non-convergence
    if converged and n_iter == 0:
        grad_inf = float(jnp.linalg.norm(_probit_nll_gradient(beta,X,y),ord=jnp.inf))
    elif not converged or n_iter == max_iter:
        grad_inf = float(jnp.linalg.norm(_probit_nll_gradient(beta,X,y),ord=jnp.inf))

    return {
        "beta_hat": beta,
        "loglik_history": loglik_history,
        "converged": converged,
        "n_iter": n_iter,
        "grad_inf": float(grad_inf)
    }


@jax.jit
def _predict_probit_proba(A_test,beta_hat):
    '''Method Function:

        Define JIT-compiled Probit probability inference using $p1=Phi(A_test@beta_hat)$ (p₁ = Φ(A_testβ̂)) and returning [1-p1,p1]

    :parameters:
        - A_test (jnp.ndarray) - (m,p) Test design matrix with training column order
        - beta_hat (jnp.ndarray) - (p,) Trained Probit regression coefficients

    :return:
        - probabilities (jnp.ndarray) - (m,2) Negative-class and positive-class probabilities
    '''

    ### Positive probability is a normal CDF prediction, not a latent-variable prediction
    positive = norm.cdf(A_test @ beta_hat)
    probabilities = jnp.stack([1.0 - positive,positive],axis=1)

    return probabilities


class QRProbitComponent(BaseRegression):
    '''Class Introduction:

        This is a binary Probit regression component based on iteratively reweighted least squares (IRLS), Fisher scoring, column-pivoted Householder QR decomposition and Armijo step halving. The main functions are unregularized maximum-likelihood training, numerical-rank diagnostics, two-class probability inference and threshold-based classification. It is intended for small and medium dense data, teaching, algorithm verification and numerically transparent reference workflows. The input design matrix must contain any required intercept column

    Mathematical principle
    ----------------------
    (1) Binary Probit model: given n samples and p columns, the inputs are $X in R^(n*p)$ and $y in {0,1}^n$ (X ∈ R^(n×p), y ∈ {0,1}ⁿ). The caller controls column order and includes a constant column when an intercept is required. For observation i:

        $eta_i = x_i^T*beta , p_i = P(y_i=1|x_i,beta) = Phi(eta_i)$
        ηᵢ = xᵢᵀβ, pᵢ = P(yᵢ=1|xᵢ,β) = Φ(ηᵢ)

        $Phi(t) = integral_{-infinity}^{t} phi(s)*ds , phi(t) = exp(-t^2/2)/sqrt(2*pi)$
        Φ(t) = ∫₋∞ᵗφ(s)ds, φ(t) = exp(−t²/2)/√(2π), where Φ and φ are the standard normal CDF and PDF

        The equivalent latent-variable model is $y_i^*=x_i^T*beta+epsilon_i$ (yᵢ* = xᵢᵀβ+εᵢ), $epsilon_i~N(0,1)$ (εᵢ ~ N(0,1)), and $y_i=1(y_i^*>0)$ (yᵢ = 1(yᵢ*>0)). Fixing the error variance to 1 identifies the parameter scale; otherwise beta and the error scale cannot both be identified

    (2) Stable log-likelihood and concavity: the component maximizes

        $ell(beta) = sum_i[y_i*log(Phi(eta_i))+(1-y_i)*log(Phi(-eta_i))]$
        ℓ(β) = Σᵢ[yᵢlogΦ(ηᵢ)+(1−yᵢ)logΦ(−ηᵢ)]

        Both logΦ(η) and logΦ(−η) are concave, so the Probit log-likelihood is concave in beta. It is strictly concave with a unique maximizer when X has full column rank and a finite maximum exists. norm.logcdf(eta) and norm.logcdf(-eta) avoid evaluating log(0) after CDF underflow and avoid catastrophic cancellation in log(1-Phi(eta)). The likelihood always uses the original eta, never the clipped IRLS value

    (3) Score vector and Fisher information: let $p_i=Phi(eta_i)$, $q_i=Phi(-eta_i)$ and $phi_i=phi(eta_i)$ (pᵢ = Φ(ηᵢ), qᵢ = Φ(−ηᵢ), φᵢ = φ(ηᵢ)). The per-observation score with respect to eta is

        $u_i = y_i*phi_i/p_i-(1-y_i)*phi_i/q_i = phi_i*(y_i-p_i)/(p_i*q_i)$
        uᵢ = yᵢφᵢ/pᵢ−(1−yᵢ)φᵢ/qᵢ = φᵢ(yᵢ−pᵢ)/(pᵢqᵢ)

        $g(beta) = X^T*u , w_i = phi_i^2/(p_i*q_i) , A = X^T*W*X$
        g(β) = Xᵀu, wᵢ = φᵢ²/(pᵢqᵢ), A = XᵀWX, where W = diag(wᵢ)

        Fisher scoring solves $A*delta=g$ (Aδ = g) and updates $beta^+=beta+delta$ (β⁺ = β+δ). For nonzero g and positive definite A, $g^T*delta=g^T*A^(-1)*g>0$ (gᵀδ = gᵀA⁻¹g > 0), so delta is an ascent direction. Probit is not a canonical-link Bernoulli model: Fisher scoring is not generally the same as Newton's method with the observed Hessian

    (4) Log-domain scores and tail protection: inverse Mills ratios and weights are evaluated as

        $phi_i/p_i = exp(log(phi_i)-log(p_i)) , phi_i/q_i = exp(log(phi_i)-log(q_i))$
        φᵢ/pᵢ = exp(logφᵢ−logpᵢ), φᵢ/qᵢ = exp(logφᵢ−logqᵢ)

        $w_i = exp(2*log(phi_i)-log(p_i)-log(q_i))$
        wᵢ = exp(2logφᵢ−logpᵢ−logqᵢ)

        Only these IRLS calculations use $eta_safe=clip(eta,-30,30)$ (η_safe = clip(η,−30,30)). This limits numerical underflow and overflow but makes extreme-tail derivatives approximate; it is not a universal remedy for separation or misspecification

    (5) Incremental IRLS as weighted least squares: classical IRLS can use the working response $z_i=eta_i+(y_i-p_i)/phi_i$ (zᵢ = ηᵢ+(yᵢ−pᵢ)/φᵢ), but z can become extreme. Instead, define $X_w=W^(1/2)*X$ and $h=W^(-1/2)*u$ (X_w = W¹ᐟ²X, h = W⁻¹ᐟ²u), then solve

        $min_delta ||X_w*delta-h||_2^2$
        min_δ ‖X_wδ−h‖₂²

        Its normal equations are $X_w^T*X_w*delta=X_w^T*h$ (X_wᵀX_wδ = X_wᵀh), exactly equivalent to $X^T*W*X*delta=X^T*u$ (XᵀWXδ = Xᵀu). The implementation forms neither the n×n matrix W nor the p×p Fisher matrix, and never constructs z

    (6) Householder QR and column pivoting: a Householder reflection $H=I-2*v*v^T/(v^T*v)$ (H = I−2vvᵀ/(vᵀv)) is symmetric and orthogonal. Column-pivoted economic QR computes

        $X_w*P = Q*R , X_w[:,piv] = Q*R$
        X_wP = QR, equivalently X_w[:,piv] = QR

        The implementation solves $R*delta_piv=Q^T*h$ (Rδ_piv = Qᵀh) by triangular back substitution and restores the original parameter order with .at[piv].set(...). Orthogonal transformations avoid the condition-number squaring caused by explicitly forming X_wᵀX_w

        Numerical rank is estimated by $|R_jj| > max(n,p)*eps*|R_00|$ (|Rⱼⱼ| > max(n,p)·ε·|R₀₀|). If rank < p, training raises an explicit linear-algebra exception. Pivoted QR is a useful rank-revealing diagnostic, not an absolute rank guarantee or an SVD minimum-norm solver

    (7) Why IRLS + column-pivoted QR: the method is selected for numerical transparency and stability, not because it is fastest at every scale:

        ==================  ================================  ==========================================  ==================================
        Method              Main per-iteration work            Numerical/storage property               Typical scenario
        ==================  ================================  ==========================================  ==================================
        Normal equations    XᵀWX plus Cholesky                 Condition number is squared               Well-conditioned full-rank data
        Ordinary QR         Weighted economic QR               Stable but weaker rank diagnostics        Full-rank weighted least squares
        IRLS + pivoted QR   Weighted QR, about O(n*p^2), n>=p  No Hessian inverse; rank diagnostics      Dense reference and collinearity checks
        IRLS + SVD          Usually higher decomposition cost  Full spectrum and minimum-norm options    Severe rank deficiency diagnostics
        First-order methods Matrix-vector products             Lower per-step cost; no Fisher curvature   Larger parameter dimensions
        ==================  ================================  ==========================================  ==================================

        For n>=p dense data, each iteration costs approximately O(n*p^2) for economic QR, O(n*p) for weighted rows and O(p^2+n*p) for back substitution and vector operations. Storage is approximately O(n*p+p^2), depending on the backend

    (8) Armijo sufficient ascent: a full Fisher step is tried first, followed by alpha in {1,1/2,1/4,...}. A candidate is accepted only when

        $ell(beta+alpha*delta) >= ell(beta)+c*alpha*g^T*delta , c=1e-4$
        ℓ(β+αδ) ≥ ℓ(β)+cαgᵀδ, c = 10⁻⁴

        Non-finite candidates, non-ascent directions and failed backtracking are reported explicitly. Concavity alone does not make step-size control unnecessary under finite precision

    (9) Convergence criteria: the primary first-order condition is $||g||_inf/n < grad_tol$ (‖g‖∞/n < grad_tol), with default grad_tol=1e-8. After an update, the component also checks

        $||alpha*delta||_inf <= step_tol*(1+||beta||_inf)$
        ‖αδ‖∞ ≤ step_tol·(1+‖β‖∞), default step_tol=1e-9

        $|ell^+-ell|/max(1,|ell|) <= ll_tol$
        |ℓ⁺−ℓ|/max(1,|ℓ|) ≤ ll_tol, default ll_tol=1e-11

        Stable steps and likelihood alone are insufficient: the gradient is checked again before declaring convergence. Reaching max_iter=100 returns converged=False rather than reporting false success

    Algorithm flow
    --------------
    .. code-block:: text

        Input X(n×p), y(n,), optimization controls and threshold
            |
            v
        Validate float64, binary finite inputs, n>=p and caller-defined columns
            |
            v
        Initialize beta = 0 -> eta = 0 -> p = q = 1/2 -> record ell(beta)
            |
            v
        Compute log(phi), log(p), log(q) -> score u and Fisher weights w
            |
            v
        Build Xw = sqrt(w)[:,None]*X and h = u/sqrt(w)
            |
            v
        Column-pivoted economic Householder QR: Xw[:,piv] = Q*R
            |
            v
        Estimate numerical rank -> stop if rank < p -> triangular solve for delta
            |
            v
        Check QR reconstruction, ||g||inf/n and g.T*delta > 0
            |
            v
        Armijo step halving -> beta = beta + alpha*delta -> record likelihood
            |
            +------------> Joint step / likelihood / gradient convergence checks
            |
            v
        Inference: p1 = Phi(X_new*beta_hat)
            |
            +------------> predict_proba: [1-p1,p1]; reasoning: p1 >= threshold

    Step 1 (input and precision): convert inputs to float64, require A(n,p) and b(n,) with matching non-empty samples, finite values, binary labels and n>=p. The module enables jax_enable_x64 before component arrays are created

    Step 2 (design matrix responsibility): include an intercept and perform any desired preprocessing before calling train. Reuse exactly the same column order and training-derived preprocessing during inference; this component does not estimate mean or scale statistics

    Step 3 (initialization): set beta=0. All initial probabilities are 0.5, scores and Fisher weights are moderate, and the initial total log-likelihood is recorded for monotonicity checks

    Step 4 (stable scores and weights): calculate eta, clip only the IRLS copy to [-30,30], and evaluate normal log-PDF/log-CDF values. Exponentiate log-domain ratios to obtain u and w without forming W

    Step 5 (weighted rows): construct Xw and h by broadcasting sqrt(w). This encodes the Fisher system without a working response or explicit normal equations

    Step 6 (pivoted QR): call jax.scipy.linalg.qr(Xw,mode="economic",pivoting=True). Economic QR avoids unnecessary full-Q dimensions and Householder transformations provide stable orthogonal factorization

    Step 7 (rank and triangular solve): estimate numerical rank from diag(R), stop on rank deficiency, solve R by back substitution, and restore delta to original column order using immutable indexed assignment

    Step 8 (diagnostics): compute mean gradient infinity norm, total-likelihood directional derivative and relative QR reconstruction error. Rank, direction and decomposition failures are not silently accepted

    Step 9 (Armijo update): try the full step and halve until sufficient ascent is achieved or max_halvings is exhausted. The accepted likelihood is appended to history

    Step 10 (convergence): stop immediately when the mean gradient meets grad_tol. Otherwise require small relative step and likelihood change, then recheck the gradient; continue if the gradient is still too large

    Step 11 (probability inference): compute $p1=Phi(A_test*beta_hat)$ (p₁ = Φ(A_testβ̂)). predict_proba returns [1-p1,p1] with shape (m,2), and each row sums to 1

    Step 12 (class inference): reasoning returns int32 values 0 or 1 according to p1>=threshold. The threshold changes only the decision rule, not the trained probability model

    JAX unique advantages
    ---------------------
    (1) Explicit float64 and vectorized matrix computation: repeated QR, tail weights and small likelihood changes benefit from float64. Matrix products and row weighting avoid Python sample loops and n×n diagonal weight allocation

    (2) Selective JIT compilation: the stable likelihood, mean NLL, analytic gradient, fixed-shape Fisher/QR step and probability inference use @jax.jit. The outer training loop deliberately remains in Python for clear rank exceptions, Armijo control and convergence diagnostics

    (3) Automatic differentiation as an independent checker: training directions come from Fisher scoring and QR rather than gradient descent, while jax.grad can independently verify the analytic gradient:

    .. code-block:: python

        import jax
        from statjax.regression._components import _probit_mean_nll,_probit_nll_gradient

        manual = _probit_nll_gradient(beta,X,y)
        automatic = jax.grad(_probit_mean_nll)(beta,X,y)
        relative_error = jax.numpy.linalg.norm(manual-automatic) / (1.0+jax.numpy.linalg.norm(automatic))

    (4) Immutable functional updates: .at[piv].set(...) restores the permuted Fisher increment without in-place array mutation, preserving JAX array semantics and making data dependencies explicit

    (5) Reproducible verification: tests generate train and test data from independent explicit PRNGKey values. Cross-version, backend or hardware bitwise identity is not assumed, especially when pivots tie or diagonal values approach the rank threshold

    (6) Backend migration with explicit boundaries: matrix operations can run on supported JAX backends, but pivoted QR support, performance and determinism must be verified per backend. CPU is the reference environment. float(...) conversions synchronize with the host; throughput benchmarks must use block_until_ready()

    Precautions
    -----------
    (1) QR is not universally fastest: every iteration performs a matrix decomposition. Assess iterative or first-order optimizers for high-dimensional, sparse or very large workloads

    (2) Never explicitly invert Fisher information: inverse notation is mathematical shorthand. The implementation solves weighted least squares through QR and triangular back substitution

    (3) Never compute log(Phi(eta)) by evaluating Phi first: stable norm.logcdf is required in both tails. The likelihood uses original eta even though IRLS derivatives use tail protection

    (4) Fisher scoring is not exact Newton: w_i=phi_i^2/(p_i*q_i) defines expected Fisher information, not generally the observed Hessian

    (5) Rank deficiency is an error by design: remove redundant columns or define an explicitly regularized model. This component does not silently truncate coefficients or return an arbitrary basic solution

    (6) Complete separation can make unregularized maximum-likelihood parameters diverge: inspect converged, coefficient norms and likelihood history; QR and backtracking cannot create a finite MLE that does not exist

    (7) ETA tail protection is approximate: frequent activation at |eta|>30 calls for stable asymptotic inverse Mills ratios and data/model review, not merely a larger clip range

    (8) Keep n>=p and use dense inputs: high-dimensional p>n, sparse massive data and streaming data require a different solver or an explicitly changed statistical objective

    (9) Standardization is a caller responsibility: if used, estimate statistics on training data only and reuse them for inference. Preserve intercept placement and feature order

    (10) Python loops cause host synchronization: this is an intentional reference-implementation boundary, not a maximum-throughput design. End-to-end device-resident control would require a separate failure-state protocol

    (11) Regularization changes the estimand: adding a penalty requires corresponding changes to the Fisher system and gradient. Adding jitter only to R would not be equivalent to ordinary Probit MLE

    (12) Standard errors are not provided: a future implementation could estimate covariance from final Fisher information using linear solves and correct pivot restoration, without explicit inversion

    (13) Accuracy is not a sufficient correctness test: retain gradient, QR reconstruction, rank, monotone likelihood, parameter, probability-range, NLL and Brier checks

    (14) Production use needs additional engineering: serialization, stronger schema checks, outlier handling, threshold calibration and genuine cross-validation are outside this component's current scope

    :example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from jax.scipy.stats import norm
        >>> from statjax.regression import QRProbitComponent
        >>> # Generate reproducible Probit data with an explicit intercept column
        >>> x_key,y_key = jax.random.split(jax.random.PRNGKey(2026))
        >>> x = jax.random.normal(x_key,(400,4),dtype=jnp.float64)
        >>> A = jnp.concatenate([jnp.ones((400,1),dtype=jnp.float64),x],axis=1)
        >>> beta_true = jnp.array([0.4,0.8,-0.6,0.5,-0.3],dtype=jnp.float64)
        >>> b = jax.random.bernoulli(y_key,norm.cdf(A@beta_true)).astype(jnp.float64)
        >>> # Train unregularized Probit MLE and perform probability/class inference
        >>> probit_instance = QRProbitComponent()
        >>> beta_hat,metrics = probit_instance.train(A=A,b=b)
        >>> probabilities = probit_instance.predict_proba(A_test=A[:10])
        >>> y_pred = probit_instance.reasoning(A_test=A[:10])
        >>> brier = jnp.mean((probabilities[:,1]-b[:10])**2)

    :reference:
        - [Probit model](https://en.wikipedia.org/wiki/Probit_model)
        - [Iteratively reweighted least squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)
        - [Householder transformation](https://en.wikipedia.org/wiki/Householder_transformation)
        - [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition)
        - [Backtracking line search](https://en.wikipedia.org/wiki/Backtracking_line_search)
        - StatJAX005 Probit Design Document `"StatJAX005 Probit SH V001"<https://github.com/redblue0216/statjax>`_
    '''

    def __init__(self,max_iter=100,grad_tol=1e-8,step_tol=1e-9,ll_tol=1e-11,max_halvings=30,threshold=0.5):
        '''Attribute Function:

            Define an initialization method for Probit optimization controls, classification threshold and trained model state

        :parameters:
            - max_iter (int) - Maximum number of Fisher-scoring iterations
            - grad_tol (float) - Mean gradient infinity-norm convergence tolerance
            - step_tol (float) - Relative parameter-step convergence tolerance
            - ll_tol (float) - Relative log-likelihood-change convergence tolerance
            - max_halvings (int) - Maximum number of Armijo step-halving attempts
            - threshold (float) - Positive-class decision threshold in [0,1]
        '''

        ### Validate host-side optimization controls before any numerical training begins
        if not isinstance(max_iter,int) or isinstance(max_iter,bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not 0.0 < grad_tol < float("inf"):
            raise ValueError("grad_tol must be finite and positive")
        if not 0.0 < step_tol < float("inf"):
            raise ValueError("step_tol must be finite and positive")
        if not 0.0 < ll_tol < float("inf"):
            raise ValueError("ll_tol must be finite and positive")
        if not isinstance(max_halvings,int) or isinstance(max_halvings,bool) or max_halvings < 1:
            raise ValueError("max_halvings must be a positive integer")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")

        self.max_iter = max_iter
        self.grad_tol = grad_tol
        self.step_tol = step_tol
        self.ll_tol = ll_tol
        self.max_halvings = max_halvings
        self.threshold = threshold
        ### None means that training has not produced coefficients or diagnostics
        self.beta_hat = None
        self.metrics = None


    def train(self,A,b):
        '''Method Function:

            Define a training method for unregularized binary Probit maximum-likelihood estimation. The caller owns intercept placement, preprocessing and column order

        :parameters:
            - A (jnp.ndarray) - (n,p) Training design matrix, including any intercept column
            - b (jnp.ndarray) - (n,) Binary training labels containing only 0 and 1

        :return:
            - beta_hat (jnp.ndarray) - (p,) Trained Probit regression coefficients
            - metrics (dict) - Log-likelihood history, convergence state, iteration count and final gradient norm
        '''

        ### Convert to float64 and reject malformed or statistically incompatible inputs before JIT calls
        A = jnp.asarray(A,dtype=jnp.float64)
        b = jnp.asarray(b,dtype=jnp.float64)
        if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
            raise ValueError("A must have shape (n,p), b must have shape (n,), and their sample counts must match")
        if A.shape[0] == 0 or A.shape[1] == 0:
            raise ValueError("A and b must be non-empty and A must contain at least one column")
        if A.shape[0] < A.shape[1]:
            raise ValueError("QRProbitComponent requires n >= p")
        if not bool(jnp.all(jnp.isfinite(A))) or not bool(jnp.all(jnp.isfinite(b))):
            raise ValueError("A and b must contain only finite values")
        if not bool(jnp.all((b == 0.0) | (b == 1.0))):
            raise ValueError("b must contain only binary labels 0 and 1")

        ### Fisher scoring + pivoted QR + Armijo; no automatic intercept, scaling or regularization
        result = _fit_qr_probit(
            A,b,self.max_iter,self.grad_tol,self.step_tol,self.ll_tol,self.max_halvings
        )
        ### Store coefficients and diagnostics for subsequent inference and convergence review
        self.beta_hat = result["beta_hat"]
        self.metrics = {
            "initial_loglik": result["loglik_history"][0],
            "final_loglik": result["loglik_history"][-1],
            "loglik_history": result["loglik_history"],
            "converged": result["converged"],
            "n_iter": result["n_iter"],
            "grad_inf": result["grad_inf"]
        }

        return self.beta_hat,self.metrics


    def predict_proba(self,A_test):
        '''Method Function:

            Define a probability inference method using the trained Probit coefficients and the caller-consistent design columns

        :parameters:
            - A_test (jnp.ndarray) - (m,p) Test design matrix with training column order

        :return:
            - probabilities (jnp.ndarray) - (m,2) Columns [P(y=0|x),P(y=1|x)]
        '''

        ### Both inference interfaces require a successfully populated parameter state
        if self.beta_hat is None:
            raise RuntimeError("QRProbitComponent has not been trained yet, please call the train method first")

        A_test = jnp.asarray(A_test,dtype=jnp.float64)
        if A_test.ndim != 2 or A_test.shape[1] != self.beta_hat.shape[0]:
            raise ValueError("A_test must have shape (m,p) with the same column dimension as training data")
        if not bool(jnp.all(jnp.isfinite(A_test))):
            raise ValueError("A_test must contain only finite values")

        ### Fixed-shape normal-CDF inference is JIT-compiled
        return _predict_probit_proba(A_test,self.beta_hat)


    def reasoning(self,A_test):
        '''Method Function:

            Define a classification inference method based on positive-class Probit probabilities

        :parameters:
            - A_test (jnp.ndarray) - (m,p) Test design matrix with training column order

        :return:
            - y_pred (jnp.ndarray) - (m,) Binary predictions with dtype int32
        '''

        ### Thresholding changes only the decision rule, not the trained probability model
        probabilities = self.predict_proba(A_test)
        y_pred = (probabilities[:,1] >= self.threshold).astype(jnp.int32)

        return y_pred


    def _info(self):
        '''Method Function:

            Define an internal method to obtain basic meta information

        :parameters:
            nothing

        :return:
            - info (str) - Text introduction of the component
        '''

        return "QRProbitComponent: binary Probit regression component based on Fisher scoring and column-pivoted Householder QR decomposition, implemented with the JAX technology stack"



##############################################################################################################################################################################
##############################################################################################################################################################################


def _multinomial_design_matrix(X,mean,scale):
    '''Method Function:

        Define a multinomial design-matrix function that reuses training-set standardization and appends the intercept column

    :parameters:
        - X (jnp.ndarray) - (n,p) Feature matrix
        - mean (jnp.ndarray) - (p,) Training-set feature means
        - scale (jnp.ndarray) - (p,) Training-set feature scales

    :return:
        - Phi (jnp.ndarray) - (n,p+1) Standardized design matrix with the intercept in the last column
    '''

    X_scaled = (X - mean) / scale
    ones = jnp.ones((X.shape[0],1),dtype=X.dtype)
    Phi = jnp.concatenate([X_scaled,ones],axis=1)

    return Phi



def _multinomial_full_logits(Phi,theta,n_classes):
    '''Method Function:

        Define reference-category Softmax logits using row-major parameter unfolding

    :parameters:
        - Phi (jnp.ndarray) - (n,d) Standardized intercept-augmented design matrix
        - theta (jnp.ndarray) - (d*(K-1),) Row-major unfolded free-category parameters
        - n_classes (int) - Total number of classes K

    :return:
        - logits (jnp.ndarray) - (n,K) Free-category logits followed by the zero reference-category logit
    '''

    n,d = Phi.shape
    q = n_classes - 1
    B = theta.reshape((d,q))
    reference = jnp.zeros((n,1),dtype=Phi.dtype)
    logits = jnp.concatenate([Phi @ B,reference],axis=1)

    return logits



def _multinomial_penalty_mask(n_design_columns,n_classes,dtype):
    '''Method Function:

        Define the L2 penalty mask, excluding every free-category intercept parameter

    :parameters:
        - n_design_columns (int) - Number of columns in the augmented design matrix
        - n_classes (int) - Total number of classes K
        - dtype (jnp.dtype) - Parameter array dtype

    :return:
        - penalty (jnp.ndarray) - Flattened feature/intercept penalty mask
    '''

    q = n_classes - 1
    penalty_by_row = jnp.concatenate(
        [jnp.ones(n_design_columns - 1,dtype=dtype),jnp.zeros(1,dtype=dtype)]
    )
    penalty = jnp.repeat(penalty_by_row,q)

    return penalty



def _multinomial_objective(theta,Phi,Y,penalty,l2,n_classes):
    '''Method Function:

        Define the stable unaveraged multinomial negative log-likelihood plus L2 regularization

    :parameters:
        - theta (jnp.ndarray) - Flattened free-category parameters
        - Phi (jnp.ndarray) - Intercept-augmented design matrix
        - Y (jnp.ndarray) - One-hot encoded class labels
        - penalty (jnp.ndarray) - Feature/intercept penalty mask
        - l2 (float) - Non-negative L2 regularization coefficient
        - n_classes (int) - Total number of classes K

    :return:
        - objective (jnp.ndarray) - Scalar regularized negative log-likelihood
    '''

    logits = _multinomial_full_logits(Phi,theta,n_classes)
    nll = jnp.sum(logsumexp(logits,axis=1) - jnp.sum(Y * logits,axis=1))
    objective = nll + 0.5 * l2 * jnp.sum(penalty * theta * theta)

    return objective



def _multinomial_matrix_gradient(theta,Phi,Y,penalty,l2,n_classes):
    '''Method Function:

        Define the analytic matrix gradient of the regularized multinomial objective

    :parameters:
        - theta (jnp.ndarray) - Flattened free-category parameters
        - Phi (jnp.ndarray) - Intercept-augmented design matrix
        - Y (jnp.ndarray) - One-hot encoded class labels
        - penalty (jnp.ndarray) - Feature/intercept penalty mask
        - l2 (float) - Non-negative L2 regularization coefficient
        - n_classes (int) - Total number of classes K

    :return:
        - gradient (jnp.ndarray) - Row-major flattened analytic gradient
    '''

    logits = _multinomial_full_logits(Phi,theta,n_classes)
    probabilities = jax.nn.softmax(logits,axis=1)
    residual = probabilities[:,:-1] - Y[:,:-1]
    data_gradient = (Phi.T @ residual).reshape(-1)
    gradient = data_gradient + l2 * penalty * theta

    return gradient



def _build_multinomial_irls_system(theta,Phi,Y,penalty,n_classes,l2,probability_floor,weight_jitter):
    '''Method Function:

        Define the per-sample whitened and regularization-augmented multinomial IRLS least-squares system

    :parameters:
        - theta (jnp.ndarray) - Current row-major flattened parameters
        - Phi (jnp.ndarray) - Intercept-augmented design matrix
        - Y (jnp.ndarray) - One-hot encoded class labels
        - penalty (jnp.ndarray) - Feature/intercept penalty mask
        - n_classes (int) - Total number of classes K
        - l2 (float) - Non-negative L2 regularization coefficient
        - probability_floor (float) - Probability floor used only for IRLS weights
        - weight_jitter (float) - Positive diagonal damping for Cholesky stability

    :return:
        - A_augmented (jnp.ndarray) - Whitened weighted design with L2 augmentation
        - c_augmented (jnp.ndarray) - Whitened working response with zero regularization response
    '''

    n,d = Phi.shape
    q = n_classes - 1
    dq = d * q
    B = theta.reshape((d,q))
    eta = Phi @ B
    logits = jnp.concatenate([eta,jnp.zeros((n,1),dtype=Phi.dtype)],axis=1)
    probabilities = jax.nn.softmax(logits,axis=1)

    ### Protect only the IRLS weights; the statistical objective uses the original logits
    probabilities = jnp.clip(probabilities,probability_floor,1.0)
    probabilities = probabilities / jnp.sum(probabilities,axis=1,keepdims=True)
    p = probabilities[:,:-1]
    error = Y[:,:-1] - p

    eye_q = jnp.eye(q,dtype=Phi.dtype)
    S = -p[:,:,None] * p[:,None,:]
    S = S + eye_q[None,:,:] * p[:,:,None]
    S = S + weight_jitter * eye_q[None,:,:]
    L = jnp.linalg.cholesky(S)

    def per_sample(L_i,x_i,eta_i,error_i):
        '''Method Function:

            Construct one Cholesky-whitened multinomial IRLS design block and triangular-solved response for jax.vmap
        '''

        ### Row-major theta requires phi_i.T kron I_q
        base_design = jnp.kron(x_i[None,:],eye_q)
        A_i = L_i.T @ base_design
        c_i = L_i.T @ eta_i + solve_triangular(
            L_i,error_i,lower=True,check_finite=False
        )
        return A_i,c_i

    A_blocks,c = jax.vmap(per_sample)(L,Phi,eta,error)
    A = A_blocks.reshape((n * q,dq))
    c = c.reshape((n * q,))

    ### Unaveraged NLL uses sqrt(l2), not the mean-loss sqrt(n*l2) convention
    regularizer = jnp.sqrt(l2) * jnp.diag(penalty)
    A_augmented = jnp.concatenate([A,regularizer],axis=0)
    c_augmented = jnp.concatenate([c,jnp.zeros(dq,dtype=Phi.dtype)])

    return A_augmented,c_augmented



def _multinomial_column_pivoted_householder_qr(A):
    '''Method Function:

        Define economic column-pivoted Householder QR with static column loops and dynamic JAX pivots

    :parameters:
        - A (jnp.ndarray) - (m,n) Least-squares matrix with m>=n

    :return:
        - Q (jnp.ndarray) - (m,n) Economic column-orthogonal factor
        - R (jnp.ndarray) - (n,n) Upper triangular factor
        - permutation (jnp.ndarray) - Column order satisfying A[:,permutation]=Q@R
    '''

    m,n = A.shape
    if m < n:
        raise ValueError("The Householder QR implementation requires m >= n")

    dtype = A.dtype
    permutation = jnp.arange(n)
    work = A
    householder_vectors = jnp.zeros((m,n),dtype=dtype)
    betas = jnp.zeros(n,dtype=dtype)
    tiny = jnp.finfo(dtype).tiny

    for k in range(n):
        trailing = work[k:,k:]
        squared_norms = jnp.sum(trailing * trailing,axis=0)
        pivot = k + jnp.argmax(squared_norms)

        column_k = work[:,k]
        column_p = work[:,pivot]
        work = work.at[:,k].set(column_p)
        work = work.at[:,pivot].set(column_k)

        perm_k = permutation[k]
        perm_p = permutation[pivot]
        permutation = permutation.at[k].set(perm_p)
        permutation = permutation.at[pivot].set(perm_k)

        x = work[k:,k]
        norm_x = jnp.linalg.norm(x)
        alpha = jnp.where(x[0] >= 0.0,-norm_x,norm_x)
        v = x.at[0].add(-alpha)
        v_squared = jnp.sum(v * v)
        safe_v_squared = jnp.where(v_squared > tiny,v_squared,1.0)
        beta = jnp.where(v_squared > tiny,2.0 / safe_v_squared,0.0)

        block = work[k:,k:]
        block = block - beta * v[:,None] * (v @ block)[None,:]
        work = work.at[k:,k:].set(block)
        householder_vectors = householder_vectors.at[k:,k].set(v)
        betas = betas.at[k].set(beta)

        exact_column = jnp.zeros_like(x).at[0].set(alpha)
        work = work.at[k:,k].set(exact_column)

    ### Apply saved reflectors in reverse order to construct economic Q
    Q = jnp.eye(m,n,dtype=dtype)
    for k in range(n - 1,-1,-1):
        v = householder_vectors[k:,k]
        q_block = Q[k:,:]
        q_block = q_block - betas[k] * v[:,None] * (v @ q_block)[None,:]
        Q = Q.at[k:,:].set(q_block)

    R = jnp.triu(work[:n,:])

    return Q,R,permutation



def _multinomial_cpqr_solve(A,b,rank_tol=1e-12):
    '''Method Function:

        Define rank-truncated column-pivoted QR least squares without explicit matrix inversion

    :parameters:
        - A (jnp.ndarray) - (m,n) Weighted augmented least-squares matrix
        - b (jnp.ndarray) - (m,) Least-squares response
        - rank_tol (float) - Relative threshold applied to the diagonal of R

    :return:
        - solution (jnp.ndarray) - Basic least-squares solution in original column order
        - numerical_rank (jnp.ndarray) - Estimated numerical rank
    '''

    Q,R,permutation = _multinomial_column_pivoted_householder_qr(A)
    n = A.shape[1]
    diagonal = jnp.abs(jnp.diag(R))
    diagonal_scale = jnp.maximum(diagonal[0],jnp.finfo(A.dtype).tiny)
    tolerance = rank_tol * diagonal_scale
    numerical_rank = jnp.sum(diagonal > tolerance)
    projected_b = Q.T @ b
    u = jnp.zeros(n,dtype=A.dtype)

    for i in range(n - 1,-1,-1):
        remainder = R[i,i + 1:] @ u[i + 1:]
        active = jnp.logical_and(i < numerical_rank,diagonal[i] > tolerance)
        safe_denominator = jnp.where(active,R[i,i],1.0)
        candidate = (projected_b[i] - remainder) / safe_denominator
        u = u.at[i].set(jnp.where(active,candidate,0.0))

    solution = jnp.zeros(n,dtype=A.dtype).at[permutation].set(u)

    return solution,numerical_rank



def _fit_qr_multinomial_logistic(A,b,n_classes,l2,max_iter,tolerance,rank_tol,probability_floor,weight_jitter,armijo_c1,max_backtracks,check_gradient):
    '''Method Function:

        Define multinomial training through IRLS, handwritten pivoted Householder QR and Armijo backtracking

    :parameters:
        - A (jnp.ndarray) - Validated training feature matrix
        - b (jnp.ndarray) - Validated integer class labels
        - n_classes (int) - Total number of classes K
        - l2 (float) - Non-negative L2 regularization coefficient
        - max_iter (int) - Maximum number of IRLS iterations
        - tolerance (float) - Relative loss-improvement convergence tolerance
        - rank_tol (float) - Relative numerical-rank threshold
        - probability_floor (float) - Probability floor for IRLS weights
        - weight_jitter (float) - Positive IRLS weight damping
        - armijo_c1 (float) - Armijo sufficient-decrease coefficient
        - max_backtracks (int) - Maximum number of step-halving attempts
        - check_gradient (bool) - Whether to check the initial analytic gradient with jax.grad

    :return:
        - result (dict) - Parameters, preprocessing statistics and optimization diagnostics
    '''

    mean = jnp.mean(A,axis=0)
    scale = jnp.std(A,axis=0)
    scale = jnp.where(scale > 1e-12,scale,1.0)
    Phi = _multinomial_design_matrix(A,mean,scale)
    Y = jax.nn.one_hot(b,n_classes,dtype=jnp.float64)
    n,d = Phi.shape
    del n
    q = n_classes - 1
    dq = d * q
    penalty = _multinomial_penalty_mask(d,n_classes,Phi.dtype)
    theta = jnp.zeros(dq,dtype=jnp.float64)

    objective = jax.jit(
        lambda value: _multinomial_objective(value,Phi,Y,penalty,l2,n_classes)
    )
    gradient = jax.jit(
        lambda value: _multinomial_matrix_gradient(value,Phi,Y,penalty,l2,n_classes)
    )

    def irls_candidate(value):
        '''Method Function:

            Build and solve the coupled whitened IRLS system for one prospective multinomial parameter update
        '''

        A_system,c_system = _build_multinomial_irls_system(
            value,Phi,Y,penalty,n_classes,l2,probability_floor,weight_jitter
        )
        candidate,rank = _multinomial_cpqr_solve(A_system,c_system,rank_tol=rank_tol)
        return candidate,rank

    irls_candidate = jax.jit(irls_candidate)

    gradient_error = None
    if check_gradient:
        automatic = jax.grad(
            lambda value: _multinomial_objective(value,Phi,Y,penalty,l2,n_classes)
        )(theta)
        gradient_error = float(jnp.max(jnp.abs(automatic - gradient(theta))))
        if gradient_error > 1e-8:
            raise RuntimeError(f"Analytic multinomial gradient check failed: {gradient_error:.3e}")

    current_loss = float(objective(theta))
    loss_history = [current_loss]
    rank_history = []
    step_history = []
    converged = False
    stop_reason = "max_iter"
    n_iter = 0

    for _ in range(max_iter):
        candidate,rank = irls_candidate(theta)
        current_gradient = gradient(theta)
        direction = candidate - theta
        directional_derivative = float(jnp.dot(current_gradient,direction))

        ### Rank truncation or extreme weights can occasionally destroy descent
        if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
            direction = -current_gradient
            directional_derivative = -float(jnp.dot(current_gradient,current_gradient))

        accepted = False
        step = 1.0
        trial_theta = theta
        trial_loss = current_loss
        for _ in range(max_backtracks + 1):
            proposed = theta + step * direction
            proposed_loss = float(objective(proposed))
            armijo_bound = current_loss + armijo_c1 * step * directional_derivative
            if np.isfinite(proposed_loss) and proposed_loss <= armijo_bound:
                trial_theta,trial_loss = proposed,proposed_loss
                accepted = True
                break
            step *= 0.5

        if not accepted:
            stop_reason = "line_search_failed"
            break

        improvement = current_loss - trial_loss
        theta = trial_theta
        current_loss = trial_loss
        n_iter += 1
        loss_history.append(current_loss)
        rank_history.append(int(rank))
        step_history.append(float(step))

        if improvement <= tolerance * max(1.0,abs(current_loss)):
            converged = True
            stop_reason = "converged"
            break

    return {
        "theta_hat": theta,
        "mean": mean,
        "scale": scale,
        "loss_history": loss_history,
        "rank_history": rank_history,
        "step_history": step_history,
        "gradient_check_error": gradient_error,
        "converged": converged,
        "stop_reason": stop_reason,
        "n_iter": n_iter
    }



@jax.jit(static_argnums=(4,))
def _predict_multinomial_logistic_proba(A_test,theta_hat,mean,scale,n_classes):
    '''Method Function:

        Define JIT-compiled K-class probability inference using training-set preprocessing

    :parameters:
        - A_test (jnp.ndarray) - (m,p) Test feature matrix
        - theta_hat (jnp.ndarray) - Trained row-major flattened parameters
        - mean (jnp.ndarray) - Training-set feature means
        - scale (jnp.ndarray) - Training-set feature scales
        - n_classes (int) - Static total number of classes K

    :return:
        - probabilities (jnp.ndarray) - (m,K) Class probabilities in label order
    '''

    Phi = _multinomial_design_matrix(A_test,mean,scale)
    logits = _multinomial_full_logits(Phi,theta_hat,n_classes)
    probabilities = jax.nn.softmax(logits,axis=1)

    return probabilities



class QRMultinomialLogisticComponent(BaseRegression):
    '''Class Introduction:

        This is a multinomial Logistic regression component based on reference-category Softmax, jointly coupled iteratively reweighted least squares (IRLS), per-sample Cholesky whitening, handwritten column-pivoted Householder QR and Armijo backtracking. It standardizes features using training-set statistics only, appends an unpenalized intercept and returns K-class probabilities and argmax class predictions. It is intended for small and medium dense data, teaching, algorithm verification and numerically transparent reference workflows

    Mathematical principle
    ----------------------
    (1) Multinomial Logistic model: given n samples, p input features and K classes, the inputs are $X in R^(n*p)$ and integer labels $y_i in {0,...,K-1}$ (X ∈ R^(n×p), yᵢ ∈ {0,…,K−1}). Features are standardized using training-set statistics only, then a constant column is appended:

        $Phi=[X_scaled,1] in R^(n*d) , d=p+1$
        Φ = [X_scaled, 1] ∈ R^(n×d), d = p+1, where the last design-matrix row of parameters contains the class intercepts

        The last class K-1 is the reference category. Only the first $q=K-1$ (q = K−1) categories have free parameters:

        $B in R^(d*q) , theta=vec_row(B) , eta_i=B^T*phi_i$
        B ∈ R^(d×q), θ = vec_row(B), ηᵢ = Bᵀφᵢ, where vec_row means row-major unfolding

        $Z=[Phi*B,0] in R^(n*K) , P_ik=exp(Z_ik)/sum_j(exp(Z_ij))$
        Z = [ΦB, 0] ∈ R^(n×K), Pᵢₖ = exp(Zᵢₖ)/Σⱼexp(Zᵢⱼ)

        Fixing the reference logit to zero selects one identifiable representative of the Softmax equivalence class. Adding the same constant to every class logit leaves all probabilities unchanged, so unconstrained estimation of K complete parameter vectors would contain a common-shift null space

    (2) Stable unaveraged regularized objective: the component minimizes the total, not sample-averaged, negative log-likelihood plus L2 regularization on feature parameters only:

        $J(theta)=sum_i[logsumexp(Z_i)-Z_i,y_i]+(l2/2)*sum_r(penalty_r*theta_r^2)$
        J(θ) = Σᵢ[logsumexp(Zᵢ)−Zᵢ,ᵧᵢ] + (λ/2)Σᵣ(penaltyᵣθᵣ²), λ = l2 ≥ 0

        The penalty mask equals one for standardized feature parameters and zero for every free-category intercept. logsumexp evaluates cross-entropy without explicitly forming log(Softmax), avoiding overflow and log(0) from saturated probabilities. Because the data loss is unaveraged, the IRLS augmentation uses sqrt(l2); it must not use the sqrt(n*l2) convention of QRLogisticComponent, whose data loss is averaged

    (3) Analytic gradient and coupled per-sample Hessian: let $Y$ (Y) be the one-hot label matrix and $p_i=P_i,1:q$ (pᵢ = Pᵢ,₁:q). The matrix gradient and its row-major vector form are:

        $G_B=Phi^T*(P_1:q-Y_1:q)$
        G_B = Φᵀ(P₁:q−Y₁:q)

        $g=vec_row(G_B)+l2*penalty*theta$
        g = vec_row(G_B) + λ(penalty ⊙ θ)

        For one observation, the Hessian with respect to its free-category logits is the categorical covariance matrix:

        $S_i=diag(p_i)-p_i*p_i^T$
        Sᵢ = diag(pᵢ)−pᵢpᵢᵀ

        Sᵢ is positive semidefinite and is positive definite when every class probability is strictly positive. Its off-diagonal entries encode coupling between classes. Replacing Sᵢ by diagonal weights or fitting independent binary IRLS models would therefore solve a different problem

        With row-major parameters, the per-sample design is $C_i=phi_i^T kron I_q$ (Cᵢ = φᵢᵀ ⊗ I_q), and the data Hessian is:

        $H_data=sum_i(C_i^T*S_i*C_i)$
        H_data = ΣᵢCᵢᵀSᵢCᵢ

        If column-major unfolding were used, the Kronecker order would have to change to $I_q kron phi_i^T$ (I_q ⊗ φᵢᵀ). The implementation keeps logits, gradients, IRLS design and QR solution restoration in the same row-major convention

    (4) Per-sample Cholesky whitening and IRLS: numerical protection first clips and renormalizes probabilities only for weight construction, then adds $weight_jitter*I$ (weight_jitter·I) to Sᵢ. The protected weight matrix is factorized as:

        $S_i=L_i*L_i^T$
        Sᵢ = LᵢLᵢᵀ

        $A_i=L_i^T*(phi_i^T kron I_q)$
        Aᵢ = Lᵢᵀ(φᵢᵀ ⊗ I_q)

        The multinomial working response is $z_i=eta_i+S_i^(-1)*(y_i-p_i)$ (zᵢ = ηᵢ+Sᵢ⁻¹(yᵢ−pᵢ)), but neither zᵢ nor an explicit inverse is formed. Its whitened response is evaluated as:

        $c_i=L_i^T*eta_i+L_i^(-1)*(y_i-p_i)$
        cᵢ = Lᵢᵀηᵢ+Lᵢ⁻¹(yᵢ−pᵢ)

        The second term uses a lower-triangular solve. Stacking all samples gives A with n*q rows and d*q columns and c with n*q entries. The parameter dimension is $n_theta=d*q$ (n_θ = d·q). L2 is appended directly to the least-squares system:

        $A_tilde=[A;sqrt(l2)*diag(penalty)] , c_tilde=[c;0]$
        Ã = [A; √λ·diag(penalty)], c̃ = [c; 0]

        Without numerical damping, these matrices satisfy:

        $A^T*A=H_data , A^T*c=H_data*theta_old-g_data$
        AᵀA = H_data, Aᵀc = H_dataθ_old−g_data

        $(H_data+l2*D^T*D)*theta_new=H_data*theta_old-g_data$
        (H_data+λDᵀD)θ_new = H_dataθ_old−g_data

        Equivalently, $(H_data+l2*D^T*D)*(theta_new-theta_old)=-(g_data+l2*D^T*D*theta_old)$ ((H_data+λDᵀD)(θ_new−θ_old)=−(g_data+λDᵀDθ_old)). IRLS is therefore the least-squares expression of the regularized Newton step; QR solves that step without explicitly constructing H_data, normal equations or an inverse. Probability clipping and jitter make affected iterations damped approximate Newton steps, not changes to the statistical likelihood

    (5) Handwritten column-pivoted Householder QR: each Householder reflection is $H=I-beta*v*v^T$ with $beta=2/(v^T*v)$ (H = I−βvvᵀ, β = 2/(vᵀv)). Column pivoting computes:

        $A_tilde*Pi=Q*R , A_tilde[:,permutation]=Q*R$
        ÃΠ = QR, equivalently Ã[:, permutation] = QR

        The implementation chooses the largest remaining column norm at each static column step, applies reflections only to the trailing block, saves the vectors and constructs economic Q by applying the saved reflectors in reverse order. Orthogonal transformations avoid the condition-number squaring $kappa(A^T*A)=kappa(A)^2$ (κ(AᵀA)=κ(A)²) caused by explicitly forming normal equations

        Numerical rank is estimated by $|R_ii|>rank_tol*|R_00|$ (|Rᵢᵢ| > rank_tol·|R₀₀|). Back substitution retains reliable pivoted directions and sets free variables to zero. Restoring with .at[permutation].set(...) returns the original parameter order. A rank-deficient result is a basic solution determined by pivoted columns, not necessarily the SVD minimum-norm solution

    (6) Descent protection and Armijo backtracking: QR returns a candidate point, so the actual direction is $Delta_theta=theta_candidate-theta_old$ (Δθ = θ_candidate−θ_old). A valid descent direction satisfies $g^T*Delta_theta<0$ (gᵀΔθ < 0). If this value is non-finite or non-negative, the component falls back to $-g$ (−g). Starting at alpha=1 and halving the step, an accepted point must satisfy:

        $J(theta+alpha*Delta_theta)<=J(theta)+c1*alpha*g^T*Delta_theta$
        J(θ+αΔθ) ≤ J(θ)+c₁αgᵀΔθ, c₁ = armijo_c1

        Non-finite candidates are rejected. If no step is acceptable, training retains the last valid parameters and reports stop_reason='line_search_failed'. Reaching max_iter without meeting the convergence test reports converged=False and stop_reason='max_iter', rather than presenting an iteration limit as statistical convergence

    (7) Why reference-category IRLS plus whitening and pivoted QR: the method is selected for numerical transparency and independent verification, not universal speed:

        ============================  ============================================  ==========================================
        Method                        Main property                                 Typical scenario
        ============================  ============================================  ==========================================
        Full-K Softmax + SGD/Adam     Simple mini-batches; shift non-identifiability  Large, sparse or online training
        Reference Softmax + L-BFGS    No explicit Hessian; general-purpose          Medium and larger dense problems
        Explicit Hessian + Cholesky   Direct Newton solve; Hessian storage           Small positive-definite systems
        Independent binary IRLS       Ignores cross-class curvature coupling         Not equivalent to joint Softmax IRLS
        IRLS + whitening + QR         Joint curvature, rank diagnostics, no inverse  Dense reference and collinear data
        ============================  ============================================  ==========================================

        For n samples, d design columns and q free categories, probability and gradient work is about O(n*d*q), batched Cholesky is about O(n*q^3), the whitened design storage is about O(n*d*q^2), and dense QR has a leading cost about O(n*d^2*q^3) when n is large relative to d*q. These costs make the component unsuitable for arbitrarily large n*d*K workloads

    Algorithm flow
    --------------
    .. code-block:: text

        Input A(n×p), b(n,), explicit K and optimization controls
            |
            v
        Validate finite features and integer labels 0,...,K-1
            |
            v
        Training-only mean / scale -> standardize -> append intercept
            |
            v
        Initialize theta = 0 -> all initial class probabilities = 1/K
            |
            +------------> Optionally compare analytic gradient with jax.grad once
            |
            v
        Reference-category logits -> stable logsumexp unaveraged NLL + L2
            |
            v
        Softmax -> protected p -> S_i = diag(p_i)-p_i*p_i.T+jitter*I
            |
            v
        Batched Cholesky -> A_i = L_i.T@(phi_i.T kron I_q)
                         -> c_i = L_i.T@eta_i + triangular_solve(L_i,y_i-p_i)
            |
            v
        Stack sample blocks -> append sqrt(l2) penalty rows and zero responses
            |
            v
        Handwritten column-pivoted economic Householder QR
            |
            v
        Numerical-rank truncation -> masked back substitution -> restore pivots
            |
            v
        Candidate direction -> check g.T@delta < 0 -> fallback to -g if needed
            |
            v
        Armijo backtracking -> accept only finite sufficient-decrease updates
            |
            +------------> Record loss, accepted step, rank and stop reason
            |
            v
        Inference: reuse mean / scale -> append intercept -> Softmax
            |
            +------------> predict_proba: K columns; reasoning: row-wise argmax

    Step 1 (input and model structure): require a non-empty finite A(n,p), finite integer labels b(n,) and explicit n_classes=K>=2. Labels are checked before conversion to int32 so fractional values cannot be silently truncated. Explicit K preserves classes that happen to be absent from a small training sample

    Step 2 (training-only preprocessing): compute $mu_j=(1/n)*sum_iX_ij$ and $s_j=std(X_:,j)$ (μⱼ = (1/n)ΣᵢXᵢⱼ, sⱼ = std(X:,j)). Replace near-zero scales by 1, standardize and append the intercept. Test data never participates in estimating these statistics

    Step 3 (initialization and penalty): set theta=0, giving initial probability 1/K for every class and initial data loss n*log(K). Construct the row-major mask so all feature parameters are penalized and all free-category intercepts are excluded

    Step 4 (independent gradient check): when check_gradient=True, compare the handwritten matrix gradient with jax.grad at theta=0 before iterating. A mismatch above 1e-8 raises RuntimeError. Automatic differentiation is an independent checker, not the training direction or optimizer. When disabled, gradient_check_error is None

    Step 5 (stable objective): concatenate free-category logits with the zero reference logit and evaluate logsumexp(Z_i)-Z_i,y_i. Probability clipping used later for weights never replaces the original objective logits

    Step 6 (joint IRLS weights): compute protected probabilities, the full q×q categorical covariance for every sample and diagonal jitter. Retaining off-diagonal terms is essential for the joint multinomial Newton system

    Step 7 (batched whitening): use jnp.linalg.cholesky and jax.vmap to construct every A_i and c_i. The inverse-looking response term is computed by solve_triangular, and no per-sample Python loop is required

    Step 8 (regularization augmentation): stack sample blocks and append sqrt(l2)*diag(penalty) with zero responses. This scale matches the unaveraged objective, and the zero mask rows leave intercepts unpenalized

    Step 9 (QR candidate): run handwritten economic column-pivoted Householder QR, estimate rank from diag(R), perform masked back substitution and restore original column order. The rank history contains one entry per accepted update, not a fabricated initial full-rank value

    Step 10 (direction and line search): form candidate-theta, check the directional derivative, fall back to the analytic negative gradient when necessary and try the full step before halving. The Armijo condition protects each accepted update

    Step 11 (update and stopping): append accepted loss, rank and step values. Stop with converged=True when $J_old-J_new<=tolerance*max(1,abs(J_new))$ (J_old−J_new ≤ tolerance·max(1,|J_new|)). n_iter counts accepted updates and equals len(loss_history)-1. Line-search failure and max_iter have distinct non-converged stop states

    Step 12 (probability and class inference): reuse mean and scale, append the intercept and compute reference-category Softmax. predict_proba returns columns in label order 0,...,K-1 with the reference category last; reasoning returns int32 argmax labels. The reference category is a normal class, not an 'other' or rejection category

    JAX unique advantages
    ---------------------
    (1) Explicit float64 and matrix computation: repeated Cholesky, Householder reflections, small diagonal values and rank decisions benefit from float64. The module enables jax_enable_x64 before component arrays are created, and public training converts features and labels to consistent dtypes

    (2) Selective JIT compilation: fixed-shape objective, analytic gradient, IRLS candidate construction, handwritten QR and probability inference are compiled through JAX. n_classes is static for inference. The Python training loop deliberately retains host-side direction, Armijo and convergence decisions; it is not an end-to-end device-resident while loop

    (3) vmap for per-sample matrix algebra: the mathematical derivation is written for one L_i, phi_i, eta_i and residual, then lifted to all samples with jax.vmap. This avoids hand-written three-dimensional indexing and Python sample loops while preserving full class coupling

    (4) Automatic differentiation as an independent checker: training uses the analytic matrix gradient and QR candidate rather than gradient descent. jax.grad checks the same scalar objective at initialization, helping detect row/column unfolding, class-slice and penalty-mask inconsistencies:

    .. code-block:: python

        import jax
        from statjax.regression._components import _multinomial_objective,_multinomial_matrix_gradient

        manual = _multinomial_matrix_gradient(theta,Phi,Y,penalty,l2,n_classes)
        automatic = jax.grad(_multinomial_objective)(theta,Phi,Y,penalty,l2,n_classes)
        relative_error = jax.numpy.linalg.norm(manual-automatic) / (1.0+jax.numpy.linalg.norm(automatic))

    (5) Immutable functional updates: column exchanges, reflector storage, Q construction, masked back substitution and pivot restoration use .at[...].set(...). These operations preserve JAX array semantics and make data dependencies visible inside compiled functions

    (6) Reproducible independent tests: explicit PRNGKey streams generate features, true parameters, categorical labels and data splits. Tests compare against automatic differentiation, an explicitly assembled small Newton system, QR identities and probability constraints rather than relying only on classification accuracy

    (7) Backend migration with explicit boundaries: matrix operations can run on supported JAX backends, but Cholesky, handwritten QR, compilation and determinism must be assessed per backend. CPU is the reference environment. float(...) and int(...) conversions synchronize with the host; throughput benchmarks must use block_until_ready()

    Precautions
    -----------
    (1) Dense scale limitation: each iteration materializes whitened sample blocks and QR work arrays. Large n, p or K can rapidly increase compilation time, memory and compute. L-BFGS, mini-batch methods or Hessian-vector iterative solvers may be more appropriate for large sparse workloads

    (2) Do not split into independent binary models: S_i contains non-diagonal covariance terms. Independent one-vs-rest binary IRLS changes both the objective and probability normalization and is not the joint multinomial Newton method implemented here

    (3) Keep an identifiable parameterization: unconstrained full-K Softmax has a common-shift null space. This component uses a reference category; a sum-to-zero or other constraint would require consistent changes to logits, gradients and IRLS matrices

    (4) Preserve regularization scaling: the unaveraged NLL requires sqrt(l2) augmented rows. Switching to mean NLL requires corresponding changes to the objective, gradient and IRLS system rather than changing only one expression

    (5) Preserve row-major ordering: theta.reshape((d,q)) and phi_i.T kron I_q are paired conventions. Changing one without the other invalidates the Newton system even if probability predictions still appear plausible

    (6) Complete separation is distinct from collinearity: QR rank handling and positive L2 can address numerical rank issues, but an unregularized completely separated model may have no finite maximum-likelihood estimate. Finite arrays alone do not prove that a finite statistical optimum exists

    (7) rank_tol is a numerical choice: too small a value can retain noise directions, while too large a value discards informative directions. Rank-truncated QR returns a basic solution and is not principal-component regression, SVD truncation or a minimum-norm solver

    (8) Probability floor and jitter are numerical protection: probability_floor and weight_jitter affect only IRLS weights. Large values can distort curvature and should prompt review of scaling, separation, regularization and step control rather than being treated as modeling preferences

    (9) Reuse training preprocessing: validation, testing and deployment must use the stored mean and scale in the original order standardize -> append intercept -> matrix multiply -> reference Softmax. Do not standardize all data before splitting

    (10) Use float64 for the reference workflow: rank thresholds, Cholesky and numerical checks are calibrated for float64. User-created auxiliary arrays should also use consistent dtypes

    (11) Static shapes affect compilation: the handwritten Householder column loop is statically unrolled. Changes in sample, feature or class dimensions can trigger recompilation, and large parameter dimensions can increase compile time

    (12) Host synchronization is intentional: Python loss, direction and step checks are clear reference-implementation boundaries, not a maximum-throughput design. GPU use does not automatically imply faster training

    (13) Interpret convergence states precisely: a small accepted loss improvement declares convergence under the configured criterion. max_iter and line_search_failed return converged=False and preserve the last valid parameters; inspect loss, rank and step histories before accepting a model

    (14) Accuracy is not a sufficient correctness test: retain gradient, explicit Newton-system, QR reconstruction, orthogonality, rank, monotone-loss and probability-row checks. Accuracy also depends on data generation and task difficulty

    (15) Production model management is outside scope: string-label mapping, serialization, outlier handling, hyperparameter selection, calibration and genuine cross-validation require additional engineering. The integer class order and reference category must remain stable across training and deployment

    :example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from statjax.regression import QRMultinomialLogisticComponent
        >>> # Generate one reproducible population before splitting train and test data
        >>> keys = jax.random.split(jax.random.PRNGKey(42),5)
        >>> A = jax.random.normal(keys[0],(400,4),dtype=jnp.float64)
        >>> true_w = 2.0 * jax.random.normal(keys[1],(4,3),dtype=jnp.float64) / 2.0
        >>> true_b = 0.35 * jax.random.normal(keys[2],(3,),dtype=jnp.float64)
        >>> b = jax.random.categorical(keys[3],A @ true_w + true_b,axis=1).astype(jnp.int32)
        >>> permutation = jax.random.permutation(keys[4],400)
        >>> test_idx,train_idx = permutation[:100],permutation[100:]
        >>> A_train,A_test = A[train_idx],A[test_idx]
        >>> b_train,b_test = b[train_idx],b[test_idx]
        >>> # Train jointly coupled reference-category multinomial Logistic regression
        >>> multinomial_instance = QRMultinomialLogisticComponent(n_classes=3,l2=1e-4)
        >>> theta_hat,metrics = multinomial_instance.train(A=A_train,b=b_train)
        >>> probabilities = multinomial_instance.predict_proba(A_test=A_test)
        >>> y_pred = multinomial_instance.reasoning(A_test=A_test)
        >>> accuracy = jnp.mean(y_pred == b_test)

    :reference:
        - [Multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)
        - [Iteratively reweighted least squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)
        - [Householder transformation](https://en.wikipedia.org/wiki/Householder_transformation)
        - [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition)
        - [Backtracking line search](https://en.wikipedia.org/wiki/Backtracking_line_search)
        - StatJAX006 Multinomial Logistic Design Document `"StatJAX006 Multinomial Logistic SH V001"<https://github.com/redblue0216/statjax>`_
    '''


    def __init__(self,n_classes,l2=1e-4,max_iter=50,tolerance=1e-9,rank_tol=1e-12,probability_floor=1e-12,weight_jitter=1e-10,armijo_c1=1e-4,max_backtracks=30,check_gradient=True):
        '''Attribute Function:

            Define initialization for multinomial model structure, optimization controls and trained state

        :parameters:
            - n_classes (int) - Total number of classes K, at least 2
            - l2 (float) - Non-negative L2 regularization coefficient; intercepts are not penalized
            - max_iter (int) - Maximum number of IRLS iterations
            - tolerance (float) - Positive relative loss-improvement tolerance
            - rank_tol (float) - Positive relative numerical-rank threshold
            - probability_floor (float) - IRLS probability floor in (0,1)
            - weight_jitter (float) - Positive diagonal damping for IRLS weights
            - armijo_c1 (float) - Armijo coefficient in (0,1)
            - max_backtracks (int) - Maximum number of Armijo backtracking attempts
            - check_gradient (bool) - Whether to independently check the initial analytic gradient
        '''

        if not isinstance(n_classes,int) or isinstance(n_classes,bool) or n_classes < 2:
            raise ValueError("n_classes must be an integer greater than or equal to 2")
        if not 0.0 <= l2 < float("inf"):
            raise ValueError("l2 must be finite and non-negative")
        if not isinstance(max_iter,int) or isinstance(max_iter,bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not 0.0 < tolerance < float("inf"):
            raise ValueError("tolerance must be finite and positive")
        if not 0.0 < rank_tol < float("inf"):
            raise ValueError("rank_tol must be finite and positive")
        if not 0.0 < probability_floor < 1.0:
            raise ValueError("probability_floor must be in (0,1)")
        if not 0.0 < weight_jitter < float("inf"):
            raise ValueError("weight_jitter must be finite and positive")
        if not 0.0 < armijo_c1 < 1.0:
            raise ValueError("armijo_c1 must be in (0,1)")
        if not isinstance(max_backtracks,int) or isinstance(max_backtracks,bool) or max_backtracks < 1:
            raise ValueError("max_backtracks must be a positive integer")
        if not isinstance(check_gradient,bool):
            raise ValueError("check_gradient must be a boolean")

        self.n_classes = n_classes
        self.l2 = l2
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.rank_tol = rank_tol
        self.probability_floor = probability_floor
        self.weight_jitter = weight_jitter
        self.armijo_c1 = armijo_c1
        self.max_backtracks = max_backtracks
        self.check_gradient = check_gradient
        self.theta_hat = None
        self.mean = None
        self.scale = None
        self.metrics = None


    def train(self,A,b):
        '''Method Function:

            Define multinomial training with automatic training-only standardization and intercept augmentation

        :parameters:
            - A (jnp.ndarray) - (n,p) Training feature matrix without a manually appended intercept
            - b (jnp.ndarray) - (n,) Integer class labels in 0,...,K-1

        :return:
            - theta_hat (jnp.ndarray) - ((p+1)*(K-1),) Row-major flattened trained parameters
            - metrics (dict) - Loss, rank, step, gradient-check and convergence diagnostics
        '''

        A = jnp.asarray(A,dtype=jnp.float64)
        b_values = jnp.asarray(b,dtype=jnp.float64)
        if A.ndim != 2 or b_values.ndim != 1 or A.shape[0] != b_values.shape[0]:
            raise ValueError("A must have shape (n,p), b must have shape (n,), and their sample counts must match")
        if A.shape[0] == 0 or A.shape[1] == 0:
            raise ValueError("A and b must be non-empty and A must contain at least one feature")
        if not bool(jnp.all(jnp.isfinite(A))) or not bool(jnp.all(jnp.isfinite(b_values))):
            raise ValueError("A and b must contain only finite values")
        if not bool(jnp.all(b_values == jnp.trunc(b_values))):
            raise ValueError("b must contain integer class labels")
        if not bool(jnp.all((b_values >= 0) & (b_values < self.n_classes))):
            raise ValueError("b must contain class labels in the range 0,...,n_classes-1")
        b = b_values.astype(jnp.int32)

        result = _fit_qr_multinomial_logistic(
            A,b,self.n_classes,self.l2,self.max_iter,self.tolerance,self.rank_tol,
            self.probability_floor,self.weight_jitter,self.armijo_c1,
            self.max_backtracks,self.check_gradient
        )
        self.theta_hat = result["theta_hat"]
        self.mean = result["mean"]
        self.scale = result["scale"]
        self.metrics = {
            "initial_loss": result["loss_history"][0],
            "final_loss": result["loss_history"][-1],
            "loss_history": result["loss_history"],
            "rank_history": result["rank_history"],
            "step_history": result["step_history"],
            "gradient_check_error": result["gradient_check_error"],
            "converged": result["converged"],
            "stop_reason": result["stop_reason"],
            "n_iter": result["n_iter"],
            "l2": self.l2
        }

        return self.theta_hat,self.metrics


    def predict_proba(self,A_test):
        '''Method Function:

            Define K-class probability inference using the trained parameters and training preprocessing

        :parameters:
            - A_test (jnp.ndarray) - (m,p) Test feature matrix with the training feature dimension

        :return:
            - probabilities (jnp.ndarray) - (m,K) Probabilities in class-label order
        '''

        if self.theta_hat is None:
            raise RuntimeError("QRMultinomialLogisticComponent has not been trained yet, please call the train method first")
        A_test = jnp.asarray(A_test,dtype=jnp.float64)
        if A_test.ndim != 2 or A_test.shape[1] != self.mean.shape[0]:
            raise ValueError("A_test must have shape (m,p) with the same feature dimension as training data")
        if not bool(jnp.all(jnp.isfinite(A_test))):
            raise ValueError("A_test must contain only finite values")

        return _predict_multinomial_logistic_proba(
            A_test,self.theta_hat,self.mean,self.scale,self.n_classes
        )


    def reasoning(self,A_test):
        '''Method Function:

            Define multiclass inference by row-wise argmax over class probabilities

        :parameters:
            - A_test (jnp.ndarray) - (m,p) Test feature matrix

        :return:
            - y_pred (jnp.ndarray) - (m,) Predicted int32 class labels
        '''

        probabilities = self.predict_proba(A_test)
        y_pred = jnp.argmax(probabilities,axis=1).astype(jnp.int32)

        return y_pred


    def _info(self):
        '''Method Function:

            Define an internal method to obtain basic meta information

        :parameters:
            nothing

        :return:
            - info (str) - Text introduction of the component
        '''

        return "QRMultinomialLogisticComponent: multinomial Logistic regression component based on reference-category Softmax, IRLS, per-sample whitening and column-pivoted Householder QR decomposition, implemented with the JAX technology stack"



##############################################################################################################################################################################
##############################################################################################################################################################################


_MNP_DIAGONAL_RAW_CLIP = 20.0
_MNP_CDF_FLOOR = 1e-14
_MNP_UNIFORM_FLOOR = 1e-12


def _num_mnp_cov_params(d):
    '''Method Function:

        Return the number of free Cholesky parameters after fixing Sigma[0,0]=1

    :parameters:
        - d (int) - Dimension of utility differences, D=J-1

    :return:
        - count (int) - Number of free covariance parameters
    '''

    return d * (d + 1) // 2 - 1



def _initial_mnp_params(d,p):
    '''Method Function:

        Initialize beta=0 and L=I for independent standard normal utility differences

    :parameters:
        - d (int) - Utility-difference dimension
        - p (int) - Number of shared utility coefficients

    :return:
        - theta (jnp.ndarray) - Initial unconstrained parameter vector
    '''

    return jnp.zeros(p + _num_mnp_cov_params(d),dtype=jnp.float64)



def _unpack_mnp_params(theta,d,p):
    '''Method Function:

        Decode an unconstrained vector into beta and a scale-constrained lower Cholesky factor

    :parameters:
        - theta (jnp.ndarray) - Unconstrained optimization parameters
        - d (int) - Utility-difference dimension
        - p (int) - Number of shared utility coefficients

    :return:
        - beta (jnp.ndarray) - Shared utility coefficients
        - L (jnp.ndarray) - Lower triangular factor with L[0,0]=1
    '''

    beta = theta[:p]
    raw = theta[p:]
    rows = []
    k = 0
    zero = jnp.array(0.0,dtype=theta.dtype)
    one = jnp.array(1.0,dtype=theta.dtype)
    for r in range(d):
        row = []
        for c in range(d):
            if c > r:
                row.append(zero)
            elif r == 0 and c == 0:
                row.append(one)
            elif r == c:
                row.append(jnp.exp(jnp.clip(raw[k],-_MNP_DIAGONAL_RAW_CLIP,_MNP_DIAGONAL_RAW_CLIP)))
                k += 1
            else:
                row.append(raw[k])
                k += 1
        rows.append(jnp.stack(row))

    return beta,jnp.stack(rows)



@lru_cache(maxsize=None)
def _mnp_choice_contrasts(n_classes):
    '''Method Function:

        Construct C[y] such that class y is optimal exactly when C[y]@eta<0; the last class is the reference

    :parameters:
        - n_classes (int) - Total number of classes J

    :return:
        - contrasts (np.ndarray) - (J,D,D) Cached class-specific contrast matrices
    '''

    d = n_classes - 1
    matrices = []
    for y in range(n_classes):
        C = np.zeros((d,d),dtype=np.float64)
        for r,k in enumerate(k for k in range(n_classes) if k != y):
            if k < d:
                C[r,k] = 1.0
            if y < d:
                C[r,y] -= 1.0
        matrices.append(C)

    return np.stack(matrices)



def _ghk_mnp_log_probs(theta,A,uniforms):
    '''Method Function:

        Evaluate all class log probabilities with the GHK sequential truncated-normal simulator

    :parameters:
        - theta (jnp.ndarray) - Unconstrained utility and covariance parameters
        - A (jnp.ndarray) - (n,D,p) Alternative-related design tensor
        - uniforms (jnp.ndarray) - (n,R,D) Fixed or independently drawn uniform random numbers

    :return:
        - log_probabilities (jnp.ndarray) - (n,J) Simulated log probabilities in class-label order
    '''

    n,d,p = A.shape
    if uniforms.ndim != 3 or uniforms.shape[0] != n or uniforms.shape[2] != d or uniforms.shape[1] < 1:
        raise ValueError("uniforms must have shape (n,R,D) with R >= 1 and dimensions matching A")
    R = uniforms.shape[1]
    beta,L_cov = _unpack_mnp_params(theta,d,p)
    Sigma = L_cov @ L_cov.T
    mu = jnp.einsum("ndp,p->nd",A,beta)
    contrasts = jnp.asarray(_mnp_choice_contrasts(d + 1),dtype=A.dtype)

    all_log_probs = []
    for y in range(d + 1):
        C = contrasts[y]
        nu = mu @ C.T
        S = C @ Sigma @ C.T
        Gamma = jnp.linalg.cholesky(S)
        log_weight = jnp.zeros((n,R),dtype=A.dtype)
        z_previous = []

        ### Sequential GHK conditioning: z_r is drawn from (-infinity,b_r)
        for r in range(d):
            if r == 0:
                past = jnp.zeros((n,R),dtype=A.dtype)
            else:
                past = sum(Gamma[r,s] * z_previous[s] for s in range(r))
            b = -(nu[:,None,r] + past) / Gamma[r,r]
            log_weight = log_weight + log_ndtr(b)
            cdf = jnp.clip(ndtr(b),_MNP_CDF_FLOOR,1.0)
            u = jnp.clip(uniforms[:,:,r],_MNP_UNIFORM_FLOOR,1.0 - _MNP_UNIFORM_FLOOR)
            z_previous.append(ndtri(u * cdf))

        all_log_probs.append(logsumexp(log_weight,axis=1) - jnp.log(float(R)))

    return jnp.stack(all_log_probs,axis=1)



def _mnp_simulated_nll(theta,A,b,uniforms):
    '''Method Function:

        Define mean negative simulated log-likelihood conditional on fixed common random numbers

    :parameters:
        - theta (jnp.ndarray) - Unconstrained model parameters
        - A (jnp.ndarray) - Training design tensor
        - b (jnp.ndarray) - Integer observed classes
        - uniforms (jnp.ndarray) - Fixed training random numbers

    :return:
        - loss (jnp.ndarray) - Scalar mean negative simulated log-likelihood
    '''

    log_probabilities = _ghk_mnp_log_probs(theta,A,uniforms)

    return -jnp.mean(log_probabilities[jnp.arange(b.shape[0]),b])



def _fit_ghk_multinomial_probit(A,b,num_draws,max_iter,grad_tol,seed):
    '''Method Function:

        Fit multinomial Probit by fixed-draw simulated maximum likelihood and JAX BFGS with automatic differentiation

    :parameters:
        - A (jnp.ndarray) - Validated training design tensor
        - b (jnp.ndarray) - Validated integer class labels
        - num_draws (int) - Number of fixed GHK paths per observation
        - max_iter (int) - Maximum BFGS iterations
        - grad_tol (float) - BFGS gradient tolerance
        - seed (int) - Training random seed

    :return:
        - result (dict) - Parameters, interpretable covariance state and optimization diagnostics
    '''

    n,d,p = A.shape
    uniforms = jax.random.uniform(
        jax.random.PRNGKey(seed),(n,num_draws,d),dtype=jnp.float64
    )
    theta0 = _initial_mnp_params(d,p)
    objective = jax.jit(lambda theta: _mnp_simulated_nll(theta,A,b,uniforms))
    initial_nll = float(objective(theta0))
    optimization = minimize(
        objective,theta0,method="BFGS",options={"maxiter":max_iter,"gtol":grad_tol}
    )

    theta_hat = optimization.x
    beta_hat,l_hat = _unpack_mnp_params(theta_hat,d,p)
    sigma_hat = l_hat @ l_hat.T
    final_nll = float(optimization.fun)
    grad_norm = float(jnp.linalg.norm(optimization.jac))
    if not bool(jnp.all(jnp.isfinite(theta_hat))) or not np.isfinite(final_nll) or not np.isfinite(grad_norm):
        raise RuntimeError("Multinomial Probit optimization produced non-finite parameters or diagnostics")

    return {
        "theta_hat": theta_hat,
        "beta_hat": beta_hat,
        "l_hat": l_hat,
        "sigma_hat": sigma_hat,
        "initial_nll": initial_nll,
        "final_nll": final_nll,
        "converged": bool(optimization.success),
        "status": int(optimization.status),
        "n_iter": int(optimization.nit),
        "grad_norm": grad_norm
    }



@jax.jit(static_argnums=(2,3,4))
def _predict_ghk_mnp_proba(A_test,theta_hat,num_draws,seed,normalize):
    '''Method Function:

        Simulate independent inference draws, optionally normalizing finite-draw GHK probabilities by row

    :parameters:
        - A_test (jnp.ndarray) - Test design tensor
        - theta_hat (jnp.ndarray) - Trained unconstrained parameters
        - num_draws (int) - Static number of inference paths
        - seed (int) - Static independent inference seed
        - normalize (bool) - Whether to normalize each probability row

    :return:
        - probabilities (jnp.ndarray) - (m,J) Simulated class probabilities
    '''

    n,d,_ = A_test.shape
    uniforms = jax.random.uniform(
        jax.random.PRNGKey(seed),(n,num_draws,d),dtype=jnp.float64
    )
    log_probabilities = _ghk_mnp_log_probs(theta_hat,A_test,uniforms)
    if normalize:
        probabilities = jnp.exp(
            log_probabilities - logsumexp(log_probabilities,axis=1,keepdims=True)
        )
    else:
        probabilities = jnp.exp(log_probabilities)

    return probabilities



class GHKMultinomialProbitComponent(BaseRegression):
    '''Class Introduction:

        This is a multinomial Probit component using scale-constrained Cholesky covariance parameters, GHK choice-probability simulation, fixed-draw simulated maximum likelihood and BFGS with JAX automatic differentiation. It models joint choices through correlated normal latent utilities rather than independent binary classifiers. The final class is the reference category and Sigma[0,0]=1 identifies the latent utility scale. It is intended for small and medium dense data, teaching, algorithm verification and numerically transparent reference workflows

    Mathematical principle
    ----------------------
    (1) Latent utility and reference-category identification: for observation i and J mutually exclusive alternatives, the complete utility model is:

        $U_ij=v_ij+epsilon^U_ij , y_i=argmax_j U_ij$
        Uᵢⱼ = vᵢⱼ + εᵁᵢⱼ, yᵢ = argmaxⱼUᵢⱼ

        Only relative utility is identified. The final class J-1 is therefore assigned zero reference utility. With D=J-1, the utility-difference model is:

        $eta_i=X_i*beta+epsilon_i , epsilon_i~N_D(0,Sigma)$
        ηᵢ = Xᵢβ + εᵢ, εᵢ ~ N_D(0,Σ)

        $X_i in R^(D*p)$ (Xᵢ ∈ R^(D×p)) contains alternative-related covariates, $beta in R^p$ (β ∈ R^p) is shared by the non-reference rows, and the complete utility vector is $(eta_i^T,0)^T$ ((ηᵢᵀ,0)ᵀ). The caller must preserve alternative order, reference coding, feature order and any training-derived preprocessing during inference

        Multiplying every utility by a positive constant leaves argmax unchanged, so the covariance scale is not identified. This component imposes $Sigma_00=1$ (Σ₀₀ = 1). The reference-utility constraint and scale constraint address different identification problems and both are required

    (2) Scale-constrained Cholesky parameterization: instead of optimizing an unrestricted covariance matrix, the component writes:

        $Sigma=L*L^T , L_00=1 , L_rr=exp(a_r)>0 for r>=1$
        Σ = LLᵀ, L₀₀ = 1, Lᵣᵣ = exp(aᵣ) > 0 for r ≥ 1

        The remaining lower-triangular entries are unrestricted. For D=3:

        $L=[[1,0,0],[l_10,exp(a_1),0],[l_20,l_21,exp(a_2)]]$
        L = [[1,0,0],[ℓ₁₀,eᵃ¹,0],[ℓ₂₀,ℓ₂₁,eᵃ²]]

        This construction makes Sigma symmetric positive definite and automatically satisfies Sigma[0,0]=1. The free covariance-parameter count is $q=D*(D+1)/2-1$ (q = D(D+1)/2−1), and the complete unconstrained vector is theta=(beta,vartheta). Exponential inputs are clipped to [-20,20] only as overflow protection during trial optimization steps; persistent solutions at that boundary indicate a model, data or optimization problem

    (3) Choice events as correlated normal negative orthants: let $B=[I_D;0^T]$ (B = [I_D;0ᵀ]) so U_i=B eta_i. For each candidate class y, construct A_y with rows e_k-e_y for every k!=y and define:

        $C_y=A_y*B in R^(D*D) , V_i^(y)=C_y*eta_i$
        C_y = A_yB ∈ R^(D×D), Vᵢ⁽ʸ⁾ = C_yηᵢ

        Every coordinate of V_i^(y) is another alternative's utility minus the utility of y. Therefore:

        $y_i=y if and only if C_y*eta_i<0$
        yᵢ = y ⇔ C_yηᵢ < 0 coordinate-wise

        $V_i^(y)~N_D(nu_y,S_y) , nu_y=C_y*mu_i , S_y=C_y*Sigma*C_y^T , mu_i=X_i*beta$
        Vᵢ⁽ʸ⁾ ~ N_D(ν_y,S_y), ν_y = C_yμᵢ, S_y = C_yΣC_yᵀ, μᵢ = Xᵢβ

        The cached _mnp_choice_contrasts helper constructs all C_y matrices. For D=2 and reference class 2, C_2=I, C_0 encodes eta_1-eta_0<0 and -eta_0<0, and C_1 encodes eta_0-eta_1<0 and -eta_1<0. All class probabilities can then use one GHK kernel

    (4) GHK recursive conditional probabilities: for each class y, factor the transformed covariance as:

        $S_y=Gamma*Gamma^T , V_i^(y)=nu_y+Gamma*z , z~N_D(0,I)$
        S_y = ΓΓᵀ, Vᵢ⁽ʸ⁾ = ν_y + Γz, z ~ N_D(0,I)

        Gamma is distinct from L: L parameterizes the original utility-difference covariance Sigma, whereas Gamma factorizes the class-transformed S_y for simulation. Requiring V_r<0 sequentially gives the conditional upper bound:

        $b_r=-(nu_r+sum_(s<r)(Gamma_rs*z_s))/Gamma_rr$
        bᵣ = −(νᵣ + Σₛ<ᵣΓᵣₛzₛ)/Γᵣᵣ

        The r-th conditional probability is Phi(b_r), and a truncated standard normal coordinate is generated by inverse transformation:

        $z_r=Phi^(-1)(u_r*Phi(b_r)) , u_r~Uniform(0,1)$
        zᵣ = Φ⁻¹(uᵣΦ(bᵣ)), uᵣ ~ Uniform(0,1)

        One path has weight $w=product_r Phi(b_r)$ (w = ΠᵣΦ(bᵣ)). R paths produce:

        $P_hat_y=(1/R)*sum_m w^(m)$
        P̂_y = (1/R)Σₘw⁽ᵐ⁾

        $log(P_hat_y)=logsumexp_m(log(w^(m)))-log(R)$
        log P̂_y = logsumexpₘ(log w⁽ᵐ⁾) − log R

        log_ndtr computes log Phi directly in the tails, and logsumexp avoids underflow from explicit products and averages. ndtr, ndtri and uniform clipping protect inverse sampling; these numerical safeguards do not make extreme-tail simulation exact

    (5) Fixed-draw simulated maximum likelihood: after generating uniforms U with shape (n,R,D), the component minimizes:

        $L_S(theta)=-(1/n)*sum_i log(P_hat_(i,y_i)(theta;U))$
        𝓛_S(θ) = −(1/n)Σᵢlog P̂ᵢ,ᵧᵢ(θ;U)

        U remains fixed throughout optimization, making the objective deterministic, reproducible and differentiable as a function of theta. Training does not renormalize each row of simulated class probabilities, because the objective must remain the corresponding raw GHK simulated likelihood

        Although the GHK probability estimator is unbiased over random draws, finite-R simulation generally gives $E(log(P_hat))!=log(E(P_hat))=log(P)$ (E[log P̂] ≠ log E[P̂] = log P). Fixed common random numbers stabilize optimization; they do not eliminate simulation error or finite-draw log-likelihood bias

    (6) Automatic differentiation and BFGS: JAX differentiates through parameter unpacking, matrix products, both Cholesky levels, normal CDF/inverse CDF and logsumexp. BFGS uses these gradients and an inverse-Hessian approximation; no handwritten MNP gradient or explicit true Hessian is required. The main work is approximately $O(J*n*R*D^2)$ (O(JnRD²)), with small-matrix factorization and static-loop compilation overhead. The stored uniforms require $O(n*R*D)$ (O(nRD)) memory

        The method is chosen for correlated-error fidelity and transparent simulation, not universal speed:

        ============================  ============================================  ==========================================
        Method                        Main property                                 Typical scenario
        ============================  ============================================  ==========================================
        Cholesky + GHK + MSL + BFGS   Correlated normal errors; automatic gradients  This reference implementation
        Direct numerical integration  Deterministic and accurate in low dimensions  Very small D only
        EM + GLS/QR                   Clear matrix-regression M step                 E step still needs truncated moments
        Albert-Chib + Gibbs           Full Bayesian posterior and latent utilities   Bayesian rather than frequentist MSL
        Binary Probit IRLS            Scalar normal-CDF likelihood and weights       Not the joint MNP regional integral
        Mixed Logit + MSL             Flexible simulated choice model               Different error distribution and model
        ============================  ============================================  ==========================================

    Algorithm flow
    --------------
    .. code-block:: text

        Input A(n,D,p), integer b(n,), simulation and BFGS controls
            |
            v
        Validate finite dimensions and labels 0,...,D; final class = reference
            |
            v
        Initialize beta=0 and L=I; Sigma[0,0]=1 by construction
            |
            v
        Generate fixed uniforms U(n,R,D) once for the whole training run
            |
            v
        theta -> beta,L -> Sigma=L@L.T -> mu=A@beta
            |
            v
        For every class y: nu=C[y]@mu and S=C[y]@Sigma@C[y].T
            |
            v
        Gamma=Cholesky(S) -> sequential b_r and truncated-normal z_r
            |
            v
        Accumulate log Phi(b_r) -> logsumexp over R paths -> log P_hat(n,J)
            |
            v
        Select observed-class log P_hat -> mean negative simulated likelihood
            |
            v
        JAX automatic differentiation -> BFGS update -> convergence diagnostics
            |
            +------------> Save theta, beta, L, Sigma, NLL, status and gradient norm
            |
            v
        Inference: independent uniforms -> same GHK kernel -> optional row normalization
            |
            +------------> predict_proba: (m,J); reasoning: int32 row-wise argmax

    Step 1 (input and encoding): require non-empty finite A(n,D,p) and finite integer labels b(n,), with matching samples and labels in 0,...,D. D=J-1 determines the number of classes, so no redundant n_classes argument is needed. Labels are checked before conversion to int32

    Step 2 (identification and initialization): use the final class as zero reference, fix L[0,0]=1 and start at beta=0,L=I. The initial covariance is independent and standard normal in the utility-difference space

    Step 3 (common random numbers): generate the complete training uniform array from seed before BFGS. Every objective and gradient evaluation uses this same array, preventing the optimizer from chasing fresh simulation noise

    Step 4 (parameter decoding): unpack theta into beta and lower-triangular L. Positive diagonal exponentials and fixed L[0,0] guarantee the required covariance structure; static Python loops depend only on D and p

    Step 5 (class transformations): compute mu and every nu_y,S_y. The choice contrasts map each discrete selection event to a correlated normal negative-orthant probability

    Step 6 (GHK simulation): Cholesky-factor each S_y, recursively evaluate conditional upper bounds, draw truncated normal coordinates and accumulate log conditional probabilities. Sample and path dimensions are vectorized; only static class and coordinate loops remain in Python

    Step 7 (MSL objective): average each path's weights in the log domain, select the observed class and form mean negative simulated log-likelihood. Raw probabilities are not row-normalized inside the training objective

    Step 8 (BFGS and diagnostics): JIT-compile the fixed-shape objective closed over A,b,U. JAX supplies gradients to jax.scipy.optimize.minimize. Save actual success/status, iteration count, gradient norm and initial/final NLL; reaching max_iter is not reported as convergence

    Step 9 (trained state): retain theta for simulation and expose beta_hat, l_hat and sigma_hat=l_hat@l_hat.T for interpretation. Non-finite final parameters, objectives or gradients raise an explicit error

    Step 10 (independent inference): generate fresh uniforms from inference_seed, evaluate all J class probabilities and optionally normalize in the log domain. Normalization enforces unit row sums under finite R but does not correct model misspecification or systematic simulation bias

    Step 11 (classification): reasoning applies row-wise argmax to predict_proba and returns int32 zero-based labels. The reference class is an ordinary predicted class, not an 'other' or rejection category

    Step 12 (verification): tests independently check parameter constraints, explicit choice contrasts, low-dimensional normal-CDF references, raw probability completeness, automatic differentiation, fixed-draw determinism, BFGS convergence, inference and error behavior. Accuracy alone is not treated as proof of correctness

    JAX unique advantages
    ---------------------
    (1) End-to-end automatic differentiation of the simulated objective: Cholesky, einsum, ndtr, ndtri, log_ndtr and logsumexp remain in one differentiable computation graph. This avoids handwritten derivatives of regional normal probabilities with respect to both beta and covariance parameters

    (2) Explicit functional random streams: PRNGKey separates data generation, training common random numbers, independent CDF checks and inference. Reusing a key reproduces the same numerical experiment; changing the inference key exposes reasonable simulation variation

    (3) Selective JIT compilation: the training objective is compiled after static A,b,U shapes are closed over, and probability inference uses static draw-count, seed and normalization controls. Class and coordinate loops are statically unrolled. Shape changes can trigger recompilation

    (4) Shared train/inference GHK kernel: both paths call _ghk_mnp_log_probs, reducing the risk of inconsistent choice contrasts or recursion. Different uniforms, rather than different statistical formulas, distinguish optimization from independent evaluation

    (5) Stable matrix and log-domain primitives: Cholesky enforces the covariance factorization, log_ndtr evaluates normal tail log probabilities and logsumexp performs path averaging and normalized inference without explicitly dividing by a potentially zero raw probability sum

    (6) Independent numerical verification: tests use only existing NumPy, standard-library erf and JAX. D=1 has an analytic normal-CDF reference; D=2 uses deterministic one-dimensional quadrature of a conditional normal probability. SciPy is not a runtime or mandatory test dependency

    (7) Backend migration with explicit boundaries: matrix operations can run on supported JAX backends, but small-matrix Cholesky, static compilation, random algorithms and performance must be assessed per backend. CPU + float64 is the reference environment, and GPU execution is not automatically faster for small problems

    Precautions
    -----------
    (1) Identification cannot be omitted: Cholesky guarantees positive definiteness but does not identify location or scale. Keep both the zero reference utility and Sigma[0,0]=1

    (2) Interpret the estimated covariance correctly: Sigma is the covariance of utility differences relative to the final reference, not every element of the original complete-utility error covariance

    (3) Finite R creates simulation error: increase num_draws and inference_num_draws and vary seeds for sensitivity analysis. Fixed random numbers improve optimization stability but do not remove finite-draw bias

    (4) Do not normalize the training target: row normalization during training would change the raw GHK simulated likelihood. Inference normalization is only numerical presentation and does not repair estimation bias

    (5) Tail clipping has limits: CDF and uniform floors prevent exact 0/1 inverse-CDF inputs, while diagonal clipping prevents exponential overflow. Extreme tails can still cause approximation error, vanishing gradients or slow convergence

    (6) Positive definite does not mean well conditioned: Sigma can have a large condition number despite valid Cholesky factors. Inspect covariance eigenvalues, optimization diagnostics and simulation sensitivity for difficult data

    (7) BFGS convergence is local: success and a small gradient do not establish a global optimum. Multiple initializations, alternative covariance starts or Multinomial Logit initialization may be useful in a separate application layer

    (8) Hessian-based standard errors are not provided: the inverse approximate Hessian conditional on fixed draws is not automatically a correct frequentist standard-error estimator. Simulation-aware robust inference requires additional work

    (9) Memory and compile time scale quickly: uniforms use O(nRD) storage and static J/D loops enlarge the compiled graph. Large samples, high class counts or high D require chunking, minibatches or a different estimator

    (10) Preserve design semantics: all non-reference alternatives share beta in this parameterization. Individual-invariant covariates normally require alternative interactions or differences; otherwise they may cancel from utility differences

    (11) Binary reduction is a special case: J=2 gives D=1 and no free covariance parameter. GHK then approximates the ordinary univariate Probit probability, but this component need not match QRProbitComponent's IRLS iterations or floating-point output exactly

    (12) Convergence and accuracy are different checks: a finite objective or plausible accuracy does not prove correct choice contrasts or covariance transforms. Retain independent CDF, probability-completeness, gradient and constraint checks

    (13) Reproducibility is backend-relative: explicit seeds make the reference workflow reproducible, but bitwise identity across JAX versions, random implementations, backends or hardware is not promised

    (14) Production modeling is outside scope: missing-data handling, panel dependence, observation weights, arbitrary parameter restrictions, robust standard errors, calibration, serialization and formal model comparison require additional engineering

    :example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from statjax.regression import GHKMultinomialProbitComponent
        >>> # Generate reproducible three-class data; class 2 has zero reference utility
        >>> key_A,key_error = jax.random.split(jax.random.PRNGKey(42))
        >>> A = jax.random.normal(key_A,(400,2,3),dtype=jnp.float64)
        >>> beta_true = jnp.array([0.8,-0.5,0.6],dtype=jnp.float64)
        >>> L_true = jnp.array([[1.0,0.0],[0.35,0.8]],dtype=jnp.float64)
        >>> z = jax.random.normal(key_error,(400,2),dtype=jnp.float64)
        >>> utilities = jnp.concatenate([
        ...     jnp.einsum("ndp,p->nd",A,beta_true) + z @ L_true.T,
        ...     jnp.zeros((400,1),dtype=jnp.float64)
        ... ],axis=1)
        >>> b = jnp.argmax(utilities,axis=1).astype(jnp.int32)
        >>> # Fixed draws stabilize training; inference uses an independent random stream
        >>> component = GHKMultinomialProbitComponent(num_draws=128,inference_num_draws=512)
        >>> theta_hat,metrics = component.train(A=A,b=b)
        >>> probabilities = component.predict_proba(A_test=A[:10])
        >>> y_pred = component.reasoning(A_test=A[:10])
        >>> beta_hat = component.beta_hat
        >>> Sigma_hat = component.sigma_hat

    :reference:
        - [Probit model](https://en.wikipedia.org/wiki/Probit_model)
        - [Multinomial probit](https://en.wikipedia.org/wiki/Discrete_choice#Multinomial_probit)
        - [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition)
        - [BFGS algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
        - StatJAX007 Multinomial Probit Design Document `"StatJAX007 Multinomial Probit SH V001"<https://github.com/redblue0216/statjax>`_
    '''

    def __init__(self,num_draws=128,max_iter=100,grad_tol=1e-5,seed=123,inference_num_draws=1024,inference_seed=2027,normalize=True):
        '''Attribute Function:

            Initialize GHK simulation, BFGS optimization, independent inference controls and trained state

        :parameters:
            - num_draws (int) - Positive number of fixed training paths per observation
            - max_iter (int) - Positive maximum number of BFGS iterations
            - grad_tol (float) - Finite positive BFGS gradient tolerance
            - seed (int) - Non-negative training random seed
            - inference_num_draws (int) - Positive number of independent inference paths
            - inference_seed (int) - Non-negative inference random seed
            - normalize (bool) - Whether inference probabilities are normalized by row
        '''

        if not isinstance(num_draws,int) or isinstance(num_draws,bool) or num_draws < 1:
            raise ValueError("num_draws must be a positive integer")
        if not isinstance(max_iter,int) or isinstance(max_iter,bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not 0.0 < grad_tol < float("inf"):
            raise ValueError("grad_tol must be finite and positive")
        if not isinstance(seed,int) or isinstance(seed,bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(inference_num_draws,int) or isinstance(inference_num_draws,bool) or inference_num_draws < 1:
            raise ValueError("inference_num_draws must be a positive integer")
        if not isinstance(inference_seed,int) or isinstance(inference_seed,bool) or inference_seed < 0:
            raise ValueError("inference_seed must be a non-negative integer")
        if not isinstance(normalize,bool):
            raise ValueError("normalize must be a boolean")

        self.num_draws = num_draws
        self.max_iter = max_iter
        self.grad_tol = grad_tol
        self.seed = seed
        self.inference_num_draws = inference_num_draws
        self.inference_seed = inference_seed
        self.normalize = normalize
        self.theta_hat = None
        self.beta_hat = None
        self.l_hat = None
        self.sigma_hat = None
        self.metrics = None


    def train(self,A,b):
        '''Method Function:

            Train correlated multinomial Probit parameters through Cholesky, GHK, MSL and BFGS

        :parameters:
            - A (jnp.ndarray) - (n,D,p) Alternative-related design tensor with D=J-1
            - b (jnp.ndarray) - (n,) Integer class labels in 0,...,D

        :return:
            - theta_hat (jnp.ndarray) - Trained unconstrained parameter vector
            - metrics (dict) - Simulated likelihood and BFGS convergence diagnostics
        '''

        A = jnp.asarray(A,dtype=jnp.float64)
        b_values = jnp.asarray(b,dtype=jnp.float64)
        if A.ndim != 3 or b_values.ndim != 1 or A.shape[0] != b_values.shape[0]:
            raise ValueError("A must have shape (n,D,p), b must have shape (n,), and their sample counts must match")
        if A.shape[0] == 0 or A.shape[1] == 0 or A.shape[2] == 0:
            raise ValueError("A and b must be non-empty and A must contain positive D and p dimensions")
        if not bool(jnp.all(jnp.isfinite(A))) or not bool(jnp.all(jnp.isfinite(b_values))):
            raise ValueError("A and b must contain only finite values")
        if not bool(jnp.all(b_values == jnp.trunc(b_values))):
            raise ValueError("b must contain integer class labels")
        d = A.shape[1]
        if not bool(jnp.all((b_values >= 0) & (b_values <= d))):
            raise ValueError("b must contain class labels in the range 0,...,D")
        b = b_values.astype(jnp.int32)

        result = _fit_ghk_multinomial_probit(
            A,b,self.num_draws,self.max_iter,self.grad_tol,self.seed
        )
        self.theta_hat = result["theta_hat"]
        self.beta_hat = result["beta_hat"]
        self.l_hat = result["l_hat"]
        self.sigma_hat = result["sigma_hat"]
        self.metrics = {
            "initial_nll": result["initial_nll"],
            "final_nll": result["final_nll"],
            "converged": result["converged"],
            "status": result["status"],
            "n_iter": result["n_iter"],
            "grad_norm": result["grad_norm"],
            "num_draws": self.num_draws,
            "seed": self.seed
        }

        return self.theta_hat,self.metrics


    def predict_proba(self,A_test):
        '''Method Function:

            Return independently simulated class probabilities using the trained parameters

        :parameters:
            - A_test (jnp.ndarray) - (m,D,p) Test design tensor with training dimensions and encoding

        :return:
            - probabilities (jnp.ndarray) - (m,J) Class probabilities in label order
        '''

        if self.theta_hat is None:
            raise RuntimeError("GHKMultinomialProbitComponent has not been trained yet, please call the train method first")
        A_test = jnp.asarray(A_test,dtype=jnp.float64)
        if A_test.ndim != 3 or A_test.shape[1] != self.l_hat.shape[0] or A_test.shape[2] != self.beta_hat.shape[0]:
            raise ValueError("A_test must have shape (m,D,p) with the same D and p dimensions as training data")
        if A_test.shape[0] == 0:
            raise ValueError("A_test must be non-empty")
        if not bool(jnp.all(jnp.isfinite(A_test))):
            raise ValueError("A_test must contain only finite values")

        probabilities = _predict_ghk_mnp_proba(
            A_test,self.theta_hat,self.inference_num_draws,self.inference_seed,self.normalize
        )
        if not bool(jnp.all(jnp.isfinite(probabilities))):
            raise RuntimeError("GHK inference produced non-finite probabilities")

        return probabilities


    def reasoning(self,A_test):
        '''Method Function:

            Return int32 class predictions by row-wise argmax over simulated probabilities

        :parameters:
            - A_test (jnp.ndarray) - Test design tensor

        :return:
            - y_pred (jnp.ndarray) - (m,) Predicted class labels
        '''

        probabilities = self.predict_proba(A_test)
        y_pred = jnp.argmax(probabilities,axis=1).astype(jnp.int32)

        return y_pred


    def _info(self):
        '''Method Function:

            Return component metadata

        :parameters:
            nothing

        :return:
            - info (str) - Text introduction of the component
        '''

        return "GHKMultinomialProbitComponent: multinomial Probit regression component based on scale-constrained Cholesky covariance, GHK simulation, simulated maximum likelihood and BFGS, implemented with the JAX technology stack"



##############################################################################################################################################################################
##############################################################################################################################################################################



def _svr_rbf_kernel(X,Z,gamma):
    '''Method Function:

        Define an RBF kernel matrix using vectorized squared Euclidean distances:

        $K_ij=exp(-gamma*||X_i-Z_j||_2^2)$
        Kᵢⱼ = exp(−γ‖Xᵢ−Zⱼ‖₂²)

        The distance identity $||X_i-Z_j||^2=||X_i||^2+||Z_j||^2-2*X_i^T*Z_j$ (‖Xᵢ−Zⱼ‖² = ‖Xᵢ‖²+‖Zⱼ‖²−2XᵢᵀZⱼ) replaces Python sample loops with matrix multiplication. Clipping negative round-off distances to zero preserves the fixed-shape exponential calculation

    :parameters:
        - X (jnp.ndarray) - (n,p) First feature matrix
        - Z (jnp.ndarray) - (m,p) Second feature matrix
        - gamma (float) - Positive RBF kernel scale

    :return:
        - kernel (jnp.ndarray) - (n,m) RBF kernel matrix
    '''

    x_squared = jnp.sum(X * X,axis=1)[:,None]
    z_squared = jnp.sum(Z * Z,axis=1)[None,:]
    distance_squared = x_squared + z_squared - 2.0 * (X @ Z.T)
    distance_squared = jnp.maximum(distance_squared,0.0)

    return jnp.exp(-gamma * distance_squared)



def _svr_kkt_bounds(beta,grad,C,epsilon):
    '''Method Function:

        Define the lower and upper KKT bounds for the SVR intercept. With $r=y-K*beta=-grad$ (r = y−Kβ = −grad), inactive set members are represented by signed infinities so every operation retains shape (n,):

        $L={r_i-epsilon:beta_i<C} union {r_i+epsilon:beta_i<0}$
        𝓛 = {rᵢ−ε : βᵢ<C} ∪ {rᵢ+ε : βᵢ<0}

        $U={r_i+epsilon:beta_i>-C} union {r_i-epsilon:beta_i>0}$
        𝓤 = {rᵢ+ε : βᵢ>−C} ∪ {rᵢ−ε : βᵢ>0}

        The violation $nu=max(L)-min(U)$ (ν = max 𝓛−min 𝓤) controls working-set selection and stopping; the final intercept is the midpoint of the two extrema

    :parameters:
        - beta (jnp.ndarray) - (n,) Compact dual variables
        - grad (jnp.ndarray) - (n,) Unregularized dual gradient K@beta-y
        - C (float) - Positive box-constraint bound
        - epsilon (float) - Non-negative epsilon-insensitive width

    :return:
        - lower (jnp.ndarray) - (n,) Candidate lower intercept bounds
        - upper (jnp.ndarray) - (n,) Candidate upper intercept bounds
    '''

    residual_without_b = -grad
    negative_infinity = jnp.asarray(-jnp.inf,dtype=beta.dtype)
    positive_infinity = jnp.asarray(jnp.inf,dtype=beta.dtype)
    lower = jnp.where(beta < C,residual_without_b - epsilon,negative_infinity)
    lower = jnp.maximum(
        lower,jnp.where(beta < 0.0,residual_without_b + epsilon,negative_infinity)
    )
    upper = jnp.where(beta > -C,residual_without_b + epsilon,positive_infinity)
    upper = jnp.minimum(
        upper,jnp.where(beta > 0.0,residual_without_b - epsilon,positive_infinity)
    )

    return lower,upper



def _solve_svr_pair(K,beta,grad,C,epsilon,i,j):
    '''Method Function:

        Solve the exact one-dimensional pair subproblem beta_i += t and beta_j -= t. Opposite updates preserve the dual equality constraint, while $t_min=max(-C-beta_i,beta_j-C)$ and $t_max=min(C-beta_i,beta_j+C)$ (t_min = max(−C−βᵢ,βⱼ−C), t_max = min(C−βᵢ,βⱼ+C)) enforce both box constraints:

        $DeltaF(t)=(grad_i-grad_j)*t+0.5*a*t^2+epsilon*(|beta_i+t|-|beta_i|+|beta_j-t|-|beta_j|)$
        ΔF(t) = (gᵢ−gⱼ)t + ½at² + ε(|βᵢ+t|−|βᵢ|+|βⱼ−t|−|βⱼ|)

        $a=K_ii+K_jj-2*K_ij$ (a = Kᵢᵢ+Kⱼⱼ−2Kᵢⱼ) is the nonnegative curvature. Absolute-value knots at -beta_i and beta_j divide the feasible interval into linear-sign regions. Evaluating boundaries, clipped knots, per-interval stationary points and t=0 gives a fixed-size exact candidate set, including the zero-curvature fallback

    :parameters:
        - K (jnp.ndarray) - (n,n) Training kernel matrix
        - beta (jnp.ndarray) - (n,) Current compact dual variables
        - grad (jnp.ndarray) - (n,) Current unregularized dual gradient
        - C (float) - Positive box-constraint bound
        - epsilon (float) - Non-negative epsilon-insensitive width
        - i (int or jax.Array) - First working-set index
        - j (int or jax.Array) - Second working-set index

    :return:
        - step (jnp.ndarray) - Scalar optimal feasible pair-update step
    '''

    beta_i = beta[i]
    beta_j = beta[j]
    lower = jnp.maximum(-C - beta_i,beta_j - C)
    upper = jnp.minimum(C - beta_i,beta_j + C)
    curvature = K[i,i] + K[j,j] - 2.0 * K[i,j]
    curvature = jnp.maximum(curvature,0.0)
    linear = grad[i] - grad[j]

    ### The feasible boundaries and the two absolute-value knots partition the objective
    knots = jnp.sort(jnp.array([
        lower,
        upper,
        jnp.clip(-beta_i,lower,upper),
        jnp.clip(beta_j,lower,upper)
    ],dtype=beta.dtype))
    left = knots[:-1]
    right = knots[1:]
    middle = 0.5 * (left + right)
    sign_i = jnp.sign(beta_i + middle)
    sign_j = jnp.sign(beta_j - middle)
    slope = linear + epsilon * (sign_i - sign_j)
    safe_curvature = jnp.where(curvature > 1e-12,curvature,1.0)
    quadratic_points = jnp.clip(-slope / safe_curvature,left,right)
    quadratic_points = jnp.where(curvature > 1e-12,quadratic_points,left)
    candidates = jnp.concatenate([
        knots,quadratic_points,jnp.array([0.0],dtype=beta.dtype)
    ])
    objective_change = (
        linear * candidates
        + 0.5 * curvature * candidates ** 2
        + epsilon * (
            jnp.abs(beta_i + candidates) - jnp.abs(beta_i)
            + jnp.abs(beta_j - candidates) - jnp.abs(beta_j)
        )
    )

    return candidates[jnp.argmin(objective_change)]



class _SMOSVRState(NamedTuple):
    '''Class Introduction:

        This is the fixed-shape PyTree state used by the compiled SMO while loop
    '''

    beta: jax.Array
    grad: jax.Array
    iteration: jax.Array
    violation: jax.Array
    stalled: jax.Array



def _smo_svr(K,y,C,epsilon,tolerance,max_iterations):
    '''Method Function:

        Solve compact-dual SVR through second-order working-set SMO on the device. Every iteration computes $nu=max(L)-min(U)$ (ν = max 𝓛−min 𝓤), selects the maximum-violation first index and scores all eligible second indices by $descent^2/max(curvature,delta)$ (descent²/max(curvature,δ))

        The exact pair step updates $beta_i <- beta_i+t$ and $beta_j <- beta_j-t$ (βᵢ ← βᵢ+t, βⱼ ← βⱼ−t), followed by $grad <- grad+t*(K[:,i]-K[:,j])$ (g ← g+t(K[:,i]−K[:,j])). A zero-step second-order candidate falls back to the maximum-violation pair. lax.while_loop stops on tolerance, the iteration budget or numerical stagnation without Python host dispatch per iteration

    :parameters:
        - K (jnp.ndarray) - (n,n) Training kernel matrix
        - y (jnp.ndarray) - (n,) Standardized training targets
        - C (float) - Positive box-constraint bound
        - epsilon (float) - Non-negative epsilon-insensitive width
        - tolerance (float) - Positive KKT convergence tolerance
        - max_iterations (int) - Positive maximum number of pair updates

    :return:
        - state (_SMOSVRState) - Final dual, gradient and convergence state
        - intercept (jnp.ndarray) - Intercept in the standardized target space
    '''

    n = y.shape[0]
    diagonal = jnp.diag(K)
    indices = jnp.arange(n)

    def bounds_and_violation(beta,grad):
        '''Method Function:

            Evaluate both KKT intercept bounds and their scalar maximum-violation residual for the current dual state
        '''

        lower,upper = _svr_kkt_bounds(beta,grad,C,epsilon)
        violation = jnp.max(lower) - jnp.min(upper)
        return lower,upper,violation

    initial_state = _SMOSVRState(
        beta=jnp.zeros(n,dtype=jnp.float64),
        grad=-y,
        iteration=jnp.array(0,dtype=jnp.int32),
        violation=jnp.array(jnp.inf,dtype=jnp.float64),
        stalled=jnp.array(False)
    )

    def condition(state):
        '''Method Function:

            Continue the device-side loop only while iterations remain, KKT violation exceeds tolerance and no stagnation is flagged
        '''

        return (
            (state.iteration < max_iterations)
            & (state.violation > tolerance)
            & ~state.stalled
        )

    def body(state):
        '''Method Function:

            Execute one second-order working-set selection, exact pair update and O(n) gradient correction with immutable state
        '''

        beta = state.beta
        grad = state.grad
        lower,upper,violation = bounds_and_violation(beta,grad)
        i = jnp.argmax(lower)
        j_mvp = jnp.argmin(upper)

        ### Score every feasible second variable against the maximum-violation first variable
        curvature = diagonal[i] + diagonal - 2.0 * K[i,:]
        curvature = jnp.maximum(curvature,0.0)
        derivative_plus_i = jnp.where(beta[i] >= 0.0,1.0,-1.0)
        derivative_minus = jnp.where(beta <= 0.0,1.0,-1.0)
        directional = (
            grad[i] - grad + epsilon * (derivative_plus_i - derivative_minus)
        )
        step_lower = jnp.maximum(-C - beta[i],beta - C)
        step_upper = jnp.minimum(C - beta[i],beta + C)
        descent = jnp.maximum(
            jnp.where(step_upper > 1e-14,-directional,0.0),
            jnp.where(step_lower < -1e-14,directional,0.0)
        )
        eligible = upper < lower[i] - 0.5 * violation
        score = jnp.where(
            (descent > 0.0) & (indices != i) & eligible,
            descent ** 2 / jnp.maximum(curvature,1e-12),
            -jnp.inf
        )
        j_second_order = jnp.argmax(score)
        j = jnp.where(jnp.isfinite(score[j_second_order]),j_second_order,j_mvp)
        step = _solve_svr_pair(K,beta,grad,C,epsilon,i,j)

        ### Fall back to the maximum-violation pair when the second-order candidate stalls
        use_fallback = jnp.abs(step) < 1e-14
        fallback_step = _solve_svr_pair(K,beta,grad,C,epsilon,i,j_mvp)
        j = jnp.where(use_fallback,j_mvp,j)
        step = jnp.where(use_fallback,fallback_step,step)

        ### Opposite updates preserve the equality constraint; two kernel columns update grad in O(n)
        beta = beta.at[i].add(step).at[j].add(-step)
        grad = grad + step * (K[:,i] - K[:,j])
        _,_,new_violation = bounds_and_violation(beta,grad)

        return _SMOSVRState(
            beta=beta,
            grad=grad,
            iteration=state.iteration + 1,
            violation=new_violation,
            stalled=jnp.abs(step) < 1e-15
        )

    state = lax.while_loop(condition,body,initial_state)
    lower,upper = _svr_kkt_bounds(state.beta,state.grad,C,epsilon)
    intercept = 0.5 * (jnp.max(lower) + jnp.min(upper))

    return state,intercept



def _fit_smo_svr(X,y,C,epsilon,gamma,tolerance,max_iterations,support_tolerance):
    '''Method Function:

        Train RBF SVR and retain only the support vectors required for inference. Training-only preprocessing computes $X_s=(X-mu_X)/sigma_X$ and $y_s=(y-mu_y)/sigma_y$ (Xₛ = (X−μₓ)/σₓ, yₛ = (y−μ_y)/σ_y), with positive fallbacks for near-zero scales. Consequently, epsilon and tolerance are measured in standardized target units

        One n×n RBF matrix is passed to a JIT closure whose hyperparameters and iteration budget are static. After solving, the full beta is retained for diagnostics while $S={i:|beta_i|>support_tolerance}$ (𝒮 = {i : |βᵢ| > support_tolerance}) determines the standardized support vectors and dual_coef used by reasoning. metrics also compares the incrementally maintained gradient with K@beta-y_s and reports the equality residual, KKT violation and honest convergence state

    :parameters:
        - X (jnp.ndarray) - (n,p) Validated training features
        - y (jnp.ndarray) - (n,) Validated training targets
        - C (float) - Positive box-constraint bound
        - epsilon (float) - Non-negative epsilon-insensitive width
        - gamma (float) - Positive RBF kernel scale
        - tolerance (float) - Positive KKT convergence tolerance
        - max_iterations (int) - Positive maximum number of pair updates
        - support_tolerance (float) - Non-negative support-vector pruning threshold

    :return:
        - result (dict) - Complete dual variables, sparse inference state and diagnostics
    '''

    x_mean = jnp.mean(X,axis=0)
    x_scale = jnp.maximum(jnp.std(X,axis=0),1e-12)
    y_mean = jnp.mean(y)
    y_scale = jnp.maximum(jnp.std(y),1e-12)
    X_standardized = (X - x_mean) / x_scale
    y_standardized = (y - y_mean) / y_scale
    K = _svr_rbf_kernel(X_standardized,X_standardized,gamma)

    ### Hyperparameters and the iteration limit are static closure values for compilation
    solve = jax.jit(
        lambda kernel,target: _smo_svr(
            kernel,target,C,epsilon,tolerance,max_iterations
        )
    )
    state,intercept = solve(K,y_standardized)
    if (
        not bool(jnp.all(jnp.isfinite(state.beta)))
        or not bool(jnp.all(jnp.isfinite(state.grad)))
        or not bool(jnp.isfinite(intercept))
        or not bool(jnp.isfinite(state.violation))
    ):
        raise RuntimeError("SVR optimization produced non-finite parameters or diagnostics")

    beta_values = np.asarray(state.beta)
    support_mask = np.abs(beta_values) > support_tolerance
    support_indices = np.flatnonzero(support_mask)
    gradient_error = float(jnp.linalg.norm(
        state.grad - (K @ state.beta - y_standardized),ord=jnp.inf
    ))
    metrics = {
        "n_iter": int(state.iteration),
        "kkt_violation": float(state.violation),
        "stalled": bool(state.stalled),
        "converged": bool(state.violation <= tolerance),
        "equality_residual": float(np.sum(beta_values)),
        "gradient_error": gradient_error,
        "kernel_shape": tuple(K.shape),
        "n_support": int(support_indices.size),
        "C": C,
        "epsilon": epsilon,
        "gamma": gamma,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "support_tolerance": support_tolerance
    }

    return {
        "beta_hat": state.beta,
        "support_vectors": X_standardized[support_mask],
        "dual_coef": state.beta[support_mask],
        "support_indices": jnp.asarray(support_indices,dtype=jnp.int32),
        "intercept": intercept,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "metrics": metrics
    }



@jax.jit
def _predict_smo_svr(A_test,support_vectors,dual_coef,intercept,x_mean,x_scale,y_mean,y_scale,gamma):
    '''Method Function:

        Define JIT-compiled sparse kernel inference with target inverse standardization. Test features reuse the training mean and scale; no test statistic is estimated. With $K_test in R^(m*s)$ (K_test ∈ R^(m×s)), the two-stage prediction is:

        $f_s=K_test@dual_coef+intercept$
        fₛ = K_test·dual_coef + b

        $y_pred=y_mean+y_scale*f_s$
        ŷ = μ_y + σ_y fₛ

        When s=0, the empty matrix product contributes zeros and the result is the constant target model. No training-kernel allocation or SMO iteration is repeated during inference

    :parameters:
        - A_test (jnp.ndarray) - (m,p) Test feature matrix
        - support_vectors (jnp.ndarray) - (s,p) Standardized support vectors
        - dual_coef (jnp.ndarray) - (s,) Non-zero compact dual coefficients
        - intercept (jnp.ndarray) - Intercept in the standardized target space
        - x_mean (jnp.ndarray) - (p,) Training feature means
        - x_scale (jnp.ndarray) - (p,) Training feature scales
        - y_mean (jnp.ndarray) - Scalar training target mean
        - y_scale (jnp.ndarray) - Scalar training target scale
        - gamma (float) - Positive RBF kernel scale

    :return:
        - y_pred (jnp.ndarray) - (m,) Predictions in the original target space
    '''

    X_standardized = (A_test - x_mean) / x_scale
    K_test = _svr_rbf_kernel(X_standardized,support_vectors,gamma)
    standardized_prediction = K_test @ dual_coef + intercept

    return y_mean + y_scale * standardized_prediction



class SMOSVRComponent(BaseRegression):
    '''Class Introduction:

        This is an RBF support vector regression component based on compact-dual second-order sequential minimal optimization (SMO), exact pair subproblems and a JIT-compiled lax.while_loop. It solves the original epsilon-insensitive kernel SVR rather than a kernel approximation. The component is intended for small and medium dense data whose precomputed kernel matrix fits in memory, teaching, algorithm verification and reproducible numerical reference workflows

    Mathematical principle
    ----------------------
    (1) Primal epsilon-SVR: given n samples $D={(x_i,y_i)}$ (𝒟 = {(xᵢ,yᵢ)}), $x_i in R^p$ (xᵢ ∈ Rᵖ) and $y_i in R$ (yᵢ ∈ R), a feature map $phi(x)$ (φ(x)) and an intercept b, the primal optimization problem is:

        $min_{w,b,xi,xi^*} 0.5*||w||_2^2 + C*sum_i(xi_i+xi_i^*)$
        min ½‖w‖₂² + CΣᵢ(ξᵢ+ξᵢ*), subject to:

        $y_i-(w^T*phi(x_i)+b) <= epsilon+xi_i$
        yᵢ − (wᵀφ(xᵢ)+b) ≤ ε+ξᵢ

        $(w^T*phi(x_i)+b)-y_i <= epsilon+xi_i^*$
        (wᵀφ(xᵢ)+b) − yᵢ ≤ ε+ξᵢ*

        $xi_i >= 0 , xi_i^* >= 0$
        ξᵢ ≥ 0, ξᵢ* ≥ 0

        C>0 controls the trade-off between function complexity and violations outside the epsilon-insensitive tube. epsilon>=0 defines the tube width. Inside the tube, deviations do not contribute to the loss; outside it, slack variables measure the excess

    (2) Compact dual representation: the standard dual maintains alpha and alpha* in [0,C]^n. At an optimum, both variables for one observation need not be positive simultaneously, so this component uses:

        $beta=alpha-alpha^* , alpha+alpha^*=|beta|$
        β = α−α*, α+α* = |β|

        With $K_ij=k(x_i,x_j)$ (Kᵢⱼ = k(xᵢ,xⱼ)), the equivalent compact dual is:

        $min_beta F(beta)=0.5*beta^T*K*beta + epsilon*||beta||_1-y^T*beta$
        min F(β) = ½βᵀKβ + ε‖β‖₁ − yᵀβ

        $1^T*beta=0 , -C <= beta_i <= C$
        1ᵀβ = 0, −C ≤ βᵢ ≤ C

        This reduces the state from 2n variables to n, avoids a full 2n×2n block Q matrix and makes beta the direct coefficient vector of the kernel expansion. The y supplied to this optimization is the standardized target, so epsilon and tolerance are interpreted in that space

    (3) Training-only standardization and RBF kernel: features and targets are standardized using training statistics only:

        $X_s=(X-mu_X)/sigma_X , y_s=(y-mu_y)/sigma_y$
        Xₛ = (X−μₓ)/σₓ, yₛ = (y−μ_y)/σ_y

        Near-zero scales receive a positive numerical fallback. Test data never participates in estimating these values. The RBF kernel is:

        $K_ij=exp(-gamma*||x_i-x_j||_2^2) , gamma>0$
        Kᵢⱼ = exp(−γ‖xᵢ−xⱼ‖₂²), γ > 0

        Squared distances are evaluated with $||x-z||^2=||x||^2+||z||^2-2*x^T*z$ (‖x−z‖² = ‖x‖²+‖z‖²−2xᵀz). Small negative values caused by round-off are clipped to zero before exponentiation. Only one n×n training kernel is stored

    (4) Pair updates preserve the equality constraint: each SMO iteration chooses two indices i,j and updates:

        $beta_i <- beta_i+t , beta_j <- beta_j-t$
        βᵢ ← βᵢ+t, βⱼ ← βⱼ−t

        Starting from beta=0, every update preserves $sum_k beta_k=0$ (Σₖβₖ = 0). The feasible interval is:

        $t_min=max(-C-beta_i,beta_j-C) , t_max=min(C-beta_i,beta_j+C)$
        t_min = max(−C−βᵢ, βⱼ−C), t_max = min(C−βᵢ, βⱼ+C)

        Define the unregularized gradient $g=K*beta-y$ (g = Kβ−y) and curvature $a=K_ii+K_jj-2*K_ij$ (a = Kᵢᵢ+Kⱼⱼ−2Kᵢⱼ). Along the pair direction, the exact objective change is:

        $DeltaF(t)=(g_i-g_j)*t+0.5*a*t^2+epsilon*(|beta_i+t|-|beta_i|+|beta_j-t|-|beta_j|)$
        ΔF(t) = (gᵢ−gⱼ)t + ½at² + ε(|βᵢ+t|−|βᵢ|+|βⱼ−t|−|βⱼ|)

        The L1 terms make this a piecewise one-dimensional quadratic function. Its knots are t=-beta_i and t=beta_j. The solver checks both boundaries, both clipped knots, the stationary point in every linear interval and the current point t=0, then selects the candidate with the smallest objective change. Near-zero curvature is handled without an unsafe division

    (5) Working-set selection and O(n) gradient update: the first index is the maximum KKT violator. For every candidate second index, the implementation computes a feasible directional derivative, curvature and approximate descent benefit:

        $score=descent^2/max(curvature,delta)$
        score = descent²/max(curvature,δ)

        Only candidates that still form a significant violation with the first index are eligible. If the second-order choice produces no effective step, the solver falls back to the maximum-violation pair. After accepting t, the gradient is updated as:

        $g <- g+t*(K[:,i]-K[:,j])$
        g ← g + t(K[:,i]−K[:,j])

        This is O(n), avoiding a fresh O(n²) K@beta multiplication on every iteration. Unit tests independently recompute K@beta-y to detect accumulated update errors

    (6) KKT bounds, stopping and intercept: define the residual without the intercept as $r=y-K*beta=-g$ (r = y−Kβ = −g). Candidate lower and upper intercept sets are:

        $L={r_i-epsilon:beta_i<C} union {r_i+epsilon:beta_i<0}$
        𝓛 = {rᵢ−ε : βᵢ<C} ∪ {rᵢ+ε : βᵢ<0}

        $U={r_i+epsilon:beta_i>-C} union {r_i-epsilon:beta_i>0}$
        𝓤 = {rᵢ+ε : βᵢ>−C} ∪ {rᵢ−ε : βᵢ>0}

        At exact optimality, $max(L)<=b<=min(U)$ (max 𝓛 ≤ b ≤ min 𝓤). The stopping residual is:

        $nu=max(L)-min(U)$
        ν = max 𝓛 − min 𝓤

        Iteration stops when nu<=tolerance, the update budget is reached or a numerically stalled step is detected. The returned intercept is $b=0.5*(max(L)+min(U))$ (b = ½(max 𝓛+min 𝓤)). At positive tolerance, this midpoint can differ from either bound by approximately half the remaining interval; reaching max_iterations is never relabelled as convergence

    (7) Support vectors and prediction: observations with nonzero beta are the support vectors. The component applies the configured threshold:

        $S={i:|beta_i|>support_tolerance}$
        𝒮 = {i : |βᵢ| > support_tolerance}

        Only standardized support vectors and their nonzero coefficients are retained for inference. For m test samples and s support vectors, $K_test in R^(m*s)$ (K_test ∈ R^(m×s)) gives:

        $f_s(x)=sum_(i in S) beta_i*k(x_i,x)+b$
        f_s(x) = Σᵢ∈𝒮βᵢk(xᵢ,x)+b

        $y_pred=mu_y+sigma_y*f_s(x)$
        ŷ = μ_y + σ_y f_s(x)

        With no support vectors, the matrix product is zero and prediction reduces to the constant target model. Pruning small nonzero coefficients can introduce a small approximation; support_tolerance=0 retains every nonzero coefficient

    (8) Why precomputed-K second-order SMO: the implementation chooses exactness, fixed shapes and transparent numerical checks rather than universal scalability:

        ============================  ============================================  ==========================================
        Method                        Main property                                 Typical scenario
        ============================  ============================================  ==========================================
        Full-matrix interior point     Standard convex-QP formulation               Small teaching and benchmark problems
        LIBSVM dynamic-cache SMO       Mature exact solver with configurable cache General exact kernel regression
        Python loop + JAX operations   Simple prototype                            Debugging, with host scheduling overhead
        Precomputed K + JIT SMO        Exact dual, fixed shapes, device-side loop   This small/medium dense implementation
        Nystrom / RFF + linear SVR     Lower memory and higher matrix parallelism   Large-scale approximate kernel regression
        ============================  ============================================  ==========================================

        Precomputing K costs O(n^2) memory. Work-set selection and gradient updates are primarily O(n) per iteration, but the number of serial iterations is data and tolerance dependent

    Algorithm flow
    --------------
    .. code-block:: text

        Input A(n×p), b(n,), C, epsilon, gamma and solver controls
            |
            v
        Validate shapes, finite values and hyperparameters
            |
            v
        Training-only X/y standardization -> precompute RBF K(n,n)
            |
            v
        Initialize beta=0, grad=-y_s, iteration=0, stalled=False
            |
            v
        KKT lower/upper bounds -> violation nu=max(L)-min(U)
            |
            +------------> Stop when converged, stalled or at max_iterations
            |
            v
        Maximum-violation first index -> second-order second-index score
            |
            v
        Exact pair subproblem: boundaries + knots + stationary points + t=0
            |
            +------------> Fall back to maximum-violation pair if necessary
            |
            v
        beta_i += t, beta_j -= t -> grad += t*(K[:,i]-K[:,j])
            |
            v
        Recover intercept from KKT midpoint -> extract support vectors
            |
            v
        Inference: training preprocessing -> K_test(m,s) -> kernel expansion
            |
            v
        Inverse target standardization -> y_pred(m,) in original target space

    Step 1 (validation): require a finite two-dimensional A, a finite one-dimensional b, matching sample counts, at least two observations and at least one feature. Validate all hyperparameters before compiling numerical functions

    Step 2 (preprocessing and kernel): estimate X/y means and scales from training data only, standardize both, construct one RBF kernel and never construct the larger 2n×2n block dual matrix

    Step 3 (initialization): initialize beta=0 and grad=-y_s. The zero vector satisfies both the equality and box constraints, and the initial KKT violation is infinite so the compiled loop evaluates its first state consistently

    Step 4 (working-set selection): calculate KKT bounds and select the maximum-violation pair. Use the second-order score to seek a larger productive step and retain the maximum-violation fallback

    Step 5 (exact pair update): enumerate the complete fixed-size candidate set for the piecewise-quadratic objective. Opposite updates preserve the equality constraint and the feasible interval enforces both box constraints

    Step 6 (state update and stopping): update grad from two kernel columns, recalculate the KKT violation and use lax.while_loop conditions for tolerance, iteration limit and numerical stagnation. No Python host loop is placed around individual SMO iterations

    Step 7 (model restoration): calculate the intercept from the final KKT bounds, expose the complete beta_hat for diagnostics and retain only support vectors and coefficients required by reasoning

    Step 8 (inference and verification): standardize test features with training statistics, evaluate the sparse kernel expansion and inverse-standardize the target. Tests check independent kernel values, pair search, KKT, equality/box constraints, gradient consistency, pruning and nonlinear prediction

    JAX unique advantages
    ---------------------
    (1) Fully compiled SMO control flow: KKT evaluation, working-set selection, exact pair solving, immutable updates and stopping all execute inside one jax.jit-compiled lax.while_loop. This avoids Python scheduling and host synchronization on every pair update

    (2) Fixed-shape functional state: _SMOSVRState is a NamedTuple PyTree with stable array shapes and dtypes. JAX arrays use .at[index].add(...) rather than in-place mutation, making data dependencies explicit and XLA-compatible

    (3) Vectorized matrix kernels: pairwise distances use matrix multiplication and broadcasting. Work-set scores are evaluated for all candidate indices simultaneously, and inference constructs only the m×s support-vector kernel

    (4) Reproducible numerical checks: explicit PRNGKey values generate reference data. Identical inputs and parameters reproduce results in the reference environment, while KKT, equality, box and gradient residuals provide stronger checks than RMSE alone

    (5) Backend migration with explicit limits: the same array program can target supported JAX backends. SMO remains a serial active-set iteration, however, so GPU execution is not automatically faster than CPU execution. Cross-backend bitwise identity is not promised

    (6) Composable inference, cautious differentiation claims: support vectors, coefficients and preprocessing arrays are ordinary JAX values suitable for custom inference pipelines. This does not imply that differentiating through data-dependent SMO index selection and while-loop trip counts is a supported training strategy

    Precautions
    -----------
    (1) This is exact kernel SVR, not Nystrom or random-feature approximation. Replacing the RBF kernel requires a valid symmetric positive-semidefinite kernel and renewed numerical testing

    (2) Kernel memory is a hard constraint. In float64, n=10000 requires about 800 MB for K alone, before temporary arrays and backend workspaces. Use an approximate or linear solver when K cannot fit in memory

    (3) epsilon and tolerance live in standardized target space. Changing y_scale changes their original-space meaning. Predictions must always apply the stored inverse transformation

    (4) Maximum iterations is not convergence. Inspect converged, kkt_violation, stalled and n_iter. A finite parameter vector alone is not evidence of an acceptable optimum

    (5) Numerical stagnation can result from very small tolerances, ill-conditioned kernels, unusual scales or floating-point limits. Increasing max_iterations alone does not address stagnation

    (6) Support-vector pruning is approximate when support_tolerance>0. Compare full and sparse predictions for sensitive applications; use support_tolerance=0 to retain all nonzero coefficients

    (7) Constant targets and duplicate samples are supported numerically, but a finite result need not imply a statistically useful model. Empty support sets intentionally produce a constant prediction

    (8) Keep shapes static inside compiled code. Dynamic shrinking, dynamic LRU caches and Python sample loops would change the compilation model and are deliberately outside this implementation

    (9) The complete beta_hat is in standardized target space and is not a primal linear coefficient vector. dual_coef is only its retained nonzero subset, and support_vectors are standardized features

    (10) Cross-version and cross-backend floating-point results can differ slightly, including iteration counts near working-set ties. Validate with KKT and prediction tolerances rather than exact historical scalars

    (11) Model selection and production management are outside scope: cross-validation, gamma/C/epsilon tuning, sample weights, robust scaling, serialization and monitoring require an application layer

    (12) Accuracy or RMSE is not a sufficient correctness proof. Retain independent kernel, exact-pair, KKT, equality, box, gradient and support-pruning tests

    :example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from statjax.regression import SMOSVRComponent
        >>> # Generate reproducible nonlinear regression data
        >>> key = jax.random.PRNGKey(42)
        >>> A = jnp.linspace(-3.0,3.0,40,dtype=jnp.float64)[:,None]
        >>> b = jnp.sin(A[:,0]) + 0.05*jax.random.normal(key,(40,),dtype=jnp.float64)
        >>> # Train the exact RBF dual and perform sparse continuous inference
        >>> svr_instance = SMOSVRComponent(C=10.0,epsilon=0.05,gamma=0.5)
        >>> beta_hat,metrics = svr_instance.train(A=A,b=b)
        >>> y_pred = svr_instance.reasoning(A_test=A[:10])
        >>> support_vectors = svr_instance.support_vectors
        >>> dual_coef = svr_instance.dual_coef
        >>> converged = metrics["converged"]

    :reference:
        - [Support vector regression](https://en.wikipedia.org/wiki/Support_vector_machine#Regression)
        - [Sequential minimal optimization](https://en.wikipedia.org/wiki/Sequential_minimal_optimization)
        - [Radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)
        - [Karush-Kuhn-Tucker conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)
        - StatJAX008 Support Vector Regression Design Document `"StatJAX008 SVM SH V001"<https://github.com/redblue0216/statjax>`_
    '''

    def __init__(self,C=10.0,epsilon=0.05,gamma=0.5,tolerance=1e-4,max_iterations=20000,support_tolerance=1e-7):
        '''Attribute Function:

            Initialize RBF SVR hyperparameters and the untrained model state

        :parameters:
            - C (float) - Positive box-constraint bound
            - epsilon (float) - Non-negative epsilon-insensitive width in standardized target space
            - gamma (float) - Positive RBF kernel scale
            - tolerance (float) - Positive KKT tolerance in standardized target space
            - max_iterations (int) - Positive maximum number of pair updates
            - support_tolerance (float) - Non-negative support-vector pruning threshold
        '''

        parameters = {
            "C": (C,False),
            "epsilon": (epsilon,True),
            "gamma": (gamma,False),
            "tolerance": (tolerance,False),
            "support_tolerance": (support_tolerance,True)
        }
        validated = {}
        for name,(value,allow_zero) in parameters.items():
            if (
                isinstance(value,(bool,np.bool_))
                or not isinstance(value,(int,float,np.integer,np.floating))
            ):
                raise ValueError(f"{name} must be a finite real number")
            value = float(value)
            valid_range = value >= 0.0 if allow_zero else value > 0.0
            if not np.isfinite(value) or not valid_range:
                comparison = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {comparison}")
            validated[name] = value
        if (
            not isinstance(max_iterations,int)
            or isinstance(max_iterations,(bool,np.bool_))
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")

        self.C = validated["C"]
        self.epsilon = validated["epsilon"]
        self.gamma = validated["gamma"]
        self.tolerance = validated["tolerance"]
        self.max_iterations = max_iterations
        self.support_tolerance = validated["support_tolerance"]
        self.beta_hat = None
        self.support_vectors = None
        self.dual_coef = None
        self.support_indices = None
        self.intercept = None
        self.x_mean = None
        self.x_scale = None
        self.y_mean = None
        self.y_scale = None
        self.metrics = None


    def train(self,A,b):
        '''Method Function:

            Train RBF SVR through standardized compact-dual second-order SMO

        :parameters:
            - A (jnp.ndarray) - (n,p) Training feature matrix
            - b (jnp.ndarray) - (n,) Continuous training targets

        :return:
            - beta_hat (jnp.ndarray) - (n,) Complete compact dual variables in standardized target space
            - metrics (dict) - KKT, equality, gradient, support-vector and convergence diagnostics
        '''

        A = jnp.asarray(A,dtype=jnp.float64)
        b = jnp.asarray(b,dtype=jnp.float64)
        if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
            raise ValueError("A must have shape (n,p), b must have shape (n,), and their sample counts must match")
        if A.shape[0] < 2 or A.shape[1] == 0:
            raise ValueError("A and b must contain at least two samples and A must contain at least one feature")
        if not bool(jnp.all(jnp.isfinite(A))) or not bool(jnp.all(jnp.isfinite(b))):
            raise ValueError("A and b must contain only finite values")

        result = _fit_smo_svr(
            A,b,self.C,self.epsilon,self.gamma,self.tolerance,
            self.max_iterations,self.support_tolerance
        )
        self.beta_hat = result["beta_hat"]
        self.support_vectors = result["support_vectors"]
        self.dual_coef = result["dual_coef"]
        self.support_indices = result["support_indices"]
        self.intercept = result["intercept"]
        self.x_mean = result["x_mean"]
        self.x_scale = result["x_scale"]
        self.y_mean = result["y_mean"]
        self.y_scale = result["y_scale"]
        self.metrics = result["metrics"]

        return self.beta_hat,self.metrics


    def reasoning(self,A_test):
        '''Method Function:

            Predict continuous targets using the trained sparse RBF kernel expansion

        :parameters:
            - A_test (jnp.ndarray) - (m,p) Test feature matrix with the training feature dimension

        :return:
            - y_pred (jnp.ndarray) - (m,) Float64 predictions in the original target space
        '''

        if self.beta_hat is None:
            raise RuntimeError("SMOSVRComponent has not been trained yet, please call the train method first")
        A_test = jnp.asarray(A_test,dtype=jnp.float64)
        if A_test.ndim != 2 or A_test.shape[1] != self.x_mean.shape[0]:
            raise ValueError("A_test must have shape (m,p) with the same feature dimension as training data")
        if A_test.shape[0] == 0:
            raise ValueError("A_test must be non-empty")
        if not bool(jnp.all(jnp.isfinite(A_test))):
            raise ValueError("A_test must contain only finite values")

        return _predict_smo_svr(
            A_test,self.support_vectors,self.dual_coef,self.intercept,
            self.x_mean,self.x_scale,self.y_mean,self.y_scale,self.gamma
        )


    def _info(self):
        '''Method Function:

            Return component metadata

        :parameters:
            nothing

        :return:
            - info (str) - Text introduction of the component
        '''

        return "SMOSVRComponent: RBF support vector regression component based on compact-dual second-order SMO and exact pair subproblems, implemented with the JAX technology stack"



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file  
