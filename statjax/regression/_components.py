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

This is a collection of regression concrete implementation classes

- Design mode:

    (1) Combination mode

- Key points:

    (1) JAX technology stack (thin QR decomposition, triangular back substitution, thin SVD decomposition, singular value shrinkage, soft thresholding operator, Gauss-Seidel coordinate sweep, rank-1 residual update, JIT compilation)

- Main functions:

    (1) Regression

Usage examples
--------------
.. code-block:: python
    :linenos:

    from statjax.regression import HouseholderOLSComponent,SVDRidgeComponent,CDLassoComponent

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

Class Description
-----------------
(1)TestRegression: This is a regression test placeholder component implementation class, the main function is to verify the framework link, the main technical inheritance

(2)HouseholderOLSComponent: This is an OLS linear regression component implementation class based on Householder QR decomposition, the main function is to solve the least squares optimal regression coefficients and batch prediction, the main technical JAX technology stack (thin QR decomposition, triangular back substitution, JIT compilation)

(3)SVDRidgeComponent: This is a ridge regression component implementation class based on SVD decomposition, the main function is to solve the L2 regularized optimal regression coefficients and batch prediction, the main technical JAX technology stack (thin SVD decomposition, singular value shrinkage, JIT compilation)

(4)CDLassoComponent: This is a LASSO regression component implementation class based on coordinate descent, the main function is to solve the L1 regularized sparse optimal regression coefficients and batch prediction, the main technical JAX technology stack (soft thresholding operator, Gauss-Seidel coordinate sweep, rank-1 residual update, lax.fori_loop compiled control flow, JIT compilation)

References
----------
- [Ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)

- [Householder transformation](https://en.wikipedia.org/wiki/Householder_transformation)

- [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition)

- [Ridge regression](https://en.wikipedia.org/wiki/Ridge_regression)

- [Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)

- [Lasso (statistics)](https://en.wikipedia.org/wiki/Lasso_(statistics))

- [Coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent)

- StatJAX001 OLS Design Document `"StatJAX001 OLS SH V003"<https://github.com/redblue0216/statjax>`_

- StatJAX002 Ridge Regression Design Document `"StatJAX002 Ridge SH V002"<https://github.com/redblue0216/statjax>`_

- StatJAX003 LASSO Design Document `"StatJAX003 LASSO SH V001"<https://github.com/redblue0216/statjax>`_
'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
from statjax.regression._base import BaseRegression 
### Algorithm package
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import qr,solve_triangular
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


### End of file  
