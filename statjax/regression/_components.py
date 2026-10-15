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

    (1) JAX technology stack (thin QR decomposition, triangular back substitution, thin SVD decomposition, singular value shrinkage, JIT compilation)

- Main functions:

    (1) Regression

Usage examples
--------------
.. code-block:: python
    :linenos:

    from statjax.regression import HouseholderOLSComponent,SVDRidgeComponent

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

Class Description
-----------------
(1)TestRegression: This is a regression test placeholder component implementation class, the main function is to verify the framework link, the main technical inheritance

(2)HouseholderOLSComponent: This is an OLS linear regression component implementation class based on Householder QR decomposition, the main function is to solve the least squares optimal regression coefficients and batch prediction, the main technical JAX technology stack (thin QR decomposition, triangular back substitution, JIT compilation)

(3)SVDRidgeComponent: This is a ridge regression component implementation class based on SVD decomposition, the main function is to solve the L2 regularized optimal regression coefficients and batch prediction, the main technical JAX technology stack (thin SVD decomposition, singular value shrinkage, JIT compilation)

References
----------
- [Ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)

- [Householder transformation](https://en.wikipedia.org/wiki/Householder_transformation)

- [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition)

- [Ridge regression](https://en.wikipedia.org/wiki/Ridge_regression)

- [Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)

- StatJAX001 OLS Design Document `"StatJAX001 OLS SH V003"<https://github.com/redblue0216/statjax>`_

- StatJAX002 Ridge Regression Design Document `"StatJAX002 Ridge SH V002"<https://github.com/redblue0216/statjax>`_
'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
from statjax.regression._base import BaseRegression 
### Algorithm package
import jax
import jax.numpy as jnp
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



##############################################################################################################################################################################
##############################################################################################################################################################################


### End of file  
