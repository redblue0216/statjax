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

This is a unit test collection of the QRLogisticComponent binary Logistic regression component

- Design mode:

    (1) nothing

- Key points:

    (1) pytest unit testing framework

    (2) Reproducible correlated binary data generated with explicit JAX PRNG keys

    (3) Independent correctness checks based on automatic differentiation, rank-deficient QR least squares and optimization history

- Main functions:

    (1) Test the IRLS and column-pivoted Householder QR training link

    (2) Test stable probability inference and threshold-based classification

    (3) Test input validation, untrained inference and component metadata

Engineering flow closure
------------------------
Reproducible binary data generation -> training-only standardization -> IRLS weighted least squares -> pivoted QR rank truncation -> Armijo backtracking -> probability and class inference:

    (1) Generate $Z~N(0,I)$, $X=Z*M+epsilon$, $eta=0.35+X*w_true$ and $y~Bernoulli(sigmoid(eta))$ (Z ~ N(0,I), X = ZM+ε, η = 0.35+Xw_true, y ~ Bernoulli(σ(η))) from fixed random keys

    (2) Split before training, then minimize $J(beta)=mean(logaddexp(0,eta)-y*eta)+(l2/2)*||w||_2^2$ (J(β) = mean(logaddexp(0,η)−yη)+(λ/2)‖w‖₂²), leaving the intercept unpenalized

    (3) Verify $g=(1/n)*X_tilde^T*(p-y)+l2*D*beta$ (g = (1/n)X̃ᵀ(p−y)+λDβ) against jax.grad, and verify A[:,pivots]=Q@R on a rank-deficient least-squares system

    (4) Require finite losses satisfying J_(k+1)<=J_k within floating-point tolerance, confirming that Armijo acceptance protects each update

    (5) Reuse training preprocessing for $p1=sigmoid(X_tilde_test*beta)$ (p₁ = σ(X̃_testβ)), return [1-p1,p1], and classify with p1>=threshold

Verification criteria
---------------------
The gradient absolute and relative errors must be below 1e-10; the constructed QR system must report rank 2 with residual below 1e-10; probability rows must be finite, bounded and normalized; held-out accuracy above 0.60 is only an end-to-end smoke criterion, not a substitute for the mathematical checks

Usage examples
--------------
.. code-block:: python
    :linenos:

    python -m pytest statjax/tests/test_logistic.py -v

Class Description
-----------------
(1)TestQRLogisticComponent: This is a unit test class of QRLogisticComponent, covering train, predict_proba, reasoning and _info

References
----------
pytestORG `"pytest documentation"<https://docs.pytest.org/>`_

StatJAX004 Logistic Design Document `"StatJAX004 Logistic SH V001"<https://github.com/redblue0216/statjax>`_
'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
import pytest
### Algorithm package
import jax
import jax.numpy as jnp
### Project Package - Algorithm Components
from statjax.regression import QRLogisticComponent
from statjax.regression._components import _add_intercept,_logistic_gradient,_penalized_logistic_loss,_pivoted_qr_lstsq



####### Global Parameters ##########################################################################
####################################################################################################



### Enable float64 precision before any JAX array creation
jax.config.update("jax_enable_x64",True)

### Reproducible data dimensions and train/test split size
N_SAMPLES,P_DIM,LATENT_DIM = 600,8,4
N_TEST = N_SAMPLES // 5
### Default regularization and numerical verification tolerances
L2 = 1e-6
GRADIENT_TOL = 1e-10
QR_RESIDUAL_TOL = 1e-10



####### Auxiliary Functions ########################################################################
####################################################################################################



def generate_logistic_data(n=N_SAMPLES,p=P_DIM,latent_dim=LATENT_DIM):
    '''Method Function:

        Define a reproducible binary data generator with correlated features and Bernoulli labels, using explicit JAX random keys

    :parameters:
        - n (int) - Total number of samples
        - p (int) - Number of observed features
        - latent_dim (int) - Dimension of the latent variables used to correlate features

    :return:
        - A (jnp.ndarray) - (n,p) Correlated feature matrix
        - b (jnp.ndarray) - (n,) Binary labels sampled from Sigmoid probabilities
    '''

    if p < 2:
        raise ValueError("p must be at least 2")

    k_latent,k_mix,k_noise,k_weight,k_label = jax.random.split(jax.random.PRNGKey(42),5)
    latent = jax.random.normal(k_latent,(n,latent_dim),dtype=jnp.float64)
    mixing = jax.random.normal(k_mix,(latent_dim,p),dtype=jnp.float64) / jnp.sqrt(latent_dim)
    A = latent @ mixing + 0.35 * jax.random.normal(k_noise,(n,p),dtype=jnp.float64)
    true_w = 0.8 * jax.random.normal(k_weight,(p,),dtype=jnp.float64)
    logits = 0.35 + A @ true_w
    probabilities = jax.nn.sigmoid(logits)
    b = jax.random.bernoulli(k_label,probabilities).astype(jnp.float64)

    return A,b



####### Test Fixtures ##############################################################################
####################################################################################################



@pytest.fixture(scope="module")
def trained_component():
    '''Method Function:

        Define a module-level fixture that splits reproducible data before training and shares one trained component across test cases

    :parameters:
        nothing

    :return:
        - data (dict) - Trained component, train/test arrays, parameters and metrics
    '''

    A,b = generate_logistic_data()
    permutation = jax.random.permutation(jax.random.PRNGKey(43),N_SAMPLES)
    test_idx,train_idx = permutation[:N_TEST],permutation[N_TEST:]
    A_train,b_train = A[train_idx],b[train_idx]
    A_test,b_test = A[test_idx],b[test_idx]

    component = QRLogisticComponent(l2=L2)
    beta_hat,metrics = component.train(A=A_train,b=b_train)

    return {
        "component": component,
        "A_train": A_train,
        "b_train": b_train,
        "A_test": A_test,
        "b_test": b_test,
        "beta_hat": beta_hat,
        "metrics": metrics
    }



####### Test Classes ###############################################################################
####################################################################################################



class TestQRLogisticComponent:
    '''Class Introduction:

        This is a unit test class of QRLogisticComponent, covering numerical correctness, public inference interfaces and error behavior
    '''


    def test_train_output_shape(self,trained_component):
        '''Method Function:

            Test trained parameter shape, float64 precision and preprocessing state
        '''

        beta_hat = trained_component["beta_hat"]
        component = trained_component["component"]
        assert beta_hat.shape == (P_DIM + 1,)
        assert beta_hat.dtype == jnp.float64
        assert component.mean.shape == (P_DIM,)
        assert component.scale.shape == (P_DIM,)
        assert bool(jnp.all(jnp.isfinite(beta_hat)))


    def test_gradient_matches_autodiff(self,trained_component):
        '''Method Function:

            Test the analytic matrix gradient against JAX automatic differentiation on identical standardized inputs
        '''

        component = trained_component["component"]
        A_train = trained_component["A_train"][:32]
        b_train = trained_component["b_train"][:32]
        Xb = _add_intercept((A_train - component.mean) / component.scale)
        beta = jnp.linspace(-0.2,0.3,Xb.shape[1],dtype=jnp.float64)

        manual = _logistic_gradient(beta,Xb,b_train,L2)
        automatic = jax.grad(lambda value: _penalized_logistic_loss(value,Xb,b_train,L2))(beta)
        absolute_error = jnp.max(jnp.abs(manual - automatic))
        relative_error = jnp.linalg.norm(manual - automatic) / (1.0 + jnp.linalg.norm(automatic))

        assert absolute_error < GRADIENT_TOL
        assert relative_error < GRADIENT_TOL


    def test_pivoted_qr_rank_deficient(self):
        '''Method Function:

            Test pivot restoration and numerical-rank truncation on a matrix whose second column is twice its first column
        '''

        A = jnp.array([[1.0,2.0,0.0],[2.0,4.0,1.0],[3.0,6.0,0.0],[4.0,8.0,1.0]],dtype=jnp.float64)
        expected = jnp.array([0.4,-0.2,0.7],dtype=jnp.float64)
        b = A @ expected
        solution,rank = _pivoted_qr_lstsq(A,b)
        residual = jnp.linalg.norm(A @ solution - b)

        assert rank == 2
        assert residual < QR_RESIDUAL_TOL


    def test_loss_history_and_convergence(self,trained_component):
        '''Method Function:

            Test finite monotone loss history, convergence state and consistent iteration/rank history lengths
        '''

        metrics = trained_component["metrics"]
        losses = metrics["loss_history"]
        assert metrics["converged"]
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(losses))))
        assert all(later <= earlier + 1e-10 for earlier,later in zip(losses,losses[1:]))
        assert metrics["initial_loss"] == losses[0]
        assert metrics["final_loss"] == losses[-1]
        assert metrics["final_loss"] <= metrics["initial_loss"]
        assert metrics["n_iter"] == len(losses) - 1
        assert len(metrics["rank_history"]) in (metrics["n_iter"],metrics["n_iter"] + 1)


    def test_l2_excludes_intercept(self,trained_component):
        '''Method Function:

            Test that L2 changes only feature-weight gradients and leaves the intercept gradient unpenalized
        '''

        component = trained_component["component"]
        A_train = trained_component["A_train"][:16]
        b_train = trained_component["b_train"][:16]
        Xb = _add_intercept((A_train - component.mean) / component.scale)
        beta = jnp.linspace(-0.3,0.4,Xb.shape[1],dtype=jnp.float64)

        unregularized = _logistic_gradient(beta,Xb,b_train,0.0)
        regularized = _logistic_gradient(beta,Xb,b_train,1.0)
        penalty_gradient = regularized - unregularized

        assert float(jnp.abs(penalty_gradient[-1])) == 0.0
        assert float(jnp.max(jnp.abs(penalty_gradient[:-1] - beta[:-1]))) < GRADIENT_TOL


    def test_constant_and_collinear_columns(self,trained_component):
        '''Method Function:

            Test safe standardization and QR rank handling with an additional constant column and an exact duplicate feature
        '''

        A_train = trained_component["A_train"]
        b_train = trained_component["b_train"]
        A_augmented = jnp.concatenate([A_train,jnp.ones((A_train.shape[0],1),dtype=A_train.dtype),A_train[:,:1]],axis=1)
        component = QRLogisticComponent(l2=L2)
        beta_hat,metrics = component.train(A=A_augmented,b=b_train)
        probabilities = component.predict_proba(A_test=A_augmented[:10])

        assert beta_hat.shape == (P_DIM + 3,)
        assert bool(jnp.all(jnp.isfinite(beta_hat)))
        assert bool(jnp.isfinite(metrics["final_loss"]))
        assert probabilities.shape == (10,2)


    def test_predict_proba(self,trained_component):
        '''Method Function:

            Test probability shape, finiteness, range, row normalization and exact reuse of training preprocessing
        '''

        component = trained_component["component"]
        A_test = trained_component["A_test"]
        probabilities = component.predict_proba(A_test=A_test)
        Xb = _add_intercept((A_test - component.mean) / component.scale)
        expected_positive = jax.nn.sigmoid(Xb @ component.beta_hat)

        assert probabilities.shape == (N_TEST,2)
        assert bool(jnp.all(jnp.isfinite(probabilities)))
        assert bool(jnp.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        assert float(jnp.max(jnp.abs(jnp.sum(probabilities,axis=1) - 1.0))) < 1e-12
        assert float(jnp.max(jnp.abs(probabilities[:,1] - expected_positive))) < 1e-12


    def test_reasoning_predict(self,trained_component):
        '''Method Function:

            Test binary output shape, threshold consistency and end-to-end classification accuracy on the held-out data
        '''

        component = trained_component["component"]
        A_test = trained_component["A_test"]
        b_test = trained_component["b_test"]
        probabilities = component.predict_proba(A_test=A_test)
        predictions = component.reasoning(A_test=A_test)
        accuracy = jnp.mean(predictions == b_test.astype(jnp.int32))

        assert predictions.shape == (N_TEST,)
        assert predictions.dtype == jnp.int32
        assert bool(jnp.all((predictions == 0) | (predictions == 1)))
        assert bool(jnp.array_equal(predictions,(probabilities[:,1] >= component.threshold).astype(jnp.int32)))
        assert accuracy > 0.60


    def test_invalid_inputs(self,trained_component):
        '''Method Function:

            Test expected ValueError behavior for mismatched, empty, non-finite, non-binary and feature-incompatible inputs
        '''

        component = QRLogisticComponent()
        with pytest.raises(ValueError,match="sample counts must match"):
            component.train(A=jnp.ones((4,2)),b=jnp.array([0.0,1.0]))
        with pytest.raises(ValueError,match="non-empty"):
            component.train(A=jnp.empty((0,2)),b=jnp.empty((0,)))
        with pytest.raises(ValueError,match="finite values"):
            component.train(A=jnp.array([[jnp.nan,1.0],[0.0,1.0]]),b=jnp.array([0.0,1.0]))
        with pytest.raises(ValueError,match="binary labels"):
            component.train(A=jnp.ones((3,2)),b=jnp.array([0.0,1.0,2.0]))
        with pytest.raises(ValueError,match="same feature dimension"):
            trained_component["component"].predict_proba(A_test=jnp.ones((2,P_DIM + 1)))


    def test_invalid_hyperparameters(self):
        '''Method Function:

            Test expected ValueError behavior for invalid optimization and classification hyperparameters
        '''

        invalid_parameters = [
            {"l2": -1.0},
            {"max_iter": 0},
            {"tol": 0.0},
            {"weight_floor": 0.0},
            {"qr_rtol": 0.0},
            {"max_backtracks": 0},
            {"threshold": 1.1}
        ]
        for parameters in invalid_parameters:
            with pytest.raises(ValueError):
                QRLogisticComponent(**parameters)


    def test_reasoning_before_train(self):
        '''Method Function:

            Test whether probability and class inference before training throw the expected RuntimeError
        '''

        component = QRLogisticComponent()
        A_test = jnp.ones((N_TEST,P_DIM),dtype=jnp.float64)
        with pytest.raises(RuntimeError,match="has not been trained yet"):
            component.reasoning(A_test=A_test)
        with pytest.raises(RuntimeError,match="has not been trained yet"):
            component.predict_proba(A_test=A_test)


    def test_info(self):
        '''Method Function:

            Test whether the _info method returns a non-empty string
        '''

        info = QRLogisticComponent()._info()
        assert isinstance(info,str)
        assert len(info) > 0



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file
