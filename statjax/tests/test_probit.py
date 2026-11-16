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

This is a unit test collection of the QRProbitComponent binary Probit regression component

- Design mode:

    (1) nothing

- Key points:

    (1) pytest unit testing framework

    (2) Reproducible Probit data generated with explicit JAX PRNG keys

    (3) Independent correctness checks based on automatic differentiation, column-pivoted QR and optimization history

- Main functions:

    (1) Test the Fisher-scoring and column-pivoted Householder QR training link

    (2) Test stable probability inference and threshold-based classification

    (3) Test input validation, numerical-rank failure, untrained inference and component metadata

Engineering flow closure
------------------------
Reproducible Probit data generation -> stable normal log-CDF likelihood -> Fisher scores and weights -> column-pivoted QR increment -> Armijo step halving -> probability and class inference:

    (1) Generate an intercept-augmented matrix A and labels $b~Bernoulli(Phi(A@beta_true))$ (b ~ Bernoulli(Φ(Aβ_true))) from independent fixed random keys

    (2) Verify the analytic gradient $-(A.T@u)/n$ (−Aᵀu/n) against jax.grad and verify the QR increment against the explicitly evaluated Fisher system

    (3) Require finite log-likelihoods satisfying $ell_(k+1)>=ell_k$ (ℓₖ₊₁ ≥ ℓₖ) within floating-point tolerance, with final mean gradient infinity norm below 1e-8

    (4) Verify $p1=Phi(A_test@beta_hat)$ (p₁ = Φ(A_testβ̂)), normalized probability rows and threshold-consistent int32 classes

    (5) Confirm that extreme correctly classified tails remain finite and exact duplicate columns raise a numerical-rank exception rather than returning arbitrary coefficients

Verification criteria
---------------------
Gradient absolute and relative errors must be below 1e-10; QR relative reconstruction error must be below 1e-10 and the increment must agree with the Fisher normal system; probability rows must be finite, bounded and normalized. Held-out accuracy above 0.70, NLL below 0.50 and Brier score below 0.17 are fixed-data smoke criteria, not substitutes for mathematical checks. Cross-backend bitwise identity is not required

Usage examples
--------------
.. code-block:: python
    :linenos:

    python -m pytest statjax/tests/test_probit.py -v

Class Description
-----------------
(1)TestQRProbitComponent: This is a unit test class of QRProbitComponent, covering train, predict_proba, reasoning and _info

References
----------
pytestORG `"pytest documentation"<https://docs.pytest.org/>`_

StatJAX005 Probit Design Document `"StatJAX005 Probit SH V001"<https://github.com/redblue0216/statjax>`_
'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
import pytest
### Algorithm package
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
### Project Package - Algorithm Components
from statjax.regression import QRProbitComponent
from statjax.regression._components import _probit_irls_qr_step,_probit_loglik,_probit_mean_nll,_probit_nll_gradient,_probit_score_and_weights



####### Global Parameters ##########################################################################
####################################################################################################



### Enable float64 precision before any JAX array creation
jax.config.update("jax_enable_x64",True)

### Reproducible data dimensions and independent train/test sizes
N_TRAIN,N_TEST,N_FEATURES = 1200,300,4
P_DIM = N_FEATURES + 1
### Numerical verification tolerances and end-to-end smoke thresholds
GRADIENT_TOL = 1e-10
QR_RESIDUAL_TOL = 1e-10
PARAMETER_TOL = 0.25
ACCURACY_TOL = 0.70



####### Auxiliary Functions ########################################################################
####################################################################################################



def generate_probit_data(key,n,n_features=N_FEATURES):
    '''Method Function:

        Define a reproducible Probit data generator with an explicit intercept column and Bernoulli labels

    :parameters:
        - key (jax.Array) - Explicit JAX pseudo-random key
        - n (int) - Number of samples
        - n_features (int) - Number of non-intercept features

    :return:
        - A (jnp.ndarray) - (n,n_features+1) Design matrix with intercept in the first column
        - b (jnp.ndarray) - (n,) Binary labels sampled from normal-CDF probabilities
        - beta_true (jnp.ndarray) - (n_features+1,) True intercept and feature coefficients
    '''

    ### Split feature and label random streams so their generation remains independently auditable
    x_key,y_key = jax.random.split(key)
    x = jax.random.normal(x_key,(n,n_features),dtype=jnp.float64)
    ### The first column is the intercept, matching the public component's caller-prepared design contract
    A = jnp.concatenate([jnp.ones((n,1),dtype=jnp.float64),x],axis=1)
    beta_true = jnp.array([0.4,0.8,-0.6,0.5,-0.3],dtype=jnp.float64)
    ### Sample Bernoulli labels from Phi(eta), rather than deterministically thresholding latent scores
    eta = A @ beta_true
    b = jax.random.bernoulli(y_key,norm.cdf(eta)).astype(jnp.float64)

    return A,b,beta_true



####### Test Fixtures ##############################################################################
####################################################################################################



@pytest.fixture(scope="module")
def trained_component():
    '''Method Function:

        Define a module-level fixture that shares one trained component and independent train/test datasets

    :parameters:
        nothing

    :return:
        - data (dict) - Trained component, train/test arrays, true parameters and metrics
    '''

    A_train,b_train,beta_true = generate_probit_data(jax.random.PRNGKey(2026),N_TRAIN)
    A_test,b_test,_ = generate_probit_data(jax.random.PRNGKey(2027),N_TEST)
    component = QRProbitComponent()
    beta_hat,metrics = component.train(A=A_train,b=b_train)

    return {
        "component": component,
        "A_train": A_train,
        "b_train": b_train,
        "A_test": A_test,
        "b_test": b_test,
        "beta_true": beta_true,
        "beta_hat": beta_hat,
        "metrics": metrics
    }



####### Test Classes ###############################################################################
####################################################################################################



class TestQRProbitComponent:
    '''Class Introduction:

        This is a unit test class of QRProbitComponent, covering numerical correctness, public inference interfaces and error behavior
    '''

    def test_train_output_shape(self,trained_component):
        '''Method Function:

            Test trained parameter shape, float64 precision and finite values
        '''

        beta_hat = trained_component["beta_hat"]
        assert beta_hat.shape == (P_DIM,)
        assert beta_hat.dtype == jnp.float64
        assert bool(jnp.all(jnp.isfinite(beta_hat)))


    def test_gradient_matches_autodiff(self,trained_component):
        '''Method Function:

            Test the analytic mean negative log-likelihood gradient against JAX automatic differentiation
        '''

        A_train = trained_component["A_train"][:64]
        b_train = trained_component["b_train"][:64]
        beta = jnp.linspace(-0.2,0.3,P_DIM,dtype=jnp.float64)
        manual = _probit_nll_gradient(beta,A_train,b_train)
        automatic = jax.grad(_probit_mean_nll)(beta,A_train,b_train)
        absolute_error = jnp.max(jnp.abs(manual - automatic))
        relative_error = jnp.linalg.norm(manual - automatic) / (1.0 + jnp.linalg.norm(automatic))

        assert absolute_error < GRADIENT_TOL
        assert relative_error < GRADIENT_TOL


    def test_pivoted_qr_step(self,trained_component):
        '''Method Function:

            Test QR reconstruction, numerical rank, pivot restoration and equivalence to the Fisher normal system
        '''

        A = trained_component["A_train"][:80]
        b = trained_component["b_train"][:80]
        beta = jnp.linspace(-0.1,0.2,P_DIM,dtype=jnp.float64)
        delta,rank,grad_inf,slope,qr_error = _probit_irls_qr_step(beta,A,b)
        u,w = _probit_score_and_weights(beta,A,b)
        expected = jnp.linalg.solve(A.T @ (w[:,None] * A),A.T @ u)

        assert int(rank) == P_DIM
        assert float(qr_error) < QR_RESIDUAL_TOL
        assert float(grad_inf) > 0.0
        assert float(slope) > 0.0
        np.testing.assert_allclose(np.asarray(delta),np.asarray(expected),rtol=1e-8,atol=1e-8)


    def test_loglik_history_and_convergence(self,trained_component):
        '''Method Function:

            Test finite monotone log-likelihood history, convergence state and final gradient norm
        '''

        metrics = trained_component["metrics"]
        history = metrics["loglik_history"]
        assert metrics["converged"]
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(history))))
        assert all(later >= earlier - 1e-8 for earlier,later in zip(history,history[1:]))
        assert metrics["initial_loglik"] == history[0]
        assert metrics["final_loglik"] == history[-1]
        assert metrics["final_loglik"] >= metrics["initial_loglik"]
        assert metrics["n_iter"] == len(history) - 1
        assert metrics["grad_inf"] < 1e-8


    def test_parameter_recovery(self,trained_component):
        '''Method Function:

            Test that estimated coefficients are close to the true values on fixed simulated data
        '''

        parameter_error = jnp.linalg.norm(trained_component["beta_hat"] - trained_component["beta_true"])
        assert parameter_error < PARAMETER_TOL


    def test_predict_proba(self,trained_component):
        '''Method Function:

            Test probability shape, finiteness, range, row normalization and exact normal-CDF values
        '''

        component = trained_component["component"]
        A_test = trained_component["A_test"]
        probabilities = component.predict_proba(A_test=A_test)
        expected_positive = norm.cdf(A_test @ component.beta_hat)

        assert probabilities.shape == (N_TEST,2)
        assert bool(jnp.all(jnp.isfinite(probabilities)))
        assert bool(jnp.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        assert float(jnp.max(jnp.abs(jnp.sum(probabilities,axis=1) - 1.0))) < 1e-12
        assert float(jnp.max(jnp.abs(probabilities[:,1] - expected_positive))) < 1e-12


    def test_reasoning_predict(self,trained_component):
        '''Method Function:

            Test binary output shape, threshold consistency and held-out classification accuracy
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
        assert accuracy > ACCURACY_TOL


    def test_held_out_probability_metrics(self,trained_component):
        '''Method Function:

            Test held-out mean negative log-likelihood and Brier score as end-to-end smoke criteria
        '''

        component = trained_component["component"]
        A_test = trained_component["A_test"]
        b_test = trained_component["b_test"]
        probabilities = component.predict_proba(A_test=A_test)[:,1]
        test_nll = _probit_mean_nll(component.beta_hat,A_test,b_test)
        brier = jnp.mean((probabilities - b_test) ** 2)

        assert float(test_nll) < 0.50
        assert float(brier) < 0.17


    def test_tail_stability(self):
        '''Method Function:

            Test finite likelihood, scores, weights and gradients for extreme but correctly classified linear predictors
        '''

        A = jnp.array([[1.0,1000.0],[1.0,-1000.0],[1.0,0.0]],dtype=jnp.float64)
        b = jnp.array([1.0,0.0,1.0],dtype=jnp.float64)
        beta = jnp.array([0.0,1.0],dtype=jnp.float64)
        loglik = _probit_loglik(beta,A,b)
        u,w = _probit_score_and_weights(beta,A,b)
        gradient = _probit_nll_gradient(beta,A,b)

        assert bool(jnp.isfinite(loglik))
        assert bool(jnp.all(jnp.isfinite(u)))
        assert bool(jnp.all(jnp.isfinite(w)))
        assert bool(jnp.all(jnp.isfinite(gradient)))


    def test_rank_deficient_design(self,trained_component):
        '''Method Function:

            Test that an exact duplicate design column raises an explicit numerical-rank exception
        '''

        A_train = trained_component["A_train"]
        b_train = trained_component["b_train"]
        A_rank_deficient = jnp.concatenate([A_train,A_train[:,1:2]],axis=1)
        component = QRProbitComponent()

        with pytest.raises(np.linalg.LinAlgError,match="numerical rank"):
            component.train(A=A_rank_deficient,b=b_train)


    def test_invalid_inputs(self,trained_component):
        '''Method Function:

            Test expected ValueError behavior for malformed, empty, non-finite, non-binary and incompatible inputs
        '''

        component = QRProbitComponent()
        with pytest.raises(ValueError,match="sample counts must match"):
            component.train(A=jnp.ones((4,2)),b=jnp.array([0.0,1.0]))
        with pytest.raises(ValueError,match="non-empty"):
            component.train(A=jnp.empty((0,2)),b=jnp.empty((0,)))
        with pytest.raises(ValueError,match="n >= p"):
            component.train(A=jnp.ones((2,3)),b=jnp.array([0.0,1.0]))
        with pytest.raises(ValueError,match="finite values"):
            component.train(A=jnp.array([[jnp.nan,1.0],[0.0,1.0]]),b=jnp.array([0.0,1.0]))
        with pytest.raises(ValueError,match="binary labels"):
            component.train(A=jnp.ones((3,2)),b=jnp.array([0.0,1.0,2.0]))
        with pytest.raises(ValueError,match="same column dimension"):
            trained_component["component"].predict_proba(A_test=jnp.ones((2,P_DIM + 1)))


    def test_invalid_hyperparameters(self):
        '''Method Function:

            Test expected ValueError behavior for invalid optimization and classification hyperparameters
        '''

        invalid_parameters = [
            {"max_iter": 0},
            {"max_iter": True},
            {"grad_tol": 0.0},
            {"step_tol": 0.0},
            {"ll_tol": 0.0},
            {"max_halvings": 0},
            {"threshold": 1.1}
        ]
        for parameters in invalid_parameters:
            with pytest.raises(ValueError):
                QRProbitComponent(**parameters)


    def test_max_iter_returns_not_converged(self,trained_component):
        '''Method Function:

            Test that reaching the iteration limit returns an explicit non-converged state
        '''

        component = QRProbitComponent(max_iter=1)
        _,metrics = component.train(A=trained_component["A_train"],b=trained_component["b_train"])

        assert not metrics["converged"]
        assert metrics["n_iter"] == 1
        assert len(metrics["loglik_history"]) == 2
        assert metrics["final_loglik"] >= metrics["initial_loglik"]


    def test_reasoning_before_train(self):
        '''Method Function:

            Test whether probability and class inference before training throw the expected RuntimeError
        '''

        component = QRProbitComponent()
        A_test = jnp.ones((N_TEST,P_DIM),dtype=jnp.float64)
        with pytest.raises(RuntimeError,match="has not been trained yet"):
            component.reasoning(A_test=A_test)
        with pytest.raises(RuntimeError,match="has not been trained yet"):
            component.predict_proba(A_test=A_test)


    def test_info(self):
        '''Method Function:

            Test whether the _info method returns a non-empty string
        '''

        info = QRProbitComponent()._info()
        assert isinstance(info,str)
        assert len(info) > 0



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file
