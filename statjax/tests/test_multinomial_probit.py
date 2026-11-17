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

This is a unit test collection of the GHKMultinomialProbitComponent multinomial Probit regression component

- Design mode:

    (1) nothing

- Key points:

    (1) pytest unit testing framework

    (2) Reproducible correlated multinomial Probit data generated with explicit JAX PRNG keys

    (3) Independent correctness checks based on normal-CDF references, probability completeness and automatic differentiation

- Main functions:

    (1) Test scale-constrained Cholesky parameters, choice contrasts, GHK simulation and BFGS training

    (2) Test independently simulated K-class probabilities and argmax classification

    (3) Test input validation, non-convergence reporting, untrained inference and component metadata

Engineering flow closure
------------------------
Reproducible MNP data generation -> scale-constrained Cholesky parameterization -> class-specific negative-orthant contrasts -> GHK simulation -> fixed-draw MSL -> automatic-differentiation BFGS -> independent probability and class inference:

    (1) Generate $U=[A*beta+L*z,0]$ and $y=argmax(U)$ (U = [Aβ+Lz, 0], y = argmax(U)) from explicit independent random keys

    (2) Verify $Sigma=L*L^T$, $L_00=1$ and $Sigma_00=1$ (Σ = LLᵀ, L₀₀ = 1, Σ₀₀ = 1), and compare GHK probabilities with independent one-dimensional normal integrals

    (3) Require finite automatic-differentiation gradients and lower final than initial fixed-draw mean negative simulated log-likelihood

    (4) Return independently simulated J-class probabilities and int32 argmax classes; row normalization is distinguished from raw simulation completeness

Verification criteria
---------------------
The independent CDF error and raw probability-sum error use simulation-aware tolerances. Training must converge and reduce the fixed-draw objective on the reference dataset. Held-out accuracy is only an end-to-end smoke criterion, not a substitute for mathematical checks. Cross-backend bitwise identity and exact iteration counts are not required

Usage examples
--------------
.. code-block:: python
    :linenos:

    python -m pytest statjax/tests/test_multinomial_probit.py -v

Class Description
-----------------
(1)TestGHKMultinomialProbitComponent: This is a unit test class covering numerical correctness, public inference interfaces and error behavior

References
----------
pytestORG `"pytest documentation"<https://docs.pytest.org/>`_

StatJAX007 Multinomial Probit Design Document `"StatJAX007 Multinomial Probit SH V001"<https://github.com/redblue0216/statjax>`_
'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
import math
import pytest
### Algorithm package
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
### Project Package - Algorithm Components
from statjax.regression import GHKMultinomialProbitComponent
from statjax.regression._components import _ghk_mnp_log_probs,_initial_mnp_params,_mnp_choice_contrasts,_mnp_simulated_nll,_num_mnp_cov_params,_predict_ghk_mnp_proba,_unpack_mnp_params



####### Global Parameters ##########################################################################
####################################################################################################



### Enable float64 precision before any JAX array creation
jax.config.update("jax_enable_x64",True)

### Reproducible data dimensions and independent train/test sizes
N_TRAIN,N_TEST,D_DIM,P_DIM = 400,300,2,3
N_CLASSES = D_DIM + 1
### Simulation-aware tolerances and fixed-data smoke threshold
CDF_TOL = 0.02
PROBABILITY_SUM_TOL = 0.08
PARAMETER_TOL = 0.35
ACCURACY_TOL = 0.45



####### Auxiliary Functions ########################################################################
####################################################################################################



def normal_cdf_np(values):
    '''Method Function:

        Evaluate the standard normal CDF using only the standard library and NumPy

    :parameters:
        - values (np.ndarray) - Scalar or array of standard normal thresholds

    :return:
        - probabilities (np.ndarray) - Standard normal CDF values with matching shape
    '''

    shape = np.shape(values)
    flattened = np.ravel(values)
    result = [
        0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))
        for value in flattened
    ]

    return np.asarray(result,dtype=np.float64).reshape(shape)



def reference_negative_orthant_probability(mean,covariance):
    '''Method Function:

        Independently evaluate a D=1 or D=2 negative-orthant normal probability by analytic CDF or deterministic quadrature

    :parameters:
        - mean (np.ndarray) - Transformed multivariate normal mean
        - covariance (np.ndarray) - Transformed positive-definite covariance matrix

    :return:
        - probability (float) - Probability that every coordinate is below zero
    '''

    mean = np.asarray(mean,dtype=np.float64)
    covariance = np.asarray(covariance,dtype=np.float64)
    d = mean.shape[0]
    if d == 1:
        threshold = -mean[0] / math.sqrt(covariance[0,0])
        return float(normal_cdf_np(threshold))
    if d != 2:
        raise ValueError("The lightweight independent reference supports only D=1 or D=2")

    sd0 = math.sqrt(covariance[0,0])
    sd1 = math.sqrt(covariance[1,1])
    rho = covariance[0,1] / (sd0 * sd1)
    rho = float(np.clip(rho,-1.0 + 1e-12,1.0 - 1e-12))
    a = -mean[0] / sd0
    b = -mean[1] / sd1
    nodes,weights = np.polynomial.legendre.leggauss(256)
    lower = min(-12.0,a - 8.0)
    upper = a
    midpoint = 0.5 * (upper + lower)
    half_width = 0.5 * (upper - lower)
    x = midpoint + half_width * nodes
    conditional = normal_cdf_np((b - rho * x) / math.sqrt(1.0 - rho * rho))
    density = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    integral = half_width * np.sum(weights * density * conditional)

    return float(integral)



def generate_multinomial_probit_data(key,n,d=D_DIM,p=P_DIM):
    '''Method Function:

        Generate correlated MNP choices with the final class fixed as the zero-utility reference

    :parameters:
        - key (jax.Array) - Explicit JAX pseudo-random key
        - n (int) - Number of samples
        - d (int) - Utility-difference dimension
        - p (int) - Number of shared utility coefficients

    :return:
        - A (jnp.ndarray) - (n,d,p) Alternative-related design tensor
        - b (jnp.ndarray) - (n,) Integer class labels
        - beta_true (jnp.ndarray) - True shared utility coefficients
        - L_true (jnp.ndarray) - True scale-constrained Cholesky factor
    '''

    key_A,key_error = jax.random.split(key)
    A = jax.random.normal(key_A,(n,d,p),dtype=jnp.float64)
    z = jax.random.normal(key_error,(n,d),dtype=jnp.float64)
    beta_true = jnp.array([0.8,-0.5,0.6],dtype=jnp.float64)[:p]
    L_true = jnp.array([[1.0,0.0],[0.35,0.8]],dtype=jnp.float64)[:d,:d]
    mu = jnp.einsum("ndp,p->nd",A,beta_true)
    errors = z @ L_true.T
    utilities = jnp.concatenate(
        [mu + errors,jnp.zeros((n,1),dtype=A.dtype)],axis=1
    )
    b = jnp.argmax(utilities,axis=1).astype(jnp.int32)

    return A,b,beta_true,L_true



####### Test Fixtures ##############################################################################
####################################################################################################



@pytest.fixture(scope="module")
def trained_component():
    '''Method Function:

        Share one trained component and independent train/test datasets across numerical and interface tests

    :parameters:
        nothing

    :return:
        - data (dict) - Trained component, datasets, true parameters, optimized vector and metrics
    '''

    A_train,b_train,beta_true,L_true = generate_multinomial_probit_data(
        jax.random.PRNGKey(42),N_TRAIN
    )
    A_test,b_test,_,_ = generate_multinomial_probit_data(
        jax.random.PRNGKey(2027),N_TEST
    )
    component = GHKMultinomialProbitComponent(
        num_draws=128,max_iter=80,seed=123,inference_num_draws=512,inference_seed=888
    )
    theta_hat,metrics = component.train(A=A_train,b=b_train)

    return {
        "component": component,
        "A_train": A_train,
        "b_train": b_train,
        "A_test": A_test,
        "b_test": b_test,
        "beta_true": beta_true,
        "L_true": L_true,
        "theta_hat": theta_hat,
        "metrics": metrics
    }



####### Test Classes ###############################################################################
####################################################################################################



class TestGHKMultinomialProbitComponent:
    '''Class Introduction:

        This is a unit test class covering multinomial Probit numerical correctness, public inference and error behavior
    '''

    def test_train_output_shape(self,trained_component):
        '''Method Function:

            Test unconstrained and interpretable parameter shapes, float64 precision and finite state
        '''

        component = trained_component["component"]
        expected_size = P_DIM + _num_mnp_cov_params(D_DIM)
        assert trained_component["theta_hat"].shape == (expected_size,)
        assert trained_component["theta_hat"].dtype == jnp.float64
        assert component.beta_hat.shape == (P_DIM,)
        assert component.l_hat.shape == (D_DIM,D_DIM)
        assert component.sigma_hat.shape == (D_DIM,D_DIM)
        assert bool(jnp.all(jnp.isfinite(trained_component["theta_hat"])))


    def test_cholesky_identification_constraints(self,trained_component):
        '''Method Function:

            Test initial identity parameters, positive diagonal, fixed scale and symmetric positive-definite covariance
        '''

        initial = _initial_mnp_params(D_DIM,P_DIM)
        beta0,L0 = _unpack_mnp_params(initial,D_DIM,P_DIM)
        component = trained_component["component"]
        eigenvalues = jnp.linalg.eigvalsh(component.sigma_hat)

        assert bool(jnp.all(beta0 == 0.0))
        assert bool(jnp.array_equal(L0,jnp.eye(D_DIM,dtype=jnp.float64)))
        assert float(component.l_hat[0,0]) == 1.0
        assert bool(jnp.all(jnp.diag(component.l_hat) > 0.0))
        assert bool(jnp.allclose(jnp.triu(component.l_hat,1),0.0))
        np.testing.assert_allclose(
            np.asarray(component.sigma_hat),
            np.asarray(component.l_hat @ component.l_hat.T),
            rtol=1e-12,atol=1e-12
        )
        assert float(component.sigma_hat[0,0]) == 1.0
        assert bool(jnp.all(eigenvalues > 0.0))


    def test_choice_contrasts(self):
        '''Method Function:

            Test explicit three-class contrast matrices and the final reference-category event
        '''

        contrasts = _mnp_choice_contrasts(3)
        expected = np.array([
            [[-1.0,1.0],[-1.0,0.0]],
            [[1.0,-1.0],[0.0,-1.0]],
            [[1.0,0.0],[0.0,1.0]]
        ])
        np.testing.assert_array_equal(contrasts,expected)


    def test_ghk_matches_independent_cdf(self,trained_component):
        '''Method Function:

            Compare observed-class GHK probabilities with independent normal-CDF integration
        '''

        A = trained_component["A_train"][:8]
        b = trained_component["b_train"][:8]
        theta = jnp.concatenate([
            trained_component["beta_true"],
            jnp.array([0.35,jnp.log(0.8)],dtype=jnp.float64)
        ])
        uniforms = jax.random.uniform(
            jax.random.PRNGKey(99),(A.shape[0],4096,D_DIM),dtype=jnp.float64
        )
        simulated = np.exp(np.asarray(_ghk_mnp_log_probs(theta,A,uniforms)))
        beta,L = _unpack_mnp_params(theta,D_DIM,P_DIM)
        Sigma = np.asarray(L @ L.T)
        mu = np.asarray(jnp.einsum("ndp,p->nd",A,beta))
        contrasts = _mnp_choice_contrasts(N_CLASSES)

        errors = []
        for i in range(A.shape[0]):
            observed = int(b[i])
            C = contrasts[observed]
            reference = reference_negative_orthant_probability(
                mu[i] @ C.T,C @ Sigma @ C.T
            )
            errors.append(abs(simulated[i,observed] - reference))
        assert max(errors) < CDF_TOL


    def test_raw_probability_completeness(self,trained_component):
        '''Method Function:

            Test finite non-negative raw GHK probabilities and approximate completeness without row normalization
        '''

        A = trained_component["A_train"][:32]
        theta = jnp.concatenate([
            trained_component["beta_true"],
            jnp.array([0.35,jnp.log(0.8)],dtype=jnp.float64)
        ])
        uniforms = jax.random.uniform(
            jax.random.PRNGKey(7),(A.shape[0],1024,D_DIM),dtype=jnp.float64
        )
        probabilities = jnp.exp(_ghk_mnp_log_probs(theta,A,uniforms))
        sum_error = float(jnp.max(jnp.abs(jnp.sum(probabilities,axis=1) - 1.0)))

        assert probabilities.shape == (32,N_CLASSES)
        assert bool(jnp.all(jnp.isfinite(probabilities)))
        assert bool(jnp.all(probabilities >= 0.0))
        assert sum_error < PROBABILITY_SUM_TOL


    def test_autodiff_and_fixed_draw_determinism(self,trained_component):
        '''Method Function:

            Test finite JAX gradients and deterministic objectives under identical fixed common random numbers
        '''

        A = trained_component["A_train"][:24]
        b = trained_component["b_train"][:24]
        theta = trained_component["theta_hat"]
        uniforms = jax.random.uniform(
            jax.random.PRNGKey(17),(A.shape[0],128,D_DIM),dtype=jnp.float64
        )
        first = _mnp_simulated_nll(theta,A,b,uniforms)
        second = _mnp_simulated_nll(theta,A,b,uniforms)
        gradient = jax.grad(_mnp_simulated_nll)(theta,A,b,uniforms)

        assert bool(jnp.isfinite(first))
        assert float(first) == float(second)
        assert gradient.shape == theta.shape
        assert bool(jnp.all(jnp.isfinite(gradient)))


    def test_training_convergence_and_nll(self,trained_component):
        '''Method Function:

            Test BFGS convergence, objective decrease, finite gradient and consistent diagnostics
        '''

        metrics = trained_component["metrics"]
        assert metrics["converged"]
        assert metrics["final_nll"] < metrics["initial_nll"]
        assert np.isfinite(metrics["initial_nll"])
        assert np.isfinite(metrics["final_nll"])
        assert np.isfinite(metrics["grad_norm"])
        assert metrics["grad_norm"] < 1e-4
        assert 0 < metrics["n_iter"] <= 80
        assert metrics["num_draws"] == 128
        assert metrics["seed"] == 123


    def test_parameter_recovery(self,trained_component):
        '''Method Function:

            Test loose fixed-data utility-parameter recovery without treating one covariance estimate as exact
        '''

        parameter_error = jnp.linalg.norm(
            trained_component["component"].beta_hat - trained_component["beta_true"]
        )
        assert parameter_error < PARAMETER_TOL


    def test_predict_proba(self,trained_component):
        '''Method Function:

            Test independent inference shape, range, normalization, reproducibility and seed sensitivity
        '''

        component = trained_component["component"]
        A_test = trained_component["A_test"]
        probabilities = component.predict_proba(A_test=A_test)
        repeated = component.predict_proba(A_test=A_test)
        different_seed = _predict_ghk_mnp_proba(
            A_test,component.theta_hat,512,889,True
        )

        assert probabilities.shape == (N_TEST,N_CLASSES)
        assert bool(jnp.all(jnp.isfinite(probabilities)))
        assert bool(jnp.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        assert float(jnp.max(jnp.abs(jnp.sum(probabilities,axis=1) - 1.0))) < 1e-12
        assert bool(jnp.array_equal(probabilities,repeated))
        assert not bool(jnp.allclose(probabilities,different_seed))


    def test_reasoning_predict(self,trained_component):
        '''Method Function:

            Test int32 argmax classes and held-out accuracy as an end-to-end smoke criterion
        '''

        component = trained_component["component"]
        probabilities = component.predict_proba(A_test=trained_component["A_test"])
        predictions = component.reasoning(A_test=trained_component["A_test"])
        accuracy = jnp.mean(predictions == trained_component["b_test"])

        assert predictions.shape == (N_TEST,)
        assert predictions.dtype == jnp.int32
        assert bool(jnp.array_equal(predictions,jnp.argmax(probabilities,axis=1).astype(jnp.int32)))
        assert accuracy > ACCURACY_TOL


    def test_binary_probit_special_case(self):
        '''Method Function:

            Test that J=2 has no free covariance parameter and GHK approximates binary normal-CDF probabilities
        '''

        A = jnp.array([[[0.2]],[[1.0]],[[-0.7]],[[0.4]]],dtype=jnp.float64)
        theta = jnp.array([0.6],dtype=jnp.float64)
        uniforms = jax.random.uniform(
            jax.random.PRNGKey(77),(4,8192,1),dtype=jnp.float64
        )
        probabilities = jnp.exp(_ghk_mnp_log_probs(theta,A,uniforms))
        expected = norm.cdf(0.6 * A[:,0,0])

        assert _num_mnp_cov_params(1) == 0
        np.testing.assert_allclose(
            np.asarray(probabilities[:,0]),np.asarray(expected),rtol=0.0,atol=0.02
        )
        np.testing.assert_allclose(
            np.asarray(probabilities[:,1]),np.asarray(1.0 - expected),rtol=0.0,atol=0.02
        )


    def test_tail_stability(self):
        '''Method Function:

            Test finite log probabilities, simulated loss and gradients under extreme finite design values
        '''

        A = jnp.array([[[1000.0],[-800.0]],[[-1000.0],[700.0]],[[0.0],[0.0]]],dtype=jnp.float64)
        b = jnp.array([0,2,1],dtype=jnp.int32)
        theta = jnp.array([0.8,0.35,jnp.log(0.8)],dtype=jnp.float64)
        uniforms = jax.random.uniform(
            jax.random.PRNGKey(78),(3,128,2),dtype=jnp.float64
        )
        log_probabilities = _ghk_mnp_log_probs(theta,A,uniforms)
        loss = _mnp_simulated_nll(theta,A,b,uniforms)
        gradient = jax.grad(_mnp_simulated_nll)(theta,A,b,uniforms)

        assert bool(jnp.all(jnp.isfinite(log_probabilities)))
        assert bool(jnp.isfinite(loss))
        assert bool(jnp.all(jnp.isfinite(gradient)))


    def test_max_iter_returns_not_converged(self,trained_component):
        '''Method Function:

            Test explicit non-convergence and finite retained parameters at the BFGS iteration limit
        '''

        component = GHKMultinomialProbitComponent(num_draws=64,max_iter=1,seed=123)
        theta_hat,metrics = component.train(
            A=trained_component["A_train"],b=trained_component["b_train"]
        )

        assert not metrics["converged"]
        assert metrics["n_iter"] == 1
        assert np.isfinite(metrics["final_nll"])
        assert bool(jnp.all(jnp.isfinite(theta_hat)))


    def test_invalid_inputs(self,trained_component):
        '''Method Function:

            Test malformed, empty, non-finite, non-integer, out-of-range and dimension-incompatible inputs
        '''

        component = GHKMultinomialProbitComponent(num_draws=8,max_iter=1)
        with pytest.raises(ValueError,match="sample counts must match"):
            component.train(A=jnp.ones((4,2,1)),b=jnp.array([0,1]))
        with pytest.raises(ValueError,match="sample counts must match"):
            component.train(A=jnp.ones((4,2)),b=jnp.array([0,1,0,1]))
        with pytest.raises(ValueError,match="non-empty"):
            component.train(A=jnp.empty((0,2,1)),b=jnp.empty((0,)))
        with pytest.raises(ValueError,match="finite values"):
            component.train(
                A=jnp.array([[[jnp.nan],[1.0]],[[0.0],[1.0]]]),b=jnp.array([0,1])
            )
        with pytest.raises(ValueError,match="integer class labels"):
            component.train(A=jnp.ones((2,2,1)),b=jnp.array([0.0,0.5]))
        with pytest.raises(ValueError,match="class labels in the range"):
            component.train(A=jnp.ones((2,2,1)),b=jnp.array([0,3]))
        with pytest.raises(ValueError,match="same D and p dimensions"):
            trained_component["component"].predict_proba(A_test=jnp.ones((2,D_DIM + 1,P_DIM)))
        with pytest.raises(ValueError,match="non-empty"):
            trained_component["component"].predict_proba(A_test=jnp.empty((0,D_DIM,P_DIM)))


    def test_invalid_hyperparameters(self):
        '''Method Function:

            Test expected ValueError behavior for invalid simulation, optimization and normalization controls
        '''

        invalid_parameters = [
            {"num_draws": 0},
            {"num_draws": True},
            {"max_iter": 0},
            {"grad_tol": 0.0},
            {"grad_tol": float("nan")},
            {"seed": -1},
            {"inference_num_draws": 0},
            {"inference_seed": True},
            {"normalize": "yes"}
        ]
        for parameters in invalid_parameters:
            with pytest.raises(ValueError):
                GHKMultinomialProbitComponent(**parameters)


    def test_reasoning_before_train(self):
        '''Method Function:

            Test expected RuntimeError behavior for probability and class inference before training
        '''

        component = GHKMultinomialProbitComponent()
        A_test = jnp.ones((2,D_DIM,P_DIM),dtype=jnp.float64)
        with pytest.raises(RuntimeError,match="has not been trained yet"):
            component.reasoning(A_test=A_test)
        with pytest.raises(RuntimeError,match="has not been trained yet"):
            component.predict_proba(A_test=A_test)


    def test_info(self):
        '''Method Function:

            Test whether _info returns a non-empty string
        '''

        info = GHKMultinomialProbitComponent()._info()
        assert isinstance(info,str)
        assert len(info) > 0



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file
