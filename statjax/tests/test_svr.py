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

This is a unit test collection of the SMOSVRComponent RBF support vector regression component

- Design mode:

    (1) nothing

- Key points:

    (1) pytest unit testing framework

    (2) Reproducible nonlinear regression data generated with explicit JAX PRNG keys

    (3) Independent correctness checks based on RBF kernel values, KKT bounds, exact pair updates and dual constraints

- Main functions:

    (1) Test the compact-dual second-order SMO training link

    (2) Test sparse kernel inference and target inverse standardization

    (3) Test input validation, non-convergence reporting, untrained inference and component metadata

Engineering flow closure
------------------------
Reproducible nonlinear data generation -> training-only standardization -> RBF kernel construction -> KKT working-set selection -> exact pair updates -> sparse kernel inference:

    (1) Verify kernel values against explicit NumPy loops and verify pair steps against independent dense one-dimensional search

    (2) Require KKT convergence, equality and box constraints, and agreement between incremental and recomputed gradients

    (3) Verify support-vector extraction, full/sparse prediction consistency, deterministic retraining and original-space predictions

    (4) Confirm safe constant-target and repeated-sample behavior and explicit non-convergence at the iteration limit

Verification criteria
---------------------
The independent RBF comparison uses float64 tolerances near machine precision. The exact pair step must match or improve upon an independent 240001-point feasible-grid search. Training must satisfy $nu<=tolerance$ (ν ≤ tolerance), $|sum(beta)|<1e-8$ (|Σβ| < 10⁻⁸), $-C<=beta_i<=C$ (−C ≤ βᵢ ≤ C) and incremental-gradient infinity error below 1e-8

Full and pruned predictions must agree within 2e-6 on the reference dataset; sparse formula checks use 1e-12. Constant-target and repeated-sample tests require finite states and valid constraints, even when an empty support set or zero curvature occurs. Held-out sine RMSE below 0.10 and a fourfold improvement over the mean baseline are end-to-end smoke criteria, not substitutes for KKT and dual-feasibility checks. Exact cross-backend iteration counts and bitwise identity are not required

Usage examples
--------------
.. code-block:: python
    :linenos:

    python -m pytest statjax/tests/test_svr.py -v

Class Description
-----------------
(1)TestSMOSVRComponent: This is a unit test class covering numerical correctness, public inference interfaces and error behavior

References
----------
pytestORG `"pytest documentation"<https://docs.pytest.org/>`_

StatJAX008 SVR Design Document `"StatJAX008 SVM SH V001"<https://github.com/redblue0216/statjax>`_
'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
import pytest
### Algorithm package
import jax
import jax.numpy as jnp
import numpy as np
### Project Package - Algorithm Components
from statjax.regression import SMOSVRComponent
from statjax.regression._components import _solve_svr_pair,_svr_kkt_bounds,_svr_rbf_kernel



####### Global Parameters ##########################################################################
####################################################################################################



### Enable float64 precision before any JAX array creation
jax.config.update("jax_enable_x64",True)

### Reproducible data dimensions and default SVR controls
N_TRAIN,N_TEST = 40,100
C,EPSILON,GAMMA,TOLERANCE = 10.0,0.05,0.5,1e-4
NOISE_SCALE = 0.05



####### Auxiliary Functions ########################################################################
####################################################################################################



def generate_svr_data(n_samples=N_TRAIN,noise=NOISE_SCALE,seed=42):
    '''Method Function:

        Generate reproducible noisy sine-regression data on a fixed one-dimensional grid

    :parameters:
        - n_samples (int) - Number of samples
        - noise (float) - Gaussian noise standard deviation
        - seed (int) - Pseudo-random seed

    :return:
        - A (jnp.ndarray) - (n_samples,1) Feature matrix
        - b (jnp.ndarray) - (n_samples,) Continuous targets
    '''

    key = jax.random.PRNGKey(seed)
    A = jnp.linspace(-3.0,3.0,n_samples,dtype=jnp.float64)[:,None]
    noise_values = jax.random.normal(key,(n_samples,),dtype=jnp.float64)
    b = jnp.sin(A[:,0]) + noise * noise_values

    return A,b



####### Test Fixtures ##############################################################################
####################################################################################################



@pytest.fixture(scope="module")
def trained_component():
    '''Method Function:

        Share one trained SVR component and its reproducible training data across test cases

    :parameters:
        nothing

    :return:
        - data (dict) - Trained component, training data, complete dual vector and metrics
    '''

    A_train,b_train = generate_svr_data()
    component = SMOSVRComponent(C=C,epsilon=EPSILON,gamma=GAMMA,tolerance=TOLERANCE)
    beta_hat,metrics = component.train(A=A_train,b=b_train)

    return {
        "component": component,
        "A_train": A_train,
        "b_train": b_train,
        "beta_hat": beta_hat,
        "metrics": metrics
    }



####### Test Classes ###############################################################################
####################################################################################################



class TestSMOSVRComponent:
    '''Class Introduction:

        This is a unit test class covering SVR numerical correctness, public inference and error behavior
    '''

    def test_rbf_kernel_matches_explicit_reference(self):
        '''Method Function:

            Test vectorized RBF values against an explicit NumPy double loop
        '''

        X = jnp.array([[0.0,1.0],[1.0,-1.0],[2.0,0.5]],dtype=jnp.float64)
        Z = jnp.array([[-0.5,0.2],[1.5,0.7]],dtype=jnp.float64)
        gamma = 0.7
        actual = _svr_rbf_kernel(X,Z,gamma)
        X_values,Z_values = np.asarray(X),np.asarray(Z)
        expected = np.empty((X.shape[0],Z.shape[0]),dtype=np.float64)
        for i in range(X.shape[0]):
            for j in range(Z.shape[0]):
                distance = np.sum((X_values[i] - Z_values[j]) ** 2)
                expected[i,j] = np.exp(-gamma * distance)

        np.testing.assert_allclose(np.asarray(actual),expected,rtol=1e-13,atol=1e-13)
        symmetric = _svr_rbf_kernel(X,X,gamma)
        np.testing.assert_allclose(np.asarray(symmetric),np.asarray(symmetric.T),rtol=0.0,atol=1e-15)
        np.testing.assert_allclose(np.asarray(jnp.diag(symmetric)),np.ones(X.shape[0]),rtol=0.0,atol=1e-15)


    def test_kkt_bounds_match_manual_example(self):
        '''Method Function:

            Test all active and inactive KKT bound branches against a hand-calculated example
        '''

        beta = jnp.array([0.0,1.0,-1.0,0.2],dtype=jnp.float64)
        residual = jnp.array([0.4,0.2,-0.3,0.1],dtype=jnp.float64)
        lower,upper = _svr_kkt_bounds(beta,-residual,1.0,0.1)
        expected_lower = np.array([0.3,-np.inf,-0.2,0.0])
        expected_upper = np.array([0.5,0.1,np.inf,0.0])

        np.testing.assert_allclose(np.asarray(lower),expected_lower,rtol=0.0,atol=1e-15)
        np.testing.assert_allclose(np.asarray(upper),expected_upper,rtol=0.0,atol=1e-15)
        assert float(jnp.max(lower) - jnp.min(upper)) > 0.0


    def test_pair_solution_matches_independent_grid(self):
        '''Method Function:

            Test the exact piecewise-quadratic pair solution against dense feasible-grid search
        '''

        K = jnp.array([[1.0,0.2],[0.2,1.0]],dtype=jnp.float64)
        beta = jnp.array([0.15,-0.1],dtype=jnp.float64)
        y = jnp.array([0.4,-0.3],dtype=jnp.float64)
        grad = K @ beta - y
        C_pair,epsilon_pair = 0.5,0.05
        step = float(_solve_svr_pair(K,beta,grad,C_pair,epsilon_pair,0,1))
        lower = max(-C_pair - float(beta[0]),float(beta[1]) - C_pair)
        upper = min(C_pair - float(beta[0]),float(beta[1]) + C_pair)
        grid = np.linspace(lower,upper,240001)
        beta_i,beta_j = float(beta[0]),float(beta[1])
        linear = float(grad[0] - grad[1])
        curvature = float(K[0,0] + K[1,1] - 2.0 * K[0,1])
        objective = (
            linear * grid + 0.5 * curvature * grid ** 2
            + epsilon_pair * (
                np.abs(beta_i + grid) - abs(beta_i)
                + np.abs(beta_j - grid) - abs(beta_j)
            )
        )
        actual_objective = (
            linear * step + 0.5 * curvature * step ** 2
            + epsilon_pair * (
                abs(beta_i + step) - abs(beta_i)
                + abs(beta_j - step) - abs(beta_j)
            )
        )

        assert lower <= step <= upper
        assert actual_objective <= float(np.min(objective)) + 1e-10
        assert actual_objective <= 1e-15


    def test_train_output_shape_and_state(self,trained_component):
        '''Method Function:

            Test complete dual shape, float64 precision and sparse inference state
        '''

        component = trained_component["component"]
        beta_hat = trained_component["beta_hat"]
        n_support = component.dual_coef.shape[0]

        assert beta_hat.shape == (N_TRAIN,)
        assert beta_hat.dtype == jnp.float64
        assert bool(jnp.all(jnp.isfinite(beta_hat)))
        assert component.support_vectors.shape == (n_support,1)
        assert component.support_indices.shape == (n_support,)
        assert component.support_indices.dtype == jnp.int32
        assert component.x_mean.shape == (1,) and component.x_scale.shape == (1,)
        assert bool(jnp.isfinite(component.intercept))
        assert trained_component["metrics"]["n_support"] == n_support


    def test_training_convergence_and_gradient(self,trained_component):
        '''Method Function:

            Test KKT convergence, tolerance-aware intercept bounds and incremental-gradient consistency
        '''

        component = trained_component["component"]
        metrics = trained_component["metrics"]
        A = trained_component["A_train"]
        X_standardized = (A - component.x_mean) / component.x_scale
        K = _svr_rbf_kernel(X_standardized,X_standardized,GAMMA)
        y_standardized = (trained_component["b_train"] - component.y_mean) / component.y_scale
        expected_grad = K @ component.beta_hat - y_standardized
        lower,upper = _svr_kkt_bounds(component.beta_hat,expected_grad,C,EPSILON)

        assert metrics["converged"]
        assert not metrics["stalled"]
        assert metrics["kkt_violation"] <= TOLERANCE
        assert 0 < metrics["n_iter"] <= metrics["max_iterations"]
        assert metrics["gradient_error"] < 1e-8
        ### At positive KKT tolerance the midpoint can differ from either bound by half the residual interval
        assert float(jnp.max(lower)) <= float(component.intercept) + 0.5 * TOLERANCE + 1e-10
        assert float(component.intercept) <= float(jnp.min(upper)) + 0.5 * TOLERANCE + 1e-10


    def test_equality_box_and_support_constraints(self,trained_component):
        '''Method Function:

            Test the compact-dual equality, box and support-extraction constraints
        '''

        component = trained_component["component"]
        beta = np.asarray(component.beta_hat)
        expected_indices = np.flatnonzero(np.abs(beta) > component.support_tolerance)

        assert abs(trained_component["metrics"]["equality_residual"]) < 1e-8
        assert np.all(beta >= -C - 1e-10) and np.all(beta <= C + 1e-10)
        np.testing.assert_array_equal(np.asarray(component.support_indices),expected_indices)
        np.testing.assert_allclose(
            np.asarray(component.dual_coef),beta[expected_indices],rtol=0.0,atol=0.0
        )


    def test_repeated_training_is_reproducible(self,trained_component):
        '''Method Function:

            Test deterministic dual variables, intercept and predictions for identical inputs
        '''

        repeated = SMOSVRComponent(C=C,epsilon=EPSILON,gamma=GAMMA,tolerance=TOLERANCE)
        beta_hat,_ = repeated.train(A=trained_component["A_train"],b=trained_component["b_train"])
        predictions = repeated.reasoning(A_test=trained_component["A_train"][:8])
        expected = trained_component["component"].reasoning(A_test=trained_component["A_train"][:8])

        np.testing.assert_allclose(
            np.asarray(beta_hat),np.asarray(trained_component["beta_hat"]),rtol=0.0,atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(repeated.intercept),np.asarray(trained_component["component"].intercept),rtol=0.0,atol=1e-12
        )
        np.testing.assert_allclose(np.asarray(predictions),np.asarray(expected),rtol=0.0,atol=1e-12)


    def test_full_and_sparse_predictions_agree(self,trained_component):
        '''Method Function:

            Test that support-vector pruning retains the full kernel-expansion prediction
        '''

        component = trained_component["component"]
        A = trained_component["A_train"]
        X_standardized = (A - component.x_mean) / component.x_scale
        K_full = _svr_rbf_kernel(X_standardized,X_standardized,GAMMA)
        full_prediction = (
            component.y_mean
            + component.y_scale * (K_full @ component.beta_hat + component.intercept)
        )
        sparse_prediction = component.reasoning(A_test=A)

        np.testing.assert_allclose(
            np.asarray(sparse_prediction),np.asarray(full_prediction),rtol=0.0,atol=2e-6
        )


    def test_reasoning_formula_and_inverse_standardization(self,trained_component):
        '''Method Function:

            Test output shape, float64 dtype and the explicit sparse prediction formula
        '''

        component = trained_component["component"]
        A_test = jnp.linspace(-2.5,2.5,17,dtype=jnp.float64)[:,None]
        predictions = component.reasoning(A_test=A_test)
        X_standardized = (A_test - component.x_mean) / component.x_scale
        K_test = _svr_rbf_kernel(X_standardized,component.support_vectors,GAMMA)
        expected = (
            component.y_mean
            + component.y_scale * (K_test @ component.dual_coef + component.intercept)
        )

        assert predictions.shape == (17,) and predictions.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(predictions),np.asarray(expected),rtol=0.0,atol=1e-12)


    def test_nonlinear_generalization(self,trained_component):
        '''Method Function:

            Test held-out sine predictions as an end-to-end smoke criterion
        '''

        A_test = jnp.linspace(-3.0,3.0,N_TEST,dtype=jnp.float64)[:,None]
        y_true = jnp.sin(A_test[:,0])
        predictions = trained_component["component"].reasoning(A_test=A_test)
        rmse = float(jnp.sqrt(jnp.mean((predictions - y_true) ** 2)))
        baseline = float(jnp.sqrt(jnp.mean((jnp.mean(trained_component["b_train"]) - y_true) ** 2)))

        assert rmse < 0.10
        assert rmse < 0.25 * baseline


    def test_constant_target_and_empty_support_set(self):
        '''Method Function:

            Test safe target scaling and zero-support prediction for a constant response
        '''

        A = jnp.linspace(-1.0,1.0,20,dtype=jnp.float64)[:,None]
        b = jnp.full((20,),3.25,dtype=jnp.float64)
        component = SMOSVRComponent()
        beta_hat,metrics = component.train(A=A,b=b)
        predictions = component.reasoning(A_test=A[:4])

        assert bool(jnp.all(beta_hat == 0.0))
        assert metrics["n_support"] == 0 and metrics["converged"]
        assert component.support_vectors.shape == (0,1)
        assert component.dual_coef.shape == (0,)
        assert float(component.y_scale) == 1e-12
        np.testing.assert_allclose(np.asarray(predictions),np.full(4,3.25),rtol=0.0,atol=1e-12)


    def test_repeated_samples_and_zero_curvature(self):
        '''Method Function:

            Test finite dual updates and predictions when duplicate points create zero curvature
        '''

        A = jnp.array([[0.0],[0.0],[1.0],[1.0]],dtype=jnp.float64)
        b = jnp.array([0.0,0.0,1.0,1.0],dtype=jnp.float64)
        component = SMOSVRComponent(max_iterations=2000)
        beta_hat,metrics = component.train(A=A,b=b)
        predictions = component.reasoning(A_test=A)

        assert bool(jnp.all(jnp.isfinite(beta_hat)))
        assert bool(jnp.all(jnp.isfinite(predictions)))
        assert metrics["gradient_error"] < 1e-8
        assert abs(metrics["equality_residual"]) < 1e-8
        assert np.all(np.asarray(beta_hat) >= -C - 1e-10)
        assert np.all(np.asarray(beta_hat) <= C + 1e-10)


    def test_max_iterations_returns_not_converged(self,trained_component):
        '''Method Function:

            Test explicit non-convergence and finite retained state at the iteration limit
        '''

        component = SMOSVRComponent(max_iterations=1,tolerance=1e-12)
        beta_hat,metrics = component.train(A=trained_component["A_train"],b=trained_component["b_train"])

        assert not metrics["converged"]
        assert metrics["n_iter"] == 1
        assert metrics["kkt_violation"] > 1e-12
        assert bool(jnp.all(jnp.isfinite(beta_hat)))


    def test_invalid_inputs(self,trained_component):
        '''Method Function:

            Test malformed, insufficient, non-finite and dimension-incompatible inputs
        '''

        component = SMOSVRComponent()
        with pytest.raises(ValueError,match="sample counts must match"):
            component.train(A=jnp.ones((4,2)),b=jnp.ones((2,)))
        with pytest.raises(ValueError,match="at least two samples"):
            component.train(A=jnp.ones((1,2)),b=jnp.ones((1,)))
        with pytest.raises(ValueError,match="at least two samples"):
            component.train(A=jnp.empty((2,0)),b=jnp.ones((2,)))
        with pytest.raises(ValueError,match="finite values"):
            component.train(A=jnp.array([[jnp.nan],[1.0]]),b=jnp.array([0.0,1.0]))
        with pytest.raises(ValueError,match="same feature dimension"):
            trained_component["component"].reasoning(A_test=jnp.ones((2,2)))
        with pytest.raises(ValueError,match="non-empty"):
            trained_component["component"].reasoning(A_test=jnp.empty((0,1)))
        with pytest.raises(ValueError,match="finite values"):
            trained_component["component"].reasoning(A_test=jnp.array([[jnp.inf]]))


    def test_invalid_hyperparameters(self):
        '''Method Function:

            Test expected ValueError behavior for invalid numerical and iteration controls
        '''

        invalid_parameters = [
            {"C": 0.0},{"C": -1.0},{"C": float("nan")},{"C": True},
            {"epsilon": -0.1},{"epsilon": float("inf")},
            {"gamma": 0.0},{"tolerance": 0.0},{"tolerance": float("nan")},
            {"max_iterations": 0},{"max_iterations": True},{"max_iterations": 1.5},
            {"support_tolerance": -1e-7},{"support_tolerance": float("inf")}
        ]
        for parameters in invalid_parameters:
            with pytest.raises(ValueError):
                SMOSVRComponent(**parameters)


    def test_reasoning_before_train(self):
        '''Method Function:

            Test expected RuntimeError behavior for inference before training
        '''

        component = SMOSVRComponent()
        with pytest.raises(RuntimeError,match="has not been trained yet"):
            component.reasoning(A_test=jnp.ones((2,1)))


    def test_info(self):
        '''Method Function:

            Test whether _info returns a non-empty string
        '''

        info = SMOSVRComponent()._info()
        assert isinstance(info,str)
        assert len(info) > 0



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file
