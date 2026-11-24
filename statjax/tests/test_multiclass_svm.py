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

This is a unit test collection of the SMOMulticlassSVMComponent exact RBF multiclass C-SVM component

- Design mode:

    (1) nothing

- Key points:

    (1) pytest unit testing framework

    (2) Reproducible three-class data and independent small binary-dual references

    (3) Numerical checks based on KKT sets, pair updates, dual constraints and bias recovery

- Main functions:

    (1) Test binary SMO and OvO/OvR multiclass training

    (2) Test sparse decision functions, voting and maximum-score classification

    (3) Test input validation, non-convergence reporting, untrained inference and metadata

Engineering flow closure
------------------------
Reproducible three-class data generation -> one RBF training kernel -> OvO/OvR binary tasks -> fixed-shape second-order SMO -> correctly signed bias recovery -> independent dual diagnostics -> support-vector compression -> OvO voting or OvR maximum-score inference:

    (1) Compare RBF values with explicit NumPy loops and compare one binary two-point dual with its analytic solution

    (2) Check every active and inactive KKT branch with a hand-calculated example and compare the clipped pair step with an independent 400001-point feasible-grid search

    (3) Require $y^T*alpha=0$ (yᵀα = 0), $0<=alpha_i<=C$ (0 ≤ αᵢ ≤ C), final KKT violation no greater than tolerance and independently recomputed gradient error below 1e-8

    (4) Verify $b=y-r$ (b = y−r) on a nonzero-intercept free-support example and $b=(m+M)/2$ (b = (m+M)/2) when no free support exists, protecting the decision-function sign in both branches

    (5) Compare complete and pruned kernel decisions, manually reconstruct OvO votes and OvR score matrices, and verify zero-score assignment and sorted-class tie handling

Verification criteria
---------------------
The independent kernel and two-point analytic-dual comparisons use float64 tolerances near machine precision. The pair step must match or improve upon the independent feasible-grid search. Per-task convergence requires m-M<=tolerance; aggregate convergence is true only when every binary task converges. equality_error, gradient_error and box_error are independently reviewed rather than inferred from class predictions

Complete and sparse binary decision values must agree within 1e-7 on the reference datasets. Reproducibility assertions apply to the same backend and inputs; cross-backend bitwise identity and exact iteration counts are not required. Held-out accuracy is only an end-to-end smoke criterion, not a substitute for dual feasibility, bias signs or KKT checks

Usage examples
--------------
.. code-block:: python
    :linenos:

    python -m pytest statjax/tests/test_multiclass_svm.py -v

Class Description
-----------------
(1)TestSMOMulticlassSVMComponent: This is a unit test class covering numerical correctness, public inference interfaces and error behavior

References
----------
pytestORG `"pytest documentation"<https://docs.pytest.org/>`_

StatJAX009 Multiclass SVM Design Document `"StatJAX009 Multiclass SVM SH V001"<https://github.com/redblue0216/statjax>`_
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
from statjax.regression import SMOMulticlassSVMComponent
from statjax.regression._components import _recover_svm_bias,_smo_binary_svm,_solve_svm_pair,_svm_rbf_kernel,_svm_working_sets



####### Global Parameters ##########################################################################
####################################################################################################



### Enable float64 precision before any JAX array creation
jax.config.update("jax_enable_x64",True)

### Reproducible data and solver controls
N_PER_CLASS,N_TRAIN_PER_CLASS = 20,15
C,GAMMA,TOLERANCE = 2.0,0.5,1e-3



####### Auxiliary Functions ########################################################################
####################################################################################################



def generate_multiclass_svm_data(seed=42,n_per_class=N_PER_CLASS,n_train_per_class=N_TRAIN_PER_CLASS):
    '''Method Function:

        Generate reproducible three-class Gaussian data with a stratified split

    :parameters:
        - seed (int) - Pseudo-random seed
        - n_per_class (int) - Total samples per class
        - n_train_per_class (int) - Training samples per class

    :return:
        - A_train (jnp.ndarray) - Training feature matrix
        - b_train (jnp.ndarray) - Training integer labels
        - A_test (jnp.ndarray) - Test feature matrix
        - b_test (jnp.ndarray) - Test integer labels
    '''

    rng = np.random.default_rng(seed)
    centers = np.array([[-2.0,-2.0],[2.0,-2.0],[0.0,2.2]])
    X_parts,y_parts = [],[]
    for class_value,center in enumerate(centers):
        X_parts.append(center + 0.75 * rng.standard_normal((n_per_class,2)))
        y_parts.append(np.full(n_per_class,class_value,dtype=np.int64))
    X,y = np.vstack(X_parts),np.concatenate(y_parts)
    train_indices,test_indices = [],[]
    for class_value in range(len(centers)):
        indices = np.flatnonzero(y == class_value)
        train_indices.extend(indices[:n_train_per_class])
        test_indices.extend(indices[n_train_per_class:])
    train_indices,test_indices = np.asarray(train_indices),np.asarray(test_indices)

    return (
        jnp.asarray(X[train_indices],dtype=jnp.float64),
        jnp.asarray(y[train_indices],dtype=jnp.int32),
        jnp.asarray(X[test_indices],dtype=jnp.float64),
        jnp.asarray(y[test_indices],dtype=jnp.int32)
    )



####### Test Fixtures ##############################################################################
####################################################################################################



@pytest.fixture(scope="module",params=["ovo","ovr"])
def trained_component(request):
    '''Method Function:

        Share one trained component for each multiclass strategy across numerical tests
    '''

    A_train,b_train,A_test,b_test = generate_multiclass_svm_data()
    component = SMOMulticlassSVMComponent(
        C=C,gamma=GAMMA,strategy=request.param,tolerance=TOLERANCE
    )
    alpha_hat,metrics = component.train(A=A_train,b=b_train)

    return {
        "component": component,
        "strategy": request.param,
        "A_train": A_train,
        "b_train": b_train,
        "A_test": A_test,
        "b_test": b_test,
        "alpha_hat": alpha_hat,
        "metrics": metrics
    }



####### Test Classes ###############################################################################
####################################################################################################



class TestSMOMulticlassSVMComponent:
    '''Class Introduction:

        This is a unit test class covering multiclass C-SVM numerical correctness and public behavior
    '''

    def test_rbf_kernel_matches_explicit_reference(self):
        '''Method Function:

            Test vectorized RBF values against an explicit NumPy double loop
        '''

        X = jnp.array([[0.0,1.0],[1.0,-1.0],[2.0,0.5]],dtype=jnp.float64)
        Z = jnp.array([[-0.5,0.2],[1.5,0.7]],dtype=jnp.float64)
        actual = _svm_rbf_kernel(X,Z,0.7)
        X_values,Z_values = np.asarray(X),np.asarray(Z)
        expected = np.empty((3,2),dtype=np.float64)
        for i in range(3):
            for j in range(2):
                expected[i,j] = np.exp(-0.7 * np.sum((X_values[i] - Z_values[j]) ** 2))
        np.testing.assert_allclose(np.asarray(actual),expected,rtol=1e-13,atol=1e-13)
        symmetric = _svm_rbf_kernel(X,X,0.7)
        np.testing.assert_allclose(np.asarray(symmetric),np.asarray(symmetric.T),rtol=0.0,atol=1e-15)
        np.testing.assert_allclose(np.asarray(jnp.diag(symmetric)),np.ones(3),rtol=0.0,atol=1e-15)


    def test_working_sets_match_manual_example(self):
        '''Method Function:

            Test all KKT working-set branches against a hand-calculated example
        '''

        alpha = jnp.array([0.0,0.5,1.0,0.0],dtype=jnp.float64)
        y = jnp.array([1.0,-1.0,1.0,-1.0],dtype=jnp.float64)
        grad = jnp.array([-0.2,-0.4,-0.3,-0.1],dtype=jnp.float64)
        p,p_up,p_low = _svm_working_sets(alpha,grad,y,1.0)
        np.testing.assert_allclose(np.asarray(p),np.array([0.2,-0.4,0.3,-0.1]),rtol=0.0,atol=1e-15)
        np.testing.assert_allclose(np.asarray(p_up),np.array([0.2,-0.4,-np.inf,-np.inf]),rtol=0.0,atol=1e-15)
        np.testing.assert_allclose(np.asarray(p_low),np.array([np.inf,-0.4,0.3,-0.1]),rtol=0.0,atol=1e-15)
        assert float(jnp.max(p_up) - jnp.min(p_low)) > 0.0


    def test_pair_solution_matches_independent_grid(self):
        '''Method Function:

            Test the analytic pair step against an independent feasible-grid search
        '''

        K = jnp.array([[1.0,0.2],[0.2,1.0]],dtype=jnp.float64)
        y = jnp.array([1.0,-1.0],dtype=jnp.float64)
        alpha = jnp.array([0.2,0.3],dtype=jnp.float64)
        grad = y * (K @ (y * alpha)) - 1.0
        C_pair = 0.8
        step = float(_solve_svm_pair(K,y,alpha,grad,C_pair,1e-12,0,1))
        grid = np.linspace(-2.0,2.0,400001)
        new_i = float(alpha[0]) + float(y[0]) * grid
        new_j = float(alpha[1]) - float(y[1]) * grid
        feasible = (new_i >= 0.0) & (new_i <= C_pair) & (new_j >= 0.0) & (new_j <= C_pair)
        grid = grid[feasible]
        curvature = float(K[0,0] + K[1,1] - 2.0 * K[0,1])
        linear = float(y[0] * grad[0] - y[1] * grad[1])
        objective = linear * grid + 0.5 * curvature * grid ** 2
        actual_objective = linear * step + 0.5 * curvature * step ** 2
        assert float(alpha[0]) + float(y[0]) * step >= -1e-12
        assert float(alpha[0]) + float(y[0]) * step <= C_pair + 1e-12
        assert float(alpha[1]) - float(y[1]) * step >= -1e-12
        assert float(alpha[1]) - float(y[1]) * step <= C_pair + 1e-12
        assert actual_objective <= float(np.min(objective)) + 1e-10


    def test_binary_solver_matches_two_point_analytic_dual(self):
        '''Method Function:

            Test a two-point binary dual against its analytic equality-constrained solution
        '''

        K = jnp.array([[1.0,0.25],[0.25,1.0]],dtype=jnp.float64)
        y = jnp.array([1.0,-1.0],dtype=jnp.float64)
        state = _smo_binary_svm(K,y,10.0,1e-10,1000,1e-12)
        expected_alpha = 1.0 / 0.75
        np.testing.assert_allclose(np.asarray(state.alpha),np.full(2,expected_alpha),rtol=1e-9,atol=1e-9)
        assert abs(float(y @ state.alpha)) < 1e-10
        assert float(state.m - state.M) <= 1e-10


    def test_free_support_bias_has_decision_function_sign(self):
        '''Method Function:

            Test b=y-r on a manual nonzero-intercept free-support example
        '''

        K = np.eye(2)
        y = np.array([1.0,-1.0])
        alpha = np.array([0.7,1.3])
        raw = K @ (alpha * y)
        grad = y * raw - 1.0
        bias = _recover_svm_bias(K,y,alpha,grad,2.0)
        assert bias == pytest.approx(0.3,abs=1e-15)
        np.testing.assert_allclose(y * (raw + bias),np.ones(2),rtol=0.0,atol=1e-15)


    def test_bounded_support_bias_uses_positive_kkt_midpoint(self):
        '''Method Function:

            Test b=(m+M)/2 when no free support vector exists
        '''

        K = np.eye(2)
        y = np.array([1.0,-1.0])
        alpha = np.zeros(2)
        grad = np.array([-0.8,0.2])
        bias = _recover_svm_bias(K,y,alpha,grad,1.0)
        assert bias == pytest.approx(0.5,abs=1e-15)


    def test_train_output_and_state(self,trained_component):
        '''Method Function:

            Test concatenated dual offsets, binary tasks and sparse model state
        '''

        component = trained_component["component"]
        strategy = trained_component["strategy"]
        expected_models = 3
        samples_per_model = 2 * N_TRAIN_PER_CLASS if strategy == "ovo" else 3 * N_TRAIN_PER_CLASS
        assert trained_component["alpha_hat"].shape == (expected_models * samples_per_model,)
        assert trained_component["alpha_hat"].dtype == jnp.float64
        assert bool(jnp.all(jnp.isfinite(trained_component["alpha_hat"])))
        assert component.dual_offsets.shape == (expected_models + 1,)
        assert int(component.dual_offsets[-1]) == expected_models * samples_per_model
        assert len(component.binary_models) == expected_models
        np.testing.assert_array_equal(np.asarray(component.classes),np.array([0,1,2]))
        for model in component.binary_models:
            n_support = model["dual_coef"].shape[0]
            assert model["sample_indices"].shape == (samples_per_model,)
            assert model["support_vectors"].shape == (n_support,2)
            assert model["support_indices"].shape == (n_support,)
            assert bool(jnp.isfinite(model["intercept"]))


    def test_convergence_constraints_and_gradient(self,trained_component):
        '''Method Function:

            Test KKT convergence, equality/box constraints and independent gradient recomputation
        '''

        component = trained_component["component"]
        metrics = trained_component["metrics"]
        assert metrics["converged"]
        assert metrics["max_kkt_violation"] <= TOLERANCE
        assert metrics["max_equality_error"] < 1e-8
        assert metrics["max_gradient_error"] < 1e-8
        assert metrics["max_box_error"] < 1e-10
        for index,(model,diagnostic) in enumerate(zip(component.binary_models,metrics["diagnostics"])):
            start,stop = int(component.dual_offsets[index]),int(component.dual_offsets[index + 1])
            alpha = np.asarray(trained_component["alpha_hat"][start:stop])
            indices = np.asarray(model["sample_indices"])
            labels = np.asarray(trained_component["b_train"])[indices]
            y_binary = np.where(labels == model["class_a"],1.0,-1.0)
            K_job = np.asarray(_svm_rbf_kernel(
                trained_component["A_train"][indices],trained_component["A_train"][indices],GAMMA
            ))
            grad_exact = y_binary * (K_job @ (y_binary * alpha)) - 1.0
            assert np.all(np.isfinite(grad_exact))
            assert diagnostic["converged"]
            assert diagnostic["kkt_violation"] <= TOLERANCE
            assert 0 < diagnostic["n_iter"] <= component.max_iterations
            assert abs(y_binary @ alpha) < 1e-8
            assert np.all(alpha >= -1e-10) and np.all(alpha <= C + 1e-10)
            assert diagnostic["gradient_error"] < 1e-8


    def test_sparse_decision_values_agree_with_full_dual(self,trained_component):
        '''Method Function:

            Test support-vector pruning against the complete binary kernel expansion
        '''

        component = trained_component["component"]
        A_test = trained_component["A_test"]
        for index,model in enumerate(component.binary_models):
            start,stop = int(component.dual_offsets[index]),int(component.dual_offsets[index + 1])
            alpha = trained_component["alpha_hat"][start:stop]
            indices = np.asarray(model["sample_indices"])
            labels = np.asarray(trained_component["b_train"])[indices]
            y_binary = jnp.where(labels == model["class_a"],1.0,-1.0)
            K_full = _svm_rbf_kernel(A_test,trained_component["A_train"][indices],GAMMA)
            full_score = K_full @ (alpha * y_binary) + model["intercept"]
            K_sparse = _svm_rbf_kernel(A_test,model["support_vectors"],GAMMA)
            sparse_score = K_sparse @ model["dual_coef"] + model["intercept"]
            np.testing.assert_allclose(np.asarray(sparse_score),np.asarray(full_score),rtol=0.0,atol=1e-7)


    def test_reasoning_matches_manual_combination(self,trained_component):
        '''Method Function:

            Test public predictions against manually assembled OvO votes or OvR scores
        '''

        component = trained_component["component"]
        A_test = trained_component["A_test"]
        scores = []
        for model in component.binary_models:
            K_test = _svm_rbf_kernel(A_test,model["support_vectors"],GAMMA)
            scores.append(np.asarray(K_test @ model["dual_coef"] + model["intercept"]))
        classes = np.asarray(component.classes)
        if trained_component["strategy"] == "ovr":
            expected = classes[np.argmax(np.column_stack(scores),axis=1)]
        else:
            votes = np.zeros((A_test.shape[0],len(classes)),dtype=np.int64)
            class_to_column = {int(value): index for index,value in enumerate(classes)}
            for model,score in zip(component.binary_models,scores):
                votes[score >= 0,class_to_column[model["class_a"]]] += 1
                votes[score < 0,class_to_column[model["class_b"]]] += 1
            expected = classes[np.argmax(votes,axis=1)]
        predictions = component.reasoning(A_test=A_test)
        assert predictions.shape == (A_test.shape[0],) and predictions.dtype == jnp.int32
        np.testing.assert_array_equal(np.asarray(predictions),expected)


    def test_multiclass_generalization(self,trained_component):
        '''Method Function:

            Test held-out classification accuracy as an end-to-end smoke criterion
        '''

        predictions = trained_component["component"].reasoning(A_test=trained_component["A_test"])
        accuracy = jnp.mean(predictions == trained_component["b_test"])
        assert accuracy >= 0.85


    def test_repeated_training_is_reproducible(self,trained_component):
        '''Method Function:

            Test deterministic dual variables and predictions for identical inputs
        '''

        repeated = SMOMulticlassSVMComponent(
            C=C,gamma=GAMMA,strategy=trained_component["strategy"],tolerance=TOLERANCE
        )
        alpha_hat,_ = repeated.train(A=trained_component["A_train"],b=trained_component["b_train"])
        predictions = repeated.reasoning(A_test=trained_component["A_test"])
        np.testing.assert_allclose(
            np.asarray(alpha_hat),np.asarray(trained_component["alpha_hat"]),rtol=0.0,atol=1e-12
        )
        np.testing.assert_array_equal(
            np.asarray(predictions),
            np.asarray(trained_component["component"].reasoning(A_test=trained_component["A_test"]))
        )


    def test_binary_special_case_task_counts(self):
        '''Method Function:

            Test that two classes produce one OvO task and two OvR tasks
        '''

        A_train,b_train,A_test,b_test = generate_multiclass_svm_data()
        train_mask = np.asarray(b_train) < 2
        test_mask = np.asarray(b_test) < 2
        components = [
            SMOMulticlassSVMComponent(C=C,gamma=GAMMA,strategy=strategy)
            for strategy in ("ovo","ovr")
        ]
        results = [component.train(A=A_train[train_mask],b=b_train[train_mask]) for component in components]
        assert len(components[0].binary_models) == 1
        assert len(components[1].binary_models) == 2
        assert results[0][0].shape == (2 * N_TRAIN_PER_CLASS,)
        assert results[1][0].shape == (4 * N_TRAIN_PER_CLASS,)
        for component in components:
            predictions = component.reasoning(A_test=A_test[test_mask])
            assert predictions.dtype == jnp.int32
            assert jnp.mean(predictions == b_test[test_mask]) >= 0.85


    def test_ovo_tie_and_zero_score_rules(self):
        '''Method Function:

            Test deterministic tie order and assignment of zero scores to class_a
        '''

        component = SMOMulticlassSVMComponent(strategy="ovo")
        component.alpha_hat = jnp.zeros(1,dtype=jnp.float64)
        component.classes = jnp.array([0,1,2],dtype=jnp.int32)
        component.n_features = 1
        component.binary_models = []
        for a,b,intercept in ((0,1,0.0),(0,2,-1.0),(1,2,0.0)):
            component.binary_models.append({
                "class_a": a,
                "class_b": b,
                "support_vectors": jnp.empty((0,1),dtype=jnp.float64),
                "dual_coef": jnp.empty((0,),dtype=jnp.float64),
                "intercept": jnp.asarray(intercept,dtype=jnp.float64)
            })
        predictions = component.reasoning(A_test=jnp.array([[0.0],[1.0]],dtype=jnp.float64))
        np.testing.assert_array_equal(np.asarray(predictions),np.array([0,0]))


    def test_ovr_tie_uses_sorted_class_order(self):
        '''Method Function:

            Test that equal OvR decision values select the first sorted class
        '''

        component = SMOMulticlassSVMComponent(strategy="ovr")
        component.alpha_hat = jnp.zeros(1,dtype=jnp.float64)
        component.classes = jnp.array([0,1,2],dtype=jnp.int32)
        component.n_features = 1
        component.binary_models = []
        for class_value in range(3):
            component.binary_models.append({
                "class_a": class_value,
                "class_b": None,
                "support_vectors": jnp.empty((0,1),dtype=jnp.float64),
                "dual_coef": jnp.empty((0,),dtype=jnp.float64),
                "intercept": jnp.asarray(0.0,dtype=jnp.float64)
            })
        predictions = component.reasoning(A_test=jnp.array([[0.0],[1.0]],dtype=jnp.float64))
        np.testing.assert_array_equal(np.asarray(predictions),np.array([0,0]))


    def test_repeated_samples_and_zero_curvature(self):
        '''Method Function:

            Test finite duals, biases and predictions with duplicate feature vectors
        '''

        A = jnp.array([[0.0],[0.0],[1.0],[1.0],[2.0],[2.0]],dtype=jnp.float64)
        b = jnp.array([0,0,1,1,2,2],dtype=jnp.int32)
        component = SMOMulticlassSVMComponent(max_iterations=2000)
        alpha_hat,metrics = component.train(A=A,b=b)
        predictions = component.reasoning(A_test=A)
        assert bool(jnp.all(jnp.isfinite(alpha_hat)))
        assert bool(jnp.all(jnp.isfinite(predictions)))
        assert metrics["max_equality_error"] < 1e-8
        assert metrics["max_gradient_error"] < 1e-8
        assert metrics["max_box_error"] < 1e-10


    def test_max_iterations_returns_not_converged(self):
        '''Method Function:

            Test explicit aggregate and per-binary non-convergence at the iteration limit
        '''

        A_train,b_train,_,_ = generate_multiclass_svm_data()
        component = SMOMulticlassSVMComponent(max_iterations=1,tolerance=1e-12)
        alpha_hat,metrics = component.train(A=A_train,b=b_train)
        assert not metrics["converged"]
        assert all(not item["converged"] for item in metrics["diagnostics"])
        assert all(item["n_iter"] == 1 for item in metrics["diagnostics"])
        assert bool(jnp.all(jnp.isfinite(alpha_hat)))


    def test_invalid_inputs(self,trained_component):
        '''Method Function:

            Test malformed, non-finite, non-integer, single-class and incompatible inputs
        '''

        component = SMOMulticlassSVMComponent()
        with pytest.raises(ValueError,match="sample counts must match"):
            component.train(A=jnp.ones((4,2)),b=jnp.array([0,1]))
        with pytest.raises(ValueError,match="at least two samples"):
            component.train(A=jnp.empty((0,2)),b=jnp.empty((0,)))
        with pytest.raises(ValueError,match="finite values"):
            component.train(A=jnp.array([[jnp.nan],[1.0]]),b=jnp.array([0,1]))
        with pytest.raises(ValueError,match="integer class labels"):
            component.train(A=jnp.ones((2,1)),b=jnp.array([0.0,0.5]))
        with pytest.raises(ValueError,match="at least two observed classes"):
            component.train(A=jnp.ones((2,1)),b=jnp.array([0,0]))
        with pytest.raises(ValueError,match="same feature dimension"):
            trained_component["component"].reasoning(A_test=jnp.ones((2,3)))
        with pytest.raises(ValueError,match="non-empty"):
            trained_component["component"].reasoning(A_test=jnp.empty((0,2)))
        with pytest.raises(ValueError,match="finite values"):
            trained_component["component"].reasoning(A_test=jnp.array([[jnp.inf,0.0]]))


    def test_invalid_hyperparameters(self):
        '''Method Function:

            Test expected ValueError behavior for invalid numerical and strategy controls
        '''

        invalid_parameters = [
            {"C": 0.0},{"C": True},{"gamma": 0.0},{"gamma": float("nan")},
            {"strategy": "invalid"},{"tolerance": 0.0},
            {"max_iterations": 0},{"max_iterations": True},{"max_iterations": 1.5},
            {"curvature_tolerance": 0.0},{"support_tolerance": -1e-10},
            {"support_tolerance": float("inf")}
        ]
        for parameters in invalid_parameters:
            with pytest.raises(ValueError):
                SMOMulticlassSVMComponent(**parameters)


    def test_reasoning_before_train(self):
        '''Method Function:

            Test expected RuntimeError behavior for inference before training
        '''

        component = SMOMulticlassSVMComponent()
        with pytest.raises(RuntimeError,match="has not been trained yet"):
            component.reasoning(A_test=jnp.ones((2,1)))


    def test_info(self):
        '''Method Function:

            Test whether _info returns a non-empty string
        '''

        info = SMOMulticlassSVMComponent()._info()
        assert isinstance(info,str)
        assert len(info) > 0



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file
