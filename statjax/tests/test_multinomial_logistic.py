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

This is a unit test collection of the QRMultinomialLogisticComponent multinomial Logistic regression component

- Design mode:

    (1) nothing

- Key points:

    (1) pytest unit testing framework

    (2) Reproducible multiclass data generated with explicit JAX PRNG keys

    (3) Independent correctness checks based on automatic differentiation, explicit Newton systems and handwritten column-pivoted Householder QR

- Main functions:

    (1) Test reference-category Softmax, jointly coupled IRLS and QR training

    (2) Test K-class probability inference and argmax classification

    (3) Test input validation, convergence states, untrained inference and component metadata

Engineering flow closure
------------------------
Reproducible multiclass data generation -> training-only standardization -> reference-category Softmax -> per-sample Cholesky whitening -> augmented column-pivoted Householder QR -> Armijo backtracking -> probability and class inference:

    (1) Generate one population with $Z_true=X*W_true+b_true$ and $y~Categorical(Softmax(Z_true))$ (Z_true = XW_true+b_true, y ~ Categorical(Softmax(Z_true))), then split it with an independent key so train and test data share the same data-generating process

    (2) Verify $G_B=Phi^T*(P_1:q-Y_1:q)$ and $g=vec_row(G_B)+l2*penalty*theta$ (G_B = Φᵀ(P₁:q−Y₁:q), g = vec_row(G_B)+λ·penalty⊙θ) against jax.grad on identical standardized inputs

    (3) Construct a small explicit coupled system using $S_i=diag(p_i)-p_i*p_i^T$ and $C_i=phi_i^T kron I_q$ (Sᵢ = diag(pᵢ)−pᵢpᵢᵀ, Cᵢ = φᵢᵀ⊗I_q), then check the whitened QR candidate against the regularized Newton solution

    (4) Verify $A[:,permutation]=Q@R$ (A[:,permutation] = QR), Q.T@Q=I, numerical rank and least-squares residual on an exactly rank-deficient matrix

    (5) Require finite monotone loss histories and consistent accepted-update, rank and step counts; separately verify the max_iter non-converged state through public metrics

    (6) Reuse training preprocessing for K-class Softmax probabilities, require normalized rows and exact argmax labels, and preserve an explicitly declared class that is absent from training labels

Verification criteria
---------------------
Gradient absolute and relative errors must be below 1e-10; the explicit Newton candidate must agree within 1e-8; QR reconstruction and orthogonality errors must be below 1e-10; probability rows must be finite, bounded and normalized. Held-out accuracy above 0.50 is only an end-to-end smoke criterion, not a substitute for mathematical checks. Cross-backend bitwise identity, pivot-tie order and exact iteration counts are not required

Usage examples
--------------
.. code-block:: python
    :linenos:

    python -m pytest statjax/tests/test_multinomial_logistic.py -v

Class Description
-----------------
(1)TestQRMultinomialLogisticComponent: This is a unit test class covering numerical correctness, public inference interfaces and error behavior

References
----------
pytestORG `"pytest documentation"<https://docs.pytest.org/>`_

StatJAX006 Multinomial Logistic Design Document `"StatJAX006 Multinomial Logistic SH V001"<https://github.com/redblue0216/statjax>`_
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
from statjax.regression import QRMultinomialLogisticComponent
from statjax.regression._components import _build_multinomial_irls_system,_multinomial_column_pivoted_householder_qr,_multinomial_cpqr_solve,_multinomial_design_matrix,_multinomial_full_logits,_multinomial_matrix_gradient,_multinomial_objective,_multinomial_penalty_mask



####### Global Parameters ##########################################################################
####################################################################################################



### Enable float64 precision before any JAX array creation
jax.config.update("jax_enable_x64",True)

### Reproducible data dimensions and model structure
N_TRAIN,N_TEST,P_DIM,N_CLASSES = 240,60,4,3
L2 = 1e-4
GRADIENT_TOL = 1e-10
QR_RESIDUAL_TOL = 1e-10



####### Auxiliary Functions ########################################################################
####################################################################################################



def generate_multinomial_data(key,n,n_features=P_DIM,n_classes=N_CLASSES):
    '''Method Function:

        Define a reproducible multiclass generator with linear Softmax logits and categorical labels

    :parameters:
        - key (jax.Array) - Explicit JAX pseudo-random key
        - n (int) - Number of samples
        - n_features (int) - Number of input features
        - n_classes (int) - Total number of classes

    :return:
        - A (jnp.ndarray) - (n,n_features) Feature matrix
        - b (jnp.ndarray) - (n,) Integer class labels
    '''

    x_key,w_key,b_key,y_key = jax.random.split(key,4)
    A = jax.random.normal(x_key,(n,n_features),dtype=jnp.float64)
    true_w = 2.0 * jax.random.normal(w_key,(n_features,n_classes),dtype=jnp.float64) / jnp.sqrt(n_features)
    true_b = 0.35 * jax.random.normal(b_key,(n_classes,),dtype=jnp.float64)
    logits = A @ true_w + true_b
    b = jax.random.categorical(y_key,logits,axis=1).astype(jnp.int32)

    return A,b



####### Test Fixtures ##############################################################################
####################################################################################################



@pytest.fixture(scope="module")
def trained_component():
    '''Method Function:

        Define a module-level fixture sharing one trained component and independent train/test datasets

    :parameters:
        nothing

    :return:
        - data (dict) - Trained component, datasets, parameters and metrics
    '''

    A,b = generate_multinomial_data(jax.random.PRNGKey(42),N_TRAIN + N_TEST)
    permutation = jax.random.permutation(jax.random.PRNGKey(43),N_TRAIN + N_TEST)
    test_idx,train_idx = permutation[:N_TEST],permutation[N_TEST:]
    A_train,b_train = A[train_idx],b[train_idx]
    A_test,b_test = A[test_idx],b[test_idx]
    component = QRMultinomialLogisticComponent(n_classes=N_CLASSES,l2=L2)
    theta_hat,metrics = component.train(A=A_train,b=b_train)

    return {
        "component": component,
        "A_train": A_train,
        "b_train": b_train,
        "A_test": A_test,
        "b_test": b_test,
        "theta_hat": theta_hat,
        "metrics": metrics
    }



####### Test Classes ###############################################################################
####################################################################################################



class TestQRMultinomialLogisticComponent:
    '''Class Introduction:

        This is a unit test class covering multinomial numerical correctness, public inference and error behavior
    '''


    def test_train_output_shape(self,trained_component):
        '''Method Function:

            Test trained parameter shape, float64 precision and preprocessing state
        '''

        theta_hat = trained_component["theta_hat"]
        component = trained_component["component"]
        assert theta_hat.shape == ((P_DIM + 1) * (N_CLASSES - 1),)
        assert theta_hat.dtype == jnp.float64
        assert component.mean.shape == (P_DIM,)
        assert component.scale.shape == (P_DIM,)
        assert bool(jnp.all(jnp.isfinite(theta_hat)))


    def test_gradient_matches_autodiff(self,trained_component):
        '''Method Function:

            Test the analytic row-major matrix gradient against JAX automatic differentiation
        '''

        component = trained_component["component"]
        A = trained_component["A_train"][:32]
        b = trained_component["b_train"][:32]
        Phi = _multinomial_design_matrix(A,component.mean,component.scale)
        Y = jax.nn.one_hot(b,N_CLASSES,dtype=jnp.float64)
        penalty = _multinomial_penalty_mask(Phi.shape[1],N_CLASSES,Phi.dtype)
        theta = jnp.linspace(-0.2,0.3,Phi.shape[1] * (N_CLASSES - 1),dtype=jnp.float64)

        manual = _multinomial_matrix_gradient(theta,Phi,Y,penalty,L2,N_CLASSES)
        automatic = jax.grad(_multinomial_objective)(theta,Phi,Y,penalty,L2,N_CLASSES)
        absolute_error = jnp.max(jnp.abs(manual - automatic))
        relative_error = jnp.linalg.norm(manual - automatic) / (1.0 + jnp.linalg.norm(automatic))

        assert absolute_error < GRADIENT_TOL
        assert relative_error < GRADIENT_TOL


    def test_irls_system_matches_newton(self,trained_component):
        '''Method Function:

            Test the whitened IRLS candidate against an explicitly constructed coupled Newton system
        '''

        component = trained_component["component"]
        A = trained_component["A_train"][:6]
        b = trained_component["b_train"][:6]
        Phi = _multinomial_design_matrix(A,component.mean,component.scale)
        Y = jax.nn.one_hot(b,N_CLASSES,dtype=jnp.float64)
        q = N_CLASSES - 1
        dq = Phi.shape[1] * q
        penalty = _multinomial_penalty_mask(Phi.shape[1],N_CLASSES,Phi.dtype)
        theta = jnp.linspace(-0.15,0.2,dq,dtype=jnp.float64)
        l2 = 0.03
        jitter = 1e-10

        A_system,c_system = _build_multinomial_irls_system(
            theta,Phi,Y,penalty,N_CLASSES,l2,1e-12,jitter
        )
        candidate,rank = _multinomial_cpqr_solve(A_system,c_system)

        logits = _multinomial_full_logits(Phi,theta,N_CLASSES)
        probabilities = jax.nn.softmax(logits,axis=1)
        p = probabilities[:,:-1]
        eye_q = jnp.eye(q,dtype=Phi.dtype)
        S = -p[:,:,None] * p[:,None,:]
        S = S + eye_q[None,:,:] * p[:,:,None] + jitter * eye_q[None,:,:]
        base_design = jax.vmap(lambda x: jnp.kron(x[None,:],eye_q))(Phi)
        H_data = jnp.einsum("nqi,nqk,nkj->ij",base_design,S,base_design)
        data_gradient = (Phi.T @ (p - Y[:,:-1])).reshape(-1)
        expected = jnp.linalg.solve(
            H_data + l2 * jnp.diag(penalty),H_data @ theta - data_gradient
        )

        assert int(rank) == dq
        np.testing.assert_allclose(np.asarray(candidate),np.asarray(expected),rtol=1e-8,atol=1e-8)


    def test_pivoted_qr_rank_deficient(self):
        '''Method Function:

            Test QR reconstruction, orthogonality, rank truncation and least-squares residual on a rank-deficient matrix
        '''

        key_a,key_b = jax.random.split(jax.random.PRNGKey(7))
        A = jax.random.normal(key_a,(8,4),dtype=jnp.float64)
        A = A.at[:,3].set(A[:,0])
        b = jax.random.normal(key_b,(8,),dtype=jnp.float64)

        Q,R,permutation = _multinomial_column_pivoted_householder_qr(A)
        solution,rank = _multinomial_cpqr_solve(A,b)
        reconstruction_error = jnp.linalg.norm(A[:,permutation] - Q @ R) / jnp.linalg.norm(A)
        orthogonality_error = jnp.linalg.norm(Q.T @ Q - jnp.eye(4,dtype=A.dtype))
        residual = jnp.linalg.norm(A @ solution - b)
        reference = jnp.linalg.lstsq(A,b,rcond=None)[0]
        reference_residual = jnp.linalg.norm(A @ reference - b)

        assert reconstruction_error < QR_RESIDUAL_TOL
        assert orthogonality_error < QR_RESIDUAL_TOL
        assert int(rank) == 3
        assert jnp.abs(residual - reference_residual) < QR_RESIDUAL_TOL


    def test_loss_history_and_convergence(self,trained_component):
        '''Method Function:

            Test finite monotone loss, convergence state and consistent optimization histories
        '''

        metrics = trained_component["metrics"]
        losses = metrics["loss_history"]
        assert metrics["converged"]
        assert metrics["stop_reason"] == "converged"
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(losses))))
        assert all(later <= earlier + 1e-10 for earlier,later in zip(losses,losses[1:]))
        assert metrics["initial_loss"] == losses[0]
        assert metrics["final_loss"] == losses[-1]
        assert metrics["final_loss"] <= metrics["initial_loss"]
        assert metrics["n_iter"] == len(losses) - 1
        assert len(metrics["rank_history"]) == metrics["n_iter"]
        assert len(metrics["step_history"]) == metrics["n_iter"]
        assert metrics["gradient_check_error"] < GRADIENT_TOL


    def test_l2_excludes_intercept(self,trained_component):
        '''Method Function:

            Test that L2 changes only feature-weight gradients and excludes every free-category intercept
        '''

        component = trained_component["component"]
        A = trained_component["A_train"][:16]
        b = trained_component["b_train"][:16]
        Phi = _multinomial_design_matrix(A,component.mean,component.scale)
        Y = jax.nn.one_hot(b,N_CLASSES,dtype=jnp.float64)
        penalty = _multinomial_penalty_mask(Phi.shape[1],N_CLASSES,Phi.dtype)
        theta = jnp.linspace(-0.3,0.4,Phi.shape[1] * (N_CLASSES - 1),dtype=jnp.float64)

        unregularized = _multinomial_matrix_gradient(theta,Phi,Y,penalty,0.0,N_CLASSES)
        regularized = _multinomial_matrix_gradient(theta,Phi,Y,penalty,1.0,N_CLASSES)
        penalty_gradient = regularized - unregularized

        np.testing.assert_allclose(np.asarray(penalty_gradient),np.asarray(theta * penalty),rtol=0.0,atol=GRADIENT_TOL)
        assert float(jnp.max(jnp.abs(penalty_gradient[-(N_CLASSES - 1):]))) == 0.0


    def test_predict_proba(self,trained_component):
        '''Method Function:

            Test K-class probability shape, range, normalization and exact preprocessing reuse
        '''

        component = trained_component["component"]
        A_test = trained_component["A_test"]
        probabilities = component.predict_proba(A_test=A_test)
        Phi = _multinomial_design_matrix(A_test,component.mean,component.scale)
        expected = jax.nn.softmax(
            _multinomial_full_logits(Phi,component.theta_hat,N_CLASSES),axis=1
        )

        assert probabilities.shape == (N_TEST,N_CLASSES)
        assert bool(jnp.all(jnp.isfinite(probabilities)))
        assert bool(jnp.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        assert float(jnp.max(jnp.abs(jnp.sum(probabilities,axis=1) - 1.0))) < 1e-12
        assert float(jnp.max(jnp.abs(probabilities - expected))) < 1e-12


    def test_reasoning_predict(self,trained_component):
        '''Method Function:

            Test int32 argmax predictions and held-out classification accuracy
        '''

        component = trained_component["component"]
        A_test = trained_component["A_test"]
        b_test = trained_component["b_test"]
        probabilities = component.predict_proba(A_test=A_test)
        predictions = component.reasoning(A_test=A_test)
        accuracy = jnp.mean(predictions == b_test)

        assert predictions.shape == (N_TEST,)
        assert predictions.dtype == jnp.int32
        assert bool(jnp.array_equal(predictions,jnp.argmax(probabilities,axis=1).astype(jnp.int32)))
        assert accuracy > 0.50


    def test_explicit_classes_with_missing_training_class(self):
        '''Method Function:

            Test that explicit K preserves a probability column for a class absent from training labels
        '''

        A = jax.random.normal(jax.random.PRNGKey(51),(80,3),dtype=jnp.float64)
        b = jnp.where(A[:,0] > 0.0,2,0).astype(jnp.int32)
        component = QRMultinomialLogisticComponent(
            n_classes=3,l2=1e-3,max_iter=15,check_gradient=False
        )
        theta_hat,_ = component.train(A=A,b=b)
        probabilities = component.predict_proba(A_test=A[:5])

        assert not bool(jnp.any(b == 1))
        assert theta_hat.shape == (8,)
        assert probabilities.shape == (5,3)
        assert bool(jnp.all(jnp.isfinite(probabilities)))


    def test_binary_reference_category_special_case(self):
        '''Method Function:

            Test that K=2 reference-category Softmax reduces to the binary Sigmoid probability
        '''

        Phi = jnp.array(
            [[0.2,-0.1,1.0],[1.0,0.5,1.0],[-0.7,0.3,1.0],[0.4,-0.8,1.0]],
            dtype=jnp.float64
        )
        theta = jnp.linspace(-0.4,0.5,3,dtype=jnp.float64)
        probabilities = jax.nn.softmax(_multinomial_full_logits(Phi,theta,2),axis=1)
        eta = Phi @ theta.reshape((3,1))
        expected = jax.nn.sigmoid(eta[:,0])

        np.testing.assert_allclose(np.asarray(probabilities[:,0]),np.asarray(expected),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(np.asarray(probabilities[:,1]),np.asarray(1.0 - expected),rtol=1e-12,atol=1e-12)


    def test_extreme_logits_remain_stable(self):
        '''Method Function:

            Test finite logsumexp loss and normalized probabilities for extreme logits
        '''

        Phi = jnp.array([[1.0,0.0,1.0],[0.0,1.0,1.0]],dtype=jnp.float64)
        theta = jnp.array([1e6,-1e6,1e6,-1e6,0.0,0.0],dtype=jnp.float64)
        Y = jax.nn.one_hot(jnp.array([0,1],dtype=jnp.int32),3)
        penalty = _multinomial_penalty_mask(3,3,Phi.dtype)
        objective = _multinomial_objective(theta,Phi,Y,penalty,0.0,3)
        probabilities = jax.nn.softmax(_multinomial_full_logits(Phi,theta,3),axis=1)

        assert bool(jnp.isfinite(objective))
        assert bool(jnp.all(jnp.isfinite(probabilities)))
        assert float(jnp.max(jnp.abs(jnp.sum(probabilities,axis=1) - 1.0))) < 1e-12


    def test_constant_and_collinear_columns(self,trained_component):
        '''Method Function:

            Test safe preprocessing and regularized QR with constant and duplicate features
        '''

        A = trained_component["A_train"][:80]
        b = trained_component["b_train"][:80]
        A_augmented = jnp.concatenate(
            [A,jnp.ones((A.shape[0],1),dtype=A.dtype),A[:,:1]],axis=1
        )
        component = QRMultinomialLogisticComponent(
            n_classes=N_CLASSES,l2=L2,max_iter=20,check_gradient=False
        )
        theta_hat,metrics = component.train(A=A_augmented,b=b)
        probabilities = component.predict_proba(A_test=A_augmented[:10])

        assert theta_hat.shape == ((P_DIM + 3) * (N_CLASSES - 1),)
        assert bool(jnp.all(jnp.isfinite(theta_hat)))
        assert bool(jnp.isfinite(metrics["final_loss"]))
        assert probabilities.shape == (10,N_CLASSES)


    def test_max_iter_returns_not_converged(self,trained_component):
        '''Method Function:

            Test that reaching the iteration limit returns an explicit non-converged state
        '''

        component = QRMultinomialLogisticComponent(
            n_classes=N_CLASSES,max_iter=1,check_gradient=False
        )
        _,metrics = component.train(
            A=trained_component["A_train"],b=trained_component["b_train"]
        )

        assert not metrics["converged"]
        assert metrics["stop_reason"] == "max_iter"
        assert metrics["n_iter"] == 1
        assert len(metrics["loss_history"]) == 2
        assert metrics["gradient_check_error"] is None


    def test_invalid_inputs(self,trained_component):
        '''Method Function:

            Test expected ValueError behavior for malformed, non-finite and incompatible inputs
        '''

        component = QRMultinomialLogisticComponent(n_classes=3,check_gradient=False)
        with pytest.raises(ValueError,match="sample counts must match"):
            component.train(A=jnp.ones((4,2)),b=jnp.array([0,1]))
        with pytest.raises(ValueError,match="non-empty"):
            component.train(A=jnp.empty((0,2)),b=jnp.empty((0,)))
        with pytest.raises(ValueError,match="finite values"):
            component.train(A=jnp.array([[jnp.nan,1.0],[0.0,1.0]]),b=jnp.array([0,1]))
        with pytest.raises(ValueError,match="integer class labels"):
            component.train(A=jnp.ones((2,2)),b=jnp.array([0.0,0.5]))
        with pytest.raises(ValueError,match="class labels in the range"):
            component.train(A=jnp.ones((2,2)),b=jnp.array([0,3]))
        with pytest.raises(ValueError,match="same feature dimension"):
            trained_component["component"].predict_proba(A_test=jnp.ones((2,P_DIM + 1)))


    def test_invalid_hyperparameters(self):
        '''Method Function:

            Test expected ValueError behavior for invalid model and optimization hyperparameters
        '''

        invalid_parameters = [
            {"n_classes": 1},
            {"n_classes": True},
            {"n_classes": 3,"l2": -1.0},
            {"n_classes": 3,"max_iter": 0},
            {"n_classes": 3,"tolerance": 0.0},
            {"n_classes": 3,"rank_tol": 0.0},
            {"n_classes": 3,"probability_floor": 0.0},
            {"n_classes": 3,"weight_jitter": 0.0},
            {"n_classes": 3,"armijo_c1": 1.0},
            {"n_classes": 3,"max_backtracks": 0},
            {"n_classes": 3,"check_gradient": "yes"}
        ]
        for parameters in invalid_parameters:
            with pytest.raises(ValueError):
                QRMultinomialLogisticComponent(**parameters)


    def test_reasoning_before_train(self):
        '''Method Function:

            Test whether probability and class inference before training throw expected RuntimeError
        '''

        component = QRMultinomialLogisticComponent(n_classes=3)
        A_test = jnp.ones((N_TEST,P_DIM),dtype=jnp.float64)
        with pytest.raises(RuntimeError,match="has not been trained yet"):
            component.reasoning(A_test=A_test)
        with pytest.raises(RuntimeError,match="has not been trained yet"):
            component.predict_proba(A_test=A_test)


    def test_info(self):
        '''Method Function:

            Test whether the _info method returns a non-empty string
        '''

        info = QRMultinomialLogisticComponent(n_classes=3)._info()
        assert isinstance(info,str)
        assert len(info) > 0



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file
