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

This is the public class interface of the regression module. It exposes independently usable Components while keeping implementation details in _components.py and interface constraints in BaseRegression

- Design mode:

    (1) Explicit public-interface mode

- Key points:

    (1) __all__ explicit export

    (2) Unified train / reasoning / _info component interfaces, with algorithm-specific extensions such as Logistic predict_proba

- Main functions:

    (1) Open OLS, ridge, LASSO and binary Logistic components

Usage examples
--------------
.. code-block:: python
    :linenos:

    from statjax.regression import HouseholderOLSComponent,SVDRidgeComponent,CDLassoComponent,QRLogisticComponent

    ols_instance = HouseholderOLSComponent()
    x_hat,metrics = ols_instance.train(A=A_train,b=b_train)
    y_pred = ols_instance.reasoning(A_test=A_test)

    ridge_instance = SVDRidgeComponent(lam=1.0)
    w_hat,metrics = ridge_instance.train(A=A_train,b=b_train)
    y_pred = ridge_instance.reasoning(A_test=A_test)

    lasso_instance = CDLassoComponent(lam=0.1)
    w_hat,metrics = lasso_instance.train(A=A_train,b=b_train)
    y_pred = lasso_instance.reasoning(A_test=A_test)

    ### Binary Logistic labels must contain only 0 and 1
    logistic_instance = QRLogisticComponent(l2=1e-6)
    beta_hat,metrics = logistic_instance.train(A=A_binary_train,b=b_binary_train)
    probabilities = logistic_instance.predict_proba(A_test=A_binary_test)
    y_pred = logistic_instance.reasoning(A_test=A_binary_test)

Class Description
-----------------
(1)TestRegression: This is a regression test placeholder component implementation class, the main function is to verify the framework link, the main technical inheritance

(2)HouseholderOLSComponent: This is an OLS linear regression component implementation class based on Householder QR decomposition, the main function is to solve the least squares optimal regression coefficients and batch prediction, the main technical JAX technology stack

(3)SVDRidgeComponent: This is a ridge regression component implementation class based on SVD decomposition, the main function is to solve the L2 regularized optimal regression coefficients and batch prediction, the main technical JAX technology stack

(4)CDLassoComponent: This is a LASSO regression component implementation class based on coordinate descent, the main function is to solve the L1 regularized sparse optimal regression coefficients and batch prediction, the main technical JAX technology stack

(5)QRLogisticComponent: This is a binary Logistic regression component based on IRLS and column-pivoted Householder QR decomposition, with numerical-rank truncation, Armijo backtracking, probability inference and threshold-based classification

References
----------
StatJAX Design Document `"StatJAX Design SH V001"<https://github.com/redblue0216/statjax>`_

StatJAX004 Logistic Design Document `"StatJAX004 Logistic SH V001"<https://github.com/redblue0216/statjax>`_

'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
### Algorithm package
### Project Package - Algorithm Components
from statjax.regression._components import TestRegression,HouseholderOLSComponent,SVDRidgeComponent,CDLassoComponent,QRLogisticComponent



####### Classes and Functions ###################################################################################################################################################
###
### open class list
### ------TestRegression
### ------HouseholderOLSComponent
### ------SVDRidgeComponent
### ------CDLassoComponent
### ------QRLogisticComponent
###
################################################################################################################################################################################



####### open class list #################################################################################################################################################
#########################################################################################################################################################################



__all__ = [
    "TestRegression",
    "HouseholderOLSComponent",
    "SVDRidgeComponent",
    "CDLassoComponent",
    "QRLogisticComponent"
]



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file
