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

This is a open class interface of the regression module

- Design mode:

    (1) nothing

- Key points:

    (1) __all__ explicit export

- Main functions:

    (1) open class

Usage examples
--------------
.. code-block:: python
    :linenos:

    from statjax.regression import HouseholderOLSComponent

    ols_instance = HouseholderOLSComponent()
    x_hat,metrics = ols_instance.train(A=A_train,b=b_train)
    y_pred = ols_instance.reasoning(A_test=A_test)

Class Description
-----------------
(1)TestRegression: This is a regression test placeholder component implementation class, the main function is to verify the framework link, the main technical inheritance

(2)HouseholderOLSComponent: This is an OLS linear regression component implementation class based on Householder QR decomposition, the main function is to solve the least squares optimal regression coefficients and batch prediction, the main technical JAX technology stack

References
----------

'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
### Algorithm package
### Project Package - Algorithm Components
from statjax.regression._components import TestRegression,HouseholderOLSComponent



####### Classes and Functions ###################################################################################################################################################
###
### open class list
### ------TestRegression
### ------HouseholderOLSComponent
###
################################################################################################################################################################################



####### open class list #################################################################################################################################################
#########################################################################################################################################################################



__all__ = [
    "TestRegression",
    "HouseholderOLSComponent"
]



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file
