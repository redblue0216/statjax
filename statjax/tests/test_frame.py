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

This is a framework link smoke test script

- Design mode:

    (1) nothing

- Key points:

    (1) Script-style smoke test

- Main functions:

    (1) Verify the link of the TestRegression test placeholder component: train, reasoning, _info

Usage examples
--------------
.. code-block:: python
    :linenos:

    python statjax/tests/test_frame.py

Class Description
-----------------

References
----------

'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
### Algorithm package
### Project Package - Algorithm Components
from statjax.regression import TestRegression



####### Smoke Test #################################################################################
####################################################################################################



### Instantiate the test placeholder component
test_regression = TestRegression(n_steps_ahead=1)
### Verify the training method
train_result = test_regression.train()
print("Train Result:", train_result)
### Verify the inference method
reasoning_result = test_regression.reasoning()
print("Reasoning Result:", reasoning_result)
### Verify the meta information method
info = test_regression._info()
print("Info:", info)



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file
