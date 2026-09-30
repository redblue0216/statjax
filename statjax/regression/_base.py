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

This is a collection of regression base classes

- Design mode:

    (1) Abstract base class mode

- Key points:

    (1) Metaprogramming technology init subclass

    (2) ABC abstract method

- Main functions:

    (1) Standardize the unified interface methods of regression algorithm modules, including three methods: training, inference, and meta information

Usage examples
--------------
.. code-block:: python
    :linenos:

    from statjax.regression._base import BaseRegression

    ### Define a regression component that inherits the regression base class and implements the agreed abstract methods
    class XXXXRegressionComponent(BaseRegression):

        def train(self):
            return "train implemented"

        def reasoning(self):
            return "reasoning implemented"

        def _info(self):
            return "component info"

Class Description
-----------------
(1)BaseRegression: This is a basic class of regression algorithm modules. Its main functions standardize the unified interface method of specific algorithm modules, including three methods: training, inference, and meta information

References
----------
StatJAX Design Document `"StatJAX Design SH V001"<https://github.com/redblue0216/statjax>`_
'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
from statjax.base import MetaRequestMethod
from abc import ABC,abstractmethod
### Algorithm package
### Project Package - Algorithm Components



####### Classes and Functions ###################################################################################################################################################
###
### class:BaseRegression
### ------This is a basic class of regression algorithm modules. Its main functions standardize the unified interface method of specific algorithm modules, including three methods: training, inference, and meta information
###
################################################################################################################################################################################



####### base class #################################################################################################################################################
####################################################################################################################################################################



class BaseRegression(ABC,metaclass=MetaRequestMethod):
    '''Class Introduction:

        This is a basic class of regression algorithm modules. Its main functions standardize the unified interface method of specific algorithm modules, including three methods: training, inference, and meta information. It uses the MetaRequestMethod metaclass to check whether the subclass implements all agreed abstract methods before the class is created
    '''

    @abstractmethod
    def train(self):
        '''Method Function:

            Defines an abstract method for training methods. The regression component subclass must implement the agreed training behavior, otherwise a NotImplementedError is raised when the class is created

        :parameters:
            nothing

        :return:
            nothing
        '''

        pass


    @abstractmethod
    def reasoning(self):
        '''Method Function:

            Defines an abstract method of inference method. The regression component subclass must implement the agreed inference behavior, otherwise a NotImplementedError is raised when the class is created

        :parameters:
            nothing

        :return:
            nothing
        '''

        pass


    def _info(self):
        '''Method Function:

            Define an internal method to obtain basic meta information, the main function is to return the text introduction of the algorithm module

        :parameters:
            nothing

        :return:
            nothing
        ''' 

        pass

    

##############################################################################################################################################################################
##############################################################################################################################################################################


### End of file
