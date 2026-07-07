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

This is a collection of postprocessing base classes

- Design mode:

    (1) nothing

- Key points:

    (1) nothing

- Main functions:

    (1) Post-processing

Usage examples
--------------
.. code-block:: python
    :linenos:


Class Description
-----------------
(1)BasePostprocessing: This is a basic class of algorithm modules. Its main functions standardize the unified interface method of specific algorithm modules, including three methods: training, inference, and direct operation

References
----------

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
### class:BasePostprocessing
### ------This is a basic class of algorithm modules. Its main functions standardize the unified interface method of specific algorithm modules, including three methods: training, inference, and direct operation
###
################################################################################################################################################################################



####### base class #################################################################################################################################################
####################################################################################################################################################################



class BaseRegression(ABC,metaclass=MetaRequestMethod):
    '''Class Introduction:

        This is a basic class of algorithm modules. Its main functions standardize the unified interface method of specific algorithm modules, including three methods: training, inference, and direct operation
    '''

    @abstractmethod
    def train(self):
        '''Method Function:

            Defines an abstract method for training methods

        :parameters:
            nothing

        :return:
            nothing
        '''

        pass


    @abstractmethod
    def reasoning(self):
        '''Method Function:

            Defines an abstract method of inference method

        :parameters:
            nothing

        :return:
            nothing
        '''

        pass


    def _info(self):
        '''Method Function:

            Define an internal method to obtain basic meta information

        :parameters:
            nothing

        :return:
            nothing
        ''' 

        pass

    

##############################################################################################################################################################################
##############################################################################################################################################################################


### End of file
