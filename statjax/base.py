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

This is a basic class collection of dynamic management methods

- Design mode:

    (1) Combination mode

- Key points:

    (1) Metaprogramming technology init subclass

- Main functions:

    (1) Dynamic loading method

Usage examples
--------------
.. code-block:: python
    :linenos:

Class Description
-----------------
(1)MetaRequestMethod: This is a dynamic management class method, the main function dynamic loading method, the main technical metaprogramming technology

References
----------
PythonORG `"Python datamodel"<https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__>`_
'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
from abc import ABC,ABCMeta,abstractmethod
import inspect
### Algorithm package
### Project Package - Algorithm Components



####### Classes and Functions #######################################################################
###
### class:MetaRequestMethod
### ------This is a dynamic management class method, the main function dynamic loading method, the main technical metaprogramming technology
###
######################################################################################################



####### base class ######################################################################################################################################
#########################################################################################################################################################



class MetaRequestMethod(ABCMeta):
    '''Class Introduction:

        This is a dynamic management class method, the main function dynamic loading method, the main technical metaprogramming technology
    ''' 


    def __init_subclass__(cls, *args, **kwargs):
        '''Method Function:

            Define a subclass initialization method, dynamic loading of main functional management class methods and object attribute inspection, main technical metaprogramming technology (getattr, hasattr, inspect.isfunction, inspect.signature)
        '''

        ### Get all abstract methods in the parent class
        abstract_methods = {name for name, method in inspect.getmembers(cls, inspect.isfunction) if getattr(method, '__isabstractmethod__', False)}
        ### Check whether the subclass implements all abstract methods
        for method_name in abstract_methods:
            if not hasattr(cls, method_name) or inspect.signature(getattr(cls, method_name)) == inspect.signature(lambda: None):
                raise NotImplementedError(f"Subclass must implement abstract method {method_name}")
        super().__init_subclass__(*args, **kwargs)



##############################################################################################################################################################################
##############################################################################################################################################################################


### End of file
