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

This is a collection of algorithm auxiliary utility classes

- Design mode:

    (1) Mixin mode

- Key points:

    (1) Mixin mode and static methods

- Main functions:

    (1) Algorithm auxiliary functions that can be called directly

Usage examples
--------------
.. code-block:: python
    :linenos:

    from statjax.utils import MixinUtils

    ### Static methods of the Mixin utility class can be called directly without instantiation
    MixinUtils.func_a()
    MixinUtils.func_b()

Class Description
-----------------
(1)MixinUtils: This is the concrete implementation of algorithm auxiliary functions, the main technical Mixin mode and static methods

References
----------
StatJAX Design Document `"StatJAX Design SH V001"<https://github.com/redblue0216/statjax>`_
'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
### Algorithm package
### Project Package - Algorithm Components



####### Classes and Functions #######################################################################
###
### class:MixinUtils
### ------This is the concrete implementation of algorithm auxiliary functions, the main technical Mixin mode and static methods
###
######################################################################################################



####### base class ######################################################################################################################################
#########################################################################################################################################################



class MixinUtils(object):
    '''Class Introduction:

        This is the concrete implementation of algorithm auxiliary functions, the main technical Mixin mode and static methods. As a Mixin auxiliary class, it can be mixed into various algorithm components to provide directly callable algorithm auxiliary functions
    '''


    @staticmethod
    def func_a():
        '''Method Function:

            Define an algorithm auxiliary function a, the main technical static method, which can be called directly without instantiation

        :parameters:
            nothing

        :return:
            nothing
        '''

        print("This is func_a from MixinUtils")

    @staticmethod
    def func_b():
        '''Method Function:

            Define an algorithm auxiliary function b, the main technical static method, which can be called directly without instantiation

        :parameters:
            nothing

        :return:
            nothing
        '''

        print("This is func_b from MixinUtils")



##############################################################################################################################################################################
##############################################################################################################################################################################


### End of file
