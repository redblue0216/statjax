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

This is the top-level package of StatJAX.

StatJAX is a statistical model algorithm package implemented bottom-up from the matrix computation level based on the JAX technology stack. The purposes of this open source project are as follows:

    (1) Practice makes perfect: deepen the understanding of statistical model algorithms by "reinventing the wheel"

    (2) Standing on the shoulders of giants: extend the implementation boundary of well-known statistical algorithm packages such as statsmodels and scikit-learn based on the JAX technology stack (such as GPU implementation)

    (3) Tossing a brick to attract jade: provide a learning reference for students of probability and statistics

- Design mode:

    (1) Combination mode

- Key points:

    (1) Metaprogramming technology init subclass

    (2) Three-layer architecture: Base / Component / Algo

- Main functions:

    (1) Standardize the method behavior of each module through the MetaRequestMethod metaclass, so that different algorithm engineers can jointly develop and research algorithms under the same standard

    (2) Concrete algorithms are implemented as independently openable Component algorithm components based on the JAX technology stack

Core concepts
-------------
(1) Base (module base class): Base is a module base class implemented with metaprogramming technology. Its main function is to standardize the method behavior of each module, so that different algorithm engineers can jointly develop and research algorithms under the same standard, and realize agile algorithm operation, maintenance and development.

(2) Component (algorithm component): Component is the concrete implementation of each algorithm, inheriting from the Base module base class (different algorithm base classes standardize different method behaviors). Its main function is to implement concrete algorithms. It is the main operation, maintenance and development part of algorithm engineers and can be opened independently.

(3) Algo (application component): Algo is an open component that combines various algorithm components based on Mixin. Its main function is to implement a complete algorithm application flow.

UML
---
.. code-block:: text

    classDiagram
        class MetaRequestMethod {
            <<metaclass>>
            This is a metaclass that controls class method coding, the main technical metaprogramming technology __init_subclass__
            +__init_subclass__()
            -Before the class is created, check whether the methods and attributes coded by the class meet the requirements
        }

        class BaseXXXX {
            <<abstract>>
            This is a module base class, the main function is to standardize the method behavior of each module, the main technical abstract method
            +@abstractmethod xxxx()
            -Agreed method behavior
            +_info()
            -Text introduction of the algorithm module
        }

        class XXXXComponent {
            This is the concrete implementation of an algorithm, the main technical JAX technology stack
            +xxxx()
            -Concrete implementation of the agreed method behavior
        }

        class MixinXXXX {
            This is the concrete implementation of algorithm auxiliary functions, the main technical Mixin mode and static methods
            +@staticmethod xxxx()
            -Algorithm auxiliary functions that can be called directly
        }

        BaseXXXX --|> MetaRequestMethod : Use metaclass
        XXXXComponent --|> BaseXXXX : Inherit
        XXXXComponent --> MixinXXXX : Mixin

Usage examples
--------------
.. code-block:: python
    :linenos:

    from statjax.regression import HouseholderOLSComponent

    ### Train the OLS model and solve the optimal regression coefficients
    ols_instance = HouseholderOLSComponent()
    x_hat,metrics = ols_instance.train(A=A_train,b=b_train)
    ### Batch prediction on the test set
    y_pred = ols_instance.reasoning(A_test=A_test)

Class Description
-----------------

References
----------
StatJAX Design Document `"StatJAX Design SH V001"<https://github.com/redblue0216/statjax>`_
'''



####### Load Packages ##############################################################################
####################################################################################################



### Basic package
import importlib.metadata as _im
from pathlib import Path



####### Version ####################################################################################
####################################################################################################



def _read_version() -> str:
    '''Method Function:

        Read version from pyproject.toml so development never goes stale.
    '''

    try:
        pyproject = Path(__file__).resolve().parent.parent / 'pyproject.toml'
        text = pyproject.read_text(encoding='utf-8')
        import re as _re
        m = _re.search(r'^version\s*=\s*"([^"]+)"', text, _re.M)
        return m.group(1) if m else '0.2.0'
    except Exception:
        pass
    try:
        return _im.version('statjax')
    except Exception:
        return '0.2.0'



__version__ = _read_version()



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file
