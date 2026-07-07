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
        return m.group(1) if m else '0.1.1'
    except Exception:
        pass
    try:
        return _im.version('statjax')
    except Exception:
        return '0.1.1'



__version__ = _read_version()



##############################################################################################################################################################################
##############################################################################################################################################################################



### End of file
