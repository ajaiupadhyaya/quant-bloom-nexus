"""
Utility modules and compatibility patches for Quant Bloom Nexus
"""

# Apply compatibility patches at import time
try:
    from .pyfolio_compat import patch_configparser
    patch_configparser()
except ImportError:
    pass

# Suppress warnings from deprecated packages
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='empyrical')
warnings.filterwarnings('ignore', category=UserWarning, module='zipline')
warnings.filterwarnings('ignore', category=UserWarning, module='backtrader')

# Fix pandas compatibility issues
import pandas as pd
if hasattr(pd, 'Panel'):
    # Remove deprecated Panel if it exists
    delattr(pd, 'Panel')

# Fix numpy compatibility
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float_
if not hasattr(np, 'complex'):
    np.complex = np.complex_ 