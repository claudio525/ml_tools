r"""
Library for machine learning tools.

Modules:
    - utils: General utility functions.
    - array_utils: Utility functions for arrays.
    - plotting: Plotting functions.
    - distance_calculation: Functions for calculating distances.
    - streamlit: Functions for working with streamlit.
    - dl_model_arch_utils: Functions for working with deep learning model architectures.
    - torch: Functions for working with PyTorch.
"""

from . import utils
from . import array_utils
from . import plotting
from . import distance_calculation
from . import dl_model_arch_utils
from . import streamlit
from . import quarto


optional_imports = []
try:
    from . import torch
    optional_imports.append("torch")
except ImportError:
    pass


__all__ = [
    "utils",
    "array_utils",
    "plotting",
    "distance_calculation",
    "streamlit",
    "dl_model_arch_utils",
    "quarto",
] + optional_imports