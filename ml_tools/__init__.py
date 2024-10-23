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

try:
    from . import torch
except ImportError:
    print(
        f"Failed to import the torch module. This is likely because PyTorch is not installed."
    )

try:
    from . import quarto
except ImportError:
    print(
        f"Failed to import the quarto module. This is likely because quarto is not installed."
    )

try:
    from . import streamlit
except ImportError:
    print(
        f"Failed to import the streamlit module. This is likely because streamlit is not installed."
    )
