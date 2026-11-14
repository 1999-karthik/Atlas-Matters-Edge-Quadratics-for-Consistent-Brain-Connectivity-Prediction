"""
EDGEQUADD: quadratic edge model for brain-connectivity classification.
"""
from .edgequadd_model import EDGEQUADD

from .quadratic_components import ASPPQuadraticAdapter
from .cnn_components import SimpleCNNCorrExtractor
from .cluster_pooling import DEC
from .basic_blocks import ResBlock, ASPP2d
from .normalization import RMSNorm, DropPath, _make_norm

__all__ = [
    'EDGEQUADD',
    'ASPPQuadraticAdapter', 
    'SimpleCNNCorrExtractor',
    'DEC',
    'ResBlock',
    'ASPP2d',
    'RMSNorm',
    'DropPath',
    '_make_norm'
]

from .utils import make_param_groups, split_quad_group

__all__ += ['make_param_groups', 'split_quad_group']