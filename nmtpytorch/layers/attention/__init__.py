from .mlp import MLPAttention
from .dot import DotAttention
from .hierarchical import HierarchicalAttention
from .co import CoAttention
from .mhco import MultiHeadCoAttention
from .uniform import UniformAttention
from .scaled_dot import ScaledDotAttention


def get_attention(type_):
    return {
        'mlp': MLPAttention,
        'dot': DotAttention,
        'hier': HierarchicalAttention,
        'co': CoAttention,
        'mhco': MultiHeadCoAttention,
        'uniform': UniformAttention,
        'scaled_dot': ScaledDotAttention,
    }[type_]
