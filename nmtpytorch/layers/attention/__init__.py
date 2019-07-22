from .mlp import MLPAttention
from .dot import DotAttention
from .hierarchical import HierarchicalAttention
from .co import CoAttention
from .mhco import MultiHeadCoAttention
from .uniform import UniformAttention


def get_attention(type_):
    return {
        'mlp': MLPAttention,
        'dot': DotAttention,
        'hier': HierarchicalAttention,
        'co': CoAttention,
        'mhco': MultiHeadCoAttention,
        'uniform': UniformAttention,
    }[type_]
