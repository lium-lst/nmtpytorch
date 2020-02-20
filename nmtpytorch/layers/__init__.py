try:
    import apex
    LayerNorm = apex.normalization.FusedLayerNorm
except ImportError as ie:
    import torch
    LayerNorm = torch.nn.LayerNorm

# Basic layers
from .ff import FF
from .fusion import Fusion
from .flatten import Flatten
from .argselect import ArgSelect
from .pool import Pool
from .seq_conv import SequenceConvolution
from .rnninit import RNNInitializer
from .max_margin import MaxMargin

# Embedding variants
from .embedding import *

# Attention layers
from .attention import *

# Encoder layers
from .encoders import *

# Decoder layers
from .decoders import *
