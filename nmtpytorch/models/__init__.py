from .nmt import NMT
from .tfnmt import TransformerNMT
from .sat import ShowAttendAndTell
from .nli import NLI
from .mnmt import MultimodalNMT
from .acapt import AttentiveCaptioning
#from .vcap import VideoCaptioner
from .video_cap import VideoCap
#from .vatex_cap import VATEXCaptioner
#from .vatex_simple_cap import VATEXSimpleCaptioner

# Raw images
from .amnmtraw import AttentiveRawMNMT
# Spatial features + NMT
from .amnmtfeats import AttentiveMNMTFeatures
from .hamnmtfeats import HybridAttentiveMNMTFeatures
from .amnmtfeats_coling import AttentiveMNMTFeaturesColing
from .amnmtfeats_coling_masked import AttentiveMNMTFeaturesColingMasked
# Filtered attention (LIUMCVC-MMT2018)
from .amnmtfeats_fa import AttentiveMNMTFeaturesFA

# Speech models
from .asr import ASR
from .multimodal_asr import MultimodalASR

# Multi-label classifier
from .label_classifier import LabelClassifier

from .rationale import Rationale
from .rationalev2 import Rationalev2
from .rationalev3 import Rationalev3
from .rationalev4 import Rationalev4
from .embatt_rationale import EmbAttRationale

##########################################
# Backward-compatibility with older models
##########################################
ASRv2 = ASR
AttentiveMNMT = AttentiveRawMNMT
AttentiveEncAttMNMTFeatures = AttentiveMNMTFeaturesFA
