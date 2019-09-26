from .sat import ShowAttendAndTell
from .nli import NLI
from .nmt import NMT
from .mnmt import MultimodalNMT
from .acapt import AttentiveCaptioning
from .vcap import VideoCaptioner
from .vatex_cap import VATEXCaptioner

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

##########################################
# Backward-compatibility with older models
##########################################
ASRv2 = ASR
AttentiveMNMT = AttentiveRawMNMT
AttentiveEncAttMNMTFeatures = AttentiveMNMTFeaturesFA
