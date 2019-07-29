from .sat import ShowAttendAndTell
from .nli import NLI
from .nmt import NMT
from .mnmt import MultimodalNMT
from .acapt import AttentiveCaptioning
from .vcap import VideoCaptioner

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

# Experimental: requires work/adaptation
from .multitask import Multitask
from .multitask_att import MultitaskAtt

##########################################
# Backward-compatibility with older models
##########################################
ASRv2 = ASR
AttentiveMNMT = AttentiveRawMNMT
AttentiveEncAttMNMTFeatures = AttentiveMNMTFeaturesFA
