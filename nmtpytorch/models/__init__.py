from .sat import ShowAttendAndTell
from .nmt import NMT

# Raw images
from .amnmtraw import AttentiveRawMNMT
# Spatial features + NMT
from .amnmtfeats import AttentiveMNMTFeatures

# Speech models
from .asr import ASR

# Experimental: requires work/adaptation
from .multitask import Multitask
from .multitask_att import MultitaskAtt

##########################################
# Backward-compatibility with older models
##########################################
ASRv2 = ASR
AttentiveMNMT = AttentiveRawMNMT
