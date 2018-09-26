from .sat import ShowAttendAndTell
from .nmt import NMT
from .amnmt import AttentiveMNMT
from .amnmtfeats import AttentiveMNMTFeatures
from .mnmtdecinit import MNMTDecinit

# Speech models
from .asr import ASR

# Experimental: requires work/adaptation
from .multitask import Multitask
from .multitask_att import MultitaskAtt

##########################################
# Backward-compatibility with older models
##########################################
ASRv2 = ASR
