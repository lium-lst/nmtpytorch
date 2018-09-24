from .sat import ShowAttendAndTell
from .nmt import NMT
from .amnmt import AttentiveMNMT
from .amnmtfeats import AttentiveMNMTFeatures
from .mnmtdecinit import MNMTDecinit

# Speech models
from .asr import ASR
from .mmasr import MultimodalASR
from .mm_action_asr import MMActionASR

# Multi-task speech models
from .asr_nmt import ASRNMT
from .asr_nmt_shared import ASRNMTShared

# Multi-task SLT models
from .slt_asr_otm import SLTASROneToMany

from .multitask import Multitask
from .multitask_att import MultitaskAtt

##########################################
# Backward-compatibility with older models
##########################################
ASRv2 = ASR
