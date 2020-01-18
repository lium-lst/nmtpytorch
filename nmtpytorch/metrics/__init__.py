from .metric import Metric
from .multibleu import BLEUScorer
from .sacrebleu import SACREBLEUScorer
from .meteor import METEORScorer
from .wer import WERScorer
from .cer import CERScorer
from .rouge import ROUGEScorer
from .coco import COCOMETEORScorer, COCOBLEUScorer, COCOROUGEScorer, COCOCIDERScorer


metric_info = {
    # minimized metrics
    'loss': {'lr_decay_mode': 'min', 'beam_metric': False},
    'wer': {'lr_decay_mode': 'min', 'beam_metric': True},
    'cer': {'lr_decay_mode': 'min', 'beam_metric': True},
}

__maximized_metrics = [
    "bleu", "sacrebleu", "meteor", "rouge",
    "cocobleu", "cocometeor", "cococider", "cocorouge",
]

for metric in __maximized_metrics:
    metric_info[metric] = {'lr_decay_mode': 'max', 'beam_metric': True}
