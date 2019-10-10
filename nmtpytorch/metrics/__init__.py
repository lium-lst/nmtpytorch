from .metric import Metric
from .multibleu import BLEUScorer
from .sacrebleu import SACREBLEUScorer
from .meteor import METEORScorer
from .coco_meteor import COCOMETEORScorer
from .coco_cider import COCOCIDERScorer
from .wer import WERScorer
from .cer import CERScorer
from .rouge import ROUGEScorer

beam_metrics = ["BLEU", "SACREBLEU", "METEOR",
                "COCOMETEOR", "COCOCIDER", "WER", "CER", "ROUGE"]

metric_info = {
    'BLEU': 'max',
    'SACREBLEU': 'max',
    'METEOR': 'max',
    'COCOMETEOR': 'max',
    'COCOCIDER': 'max',
    'ROUGE': 'max',
    'LOSS': 'min',
    'WER': 'min',
    'CER': 'min',
    'ACC': 'max',
    'RECALL': 'max',
    'PRECISION': 'max',
    'F1': 'max',
}
