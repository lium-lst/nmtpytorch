from .metric import Metric
from .multibleu import BLEUScorer
from .sacrebleu import SACREBLEUScorer
from .meteor import METEORScorer
from .wer import WERScorer
from .cer import CERScorer
from .rouge import ROUGEScorer
from .coco import COCOMETEORScorer, COCOBLEUScorer, COCOROUGEScorer, COCOCIDERScorer

beam_metrics = ["BLEU", "SACREBLEU", "METEOR",
                "COCOMETEOR", "COCOCIDER", "COCOBLEU", "COCOROUGE",
                "WER", "CER", "ROUGE"]

metric_info = {
    'LOSS': 'min',
    'WER': 'min',
    'CER': 'min',
    'BLEU': 'max',
    'SACREBLEU': 'max',
    'COCOBLEU': 'max',
    'METEOR': 'max',
    'COCOMETEOR': 'max',
    'COCOCIDER': 'max',
    'ROUGE': 'max',
    'COCOROUGE': 'max',
    'ACC': 'max',
    'RECALL': 'max',
    'PRECISION': 'max',
    'F1': 'max',
}
