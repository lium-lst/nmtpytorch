from .metric import Metric
from .multibleu import BLEUScorer
from .sacrebleu import SACREBLEUScorer
from .meteor import METEORScorer
from .wer import WERScorer
from .cer import CERScorer
from .rouge import ROUGEScorer

beam_metrics = ["BLEU", "SACREBLEU", "METEOR", "WER", "CER", "ROUGE"]

metric_info = {
    'BLEU': 'max',
    'SACREBLEU': 'max',
    'METEOR': 'max',
    'ROUGE': 'max',
    'LOSS': 'min',
    'WER': 'min',
    'CER': 'min',
    'ACC': 'max',
}
