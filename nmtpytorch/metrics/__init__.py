from .metric import Metric
from .multibleu import BLEUScorer
from .sacrebleu import SACREBLEUScorer
from .sacrebleu_mem import SACREBLEU_MEMScorer
from .meteor import METEORScorer
from .wer import WERScorer
from .cer import CERScorer
from .rouge import ROUGEScorer

beam_metrics = ["BLEU", "SACREBLEU", "SACREBLEU_MEM", "METEOR", "WER", "CER", "ROUGE"]

metric_info = {
    'BLEU': 'max',
    'SACREBLEU': 'max',
    'SACREBLEU_MEM': 'max',
    'METEOR': 'max',
    'ROUGE': 'max',
    'LOSS': 'min',
    'WER': 'min',
    'CER': 'min',
    'ACC': 'max',
    'RECALL': 'max',
    'PRECISION': 'max',
    'F1': 'max',
}
