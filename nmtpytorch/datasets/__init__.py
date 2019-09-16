# First the basic types
from .numpy import NumpyDataset
from .keyed_npz import KeyedNPZDataset
from .kaldi import KaldiDataset
from .imagefolder import ImageFolderDataset
from .text import TextDataset
from .onehot import OneHotDataset
from .numpy_sequence import NumpySequenceDataset
from .label import LabelDataset
from .shelve import ShelveDataset
from .msvd.mjson import MSVDJSONDataset
from .msvd.numpy import MSVDNumpyDataset
from .vatex_json import VatexJSONDataset
from .coco_json import COCOJSONDataset

# Second the selector function
def get_dataset(type_):
    return {
        'numpy': NumpyDataset,
        'keyednpz': KeyedNPZDataset,
        'numpysequence': NumpySequenceDataset,
        'kaldi': KaldiDataset,
        'imagefolder': ImageFolderDataset,
        'text': TextDataset,
        'onehot': OneHotDataset,
        'label': LabelDataset,
        'shelve': ShelveDataset,
        'msvdjson': MSVDJSONDataset,
        'msvdnumpy': MSVDNumpyDataset,
        'vatexjson': VatexJSONDataset,
        'cocojson': COCOJSONDataset,
    }[type_.lower()]


# Should always be at the end
from .multimodal import MultimodalDataset
from .multitask import MultitaskDataset
