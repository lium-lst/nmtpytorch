# First the basic types
from .numpy import NumpyDataset
from .kaldi import KaldiDataset
from .imagefolder import ImageFolderDataset
from .text import TextDataset
from .onehot import OneHotDataset
from .numpy_sequence import NumpySequenceDataset
from .label import LabelDataset
from .shelve import ShelveDataset
from .msvd.mjson import MSVDJSONDataset
from .msvd.mshelve import MSVDShelveDataset

# Second the selector function
def get_dataset(type_):
    return {
        'numpy': NumpyDataset,
        'numpysequence': NumpySequenceDataset,
        'kaldi': KaldiDataset,
        'imagefolder': ImageFolderDataset,
        'text': TextDataset,
        'onehot': OneHotDataset,
        'label': LabelDataset,
        'shelve': ShelveDataset,
        'msvdjson': MSVDJSONDataset,
        'msvdshelve': MSVDShelveDataset,
    }[type_.lower()]


# Should always be at the end
from .multimodal import MultimodalDataset
