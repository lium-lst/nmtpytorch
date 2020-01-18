import copy
import json

from . import datasets
from .logger import Logger
from .vocabulary import Vocabulary


log = Logger()


class DataStore:
    """A holistic object which holds references to all `Dataset` instances
    along with the associated :class:`Vocabulary` defined in the configuration
    file.

    Arguments:
        opts(Config): An :class:`Config` instance as parsed from the
            configuration file.

    Attributes:
        data(dict): A copy of the `data` section from the configuration file
            with data definitions replaced with actual `torch.nn.Dataset`
            instances.
        vocabulary(dict): A copy of the `vocabulary` section from the configuration
            file with vocabulary definitions replaced with actual `Vocabulary`
            instances.

    """
    def __init__(self, opts):
        self._cache = {}
        self.data = copy.deepcopy(opts.sections['data'])
        self.vocabulary = copy.deepcopy(opts.sections['vocabulary'])

        for dset, spec in self.vocabulary.items():
            for key, kwargs in spec.items():
                self.vocabulary[dset][key] = Vocabulary(**kwargs)

        for name, spec in self.data.items():
            for split, dsets in spec.items():
                for key, kwargs in dsets.items():
                    ds_hash = str(kwargs)
                    if ds_hash not in self._cache:
                        klass = kwargs.pop('type')
                        if kwargs.get('vocab', False):
                            # Replace with the actual vocabulary
                            kwargs['vocab'] = self.vocabulary[name][key]
                        ds = getattr(datasets, klass)(**kwargs)
                        self._cache[ds_hash] = ds
                    self.data[name][split][key] = self._cache[ds_hash]
                    log.log(self.data[name][split][key])

    def __repr__(self):
        return json.dumps(
            {'data': self. data, 'vocabulary': self.vocabulary},
            indent=1, default=lambda obj: str(obj))
