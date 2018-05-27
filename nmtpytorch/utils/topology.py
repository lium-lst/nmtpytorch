# -*- coding: utf-8 -*-
from collections import UserString, OrderedDict

from .. import datasets


class DataSource(UserString):
    def __init__(self, name, _type, src=False, trg=False):
        super().__init__(name)
        self._type = _type
        self.src = src
        self.trg = trg

        # Assign the method that knows how to create a tensor for a batch
        # of this type
        klass = getattr(datasets, '{}Dataset'.format(_type))
        self.to_torch = klass.to_torch

    def __repr__(self):
        return "DataSource(name={}, type={}, src={}, trg={})".format(
            self.data, self._type, self.src, self.trg)


class Topology(object):
    """A simple object that parses the direction string provided through the
        experiment configuration file.

        A direction is a string with the following syntax:
            feat:<type>, feat:<type>, ... -> feat:<type>, feat:<type>, ...

        where
            feat determines the name of the modality, i.e. 'en', 'image', etc.
            type is the prefix of the actual ``Dataset`` class to be used
                with this modality, i.e. Text, ImageFolder, OneHot, etc.
            if type is omitted, the default is Text.

        Example:
            de:Text -> en:Text
            de:Text -> en:Text, en_pos:OneHot
            de:Text, image:ImageFolder -> en:Text
    """
    def __init__(self, direction):
        self.direction = direction
        self.srcs = OrderedDict()
        self.trgs = OrderedDict()
        self.all = OrderedDict()

        srcs, trgs = direction.split('->')

        # Temporary dict to parse sources and targets in a single loop
        tmp = {
            'srcs': srcs.strip().split(','),
            'trgs': trgs.strip().split(','),
        }

        for key, values in tmp.items():
            _dict = getattr(self, key)
            for val in values:
                name, *ftype = val.strip().split(':')
                ftype = ftype[0] if len(ftype) > 0 else "Text"
                ds = DataSource(name, ftype,
                                src=(key == 'srcs'), trg=(key == 'trgs'))
                if name in self.all:
                    raise RuntimeError(
                        '"{}" already given as a data source.'.format(name))
                _dict[name] = ds
                self.all[name] = ds

    def get_src_langs(self):
        langs = [v for v in self.srcs.values() if v._type == 'Text']
        return langs

    def get_trg_langs(self):
        langs = [v for v in self.trgs.values() if v._type == 'Text']
        return langs

    def __getitem__(self, key):
        return self.all[key]

    def __repr__(self):
        s = "Sources:\n"
        for x in self.srcs.values():
            s += " {}\n".format(x.__repr__())
        s += "Targets:\n"
        for x in self.trgs.values():
            s += " {}\n".format(x.__repr__())
        return s
