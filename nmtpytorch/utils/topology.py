# -*- coding: utf-8 -*-
from collections import UserString, OrderedDict


class DataSource(UserString):
    def __init__(self, name, type_):
        super().__init__(name)
        self._type = type_

    def __repr__(self):
        return "DataSource(name={}, type={})".format(self.data, self._type)


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

        srcs, trgs = direction.split('->')

        # Temporary dict to parse sources and targets in a single loop
        tmp = {
            'srcs': srcs.strip().split(','),
            'trgs': trgs.strip().split(','),
        }

        for key, values in tmp.items():
            for val in values:
                name, *ftype = val.strip().split(':')
                ftype = ftype[0] if len(ftype) > 0 else "Text"
                ds = DataSource(name, ftype)
                if name in self.__dict__:
                    raise RuntimeError(
                        '"{}" already given as a data source.'.format(name))
                setattr(self, name, ds)
                getattr(self, key)[name] = ds

    def get_src_langs(self):
        langs = [v for v in self.srcs.values() if v._type == 'Text']
        return langs

    def get_trg_langs(self):
        langs = [v for v in self.trgs.values() if v._type == 'Text']
        return langs

    def __repr__(self):
        s = "Sources:\n"
        for x in self.srcs.values():
            s += " {}\n".format(x.__repr__())
        s += "Targets:\n"
        for x in self.trgs.values():
            s += " {}\n".format(x.__repr__())
        return s
