class Task:
    def __init__(self, name):
        assert name.count(':') == name.count('-') == 1, \
            "Task name should be in <prefix>:<src1,...,srcN>-<trg1,...,trgN> format"

        self.name = name
        self.prefix, topology = self.name.split(':')
        srcs, trgs = topology.split('-')
        self.sources = srcs.split(',')
        self.targets = trgs.split(',')
        self.multi_source = len(self.sources) > 1
        self.multi_target = len(self.targets) > 1

    def __repr__(self):
        return f'Task({self.name}, srcs={self.sources!r}, trgs={self.targets!r}'
