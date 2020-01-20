class Task:
    """A class which represents a task.

    Arguments:
        name(str): The task name as defined in the configuration file.
        params(dict, optional): If not `None`, stores the task-relevant
            arguments as well.

    """
    def __init__(self, name, params=None):
        assert name.count(':') == name.count('-') == 1, \
            "Task name should be in <prefix>:<src1,...,srcN>-<trg1,...,trgN> format"

        self.name = name
        self.params = params
        self.prefix, topology = self.name.split(':')
        srcs, trgs = topology.split('-')
        self.sources = srcs.split(',')
        self.targets = trgs.split(',')
        self.multi_source = len(self.sources) > 1
        self.multi_target = len(self.targets) > 1

        self.src, self.trg = None, None

        if not self.multi_source:
            self.src = self.sources[0]

        if not self.multi_target:
            self.trg = self.targets[0]

    def __repr__(self):
        return f'Task({self.name}, srcs={self.sources!r}, trgs={self.targets!r})'
