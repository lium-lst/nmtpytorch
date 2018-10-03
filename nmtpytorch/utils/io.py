# -*- coding: utf-8 -*-
from collections import deque


class FileRotator:
    """A fixed queue with Path() elements where pushing a new element pops
    the oldest one and removes it from disk.

    Arguments:
        maxlen(int): The capacity of the queue.
    """

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.elems = deque(maxlen=self.maxlen)

    def push(self, elem):
        if len(self.elems) == self.maxlen:
            # Remove oldest item
            popped = self.elems.pop()
            if popped.exists():
                popped.unlink()

        # Add new item
        self.elems.appendleft(elem)

    def __repr__(self):
        return self.elems.__repr__()
