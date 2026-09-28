import pandas as pd

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.label = label
        self.children = {}

    def is_leaf(self):
        return self.attribute is None
