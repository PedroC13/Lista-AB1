import pandas as pd

class Node:
    def __init__(self, atribute = None, classify = None):
        self.atributte = atribute
        self.classify = classify
        self.children = {}

    def is_leaf(self):
        return self.atributte is None
