import pandas as pd

class Node:
    def __init__(self, atributo = None, classe = None):
        self.atributo = atributo
        self.classe = classe
        self.filhos = {}

    def is_leaf(self):
        return self.atributo is None
