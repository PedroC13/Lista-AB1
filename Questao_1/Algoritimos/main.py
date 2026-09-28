import pandas as pd
from c45 import c45
from cart import cart
from id3 import id3

# Tratamento incial dos dados
original_data = pd.read_csv("database.csv")

#removendo coluna desnessesária exemplo
original_data = original_data.drop(columns=["Exemplo"])

original_data.to_csv("outputs/database_tratada.csv", index=False)

arvore_id3 = id3(original_data, verbose=True)
arvore_id3.fit("Risco")
arvore_id3.export_tree(filepath="outputs/arvore_id3.txt")
arvore_id3.classify_dataset(filepath="outputs/predicoes_id3.csv")

arvore_c45 = c45(original_data, verbose=True)
arvore_c45.fit("Risco")
arvore_c45.export_tree(filepath="outputs/arvore_c45.txt")
arvore_c45.classify_dataset(filepath="outputs/predicoes_c45.csv")

arvore_cart = cart(original_data, verbose=True)
arvore_cart.fit("Risco")
arvore_cart.export_tree(filepath="outputs/arvore_cart.txt")
arvore_cart.classify_dataset(filepath="outputs/predicoes_cart.csv")

