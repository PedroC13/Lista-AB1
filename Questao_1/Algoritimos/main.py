import pandas as pd
from c45 import c45
from cart import cart
from id3 import id3

# Tratamento inicial dos dados
original_data = pd.read_csv("database.csv")

# removendo coluna desnecessária (é só um índice)
original_data = original_data.drop(columns=["Exemplo"])

original_data.to_csv("outputs/database_tratada.csv", index=False)

id3_tree = id3(original_data, verbose=True)
id3_tree.fit("Risco")
id3_tree.export_tree(filepath="outputs/arvore_id3.txt")
id3_tree.classify_dataset(filepath="outputs/predicoes_id3.csv")

c45_tree = c45(original_data, verbose=True)
c45_tree.fit("Risco")
c45_tree.export_tree(filepath="outputs/arvore_c45.txt")
c45_tree.classify_dataset(filepath="outputs/predicoes_c45.csv")

cart_tree = cart(original_data, verbose=True)
cart_tree.fit("Risco")
cart_tree.export_tree(filepath="outputs/arvore_cart.txt")
cart_tree.classify_dataset(filepath="outputs/predicoes_cart.csv")
