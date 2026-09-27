import pandas as pd
from id3 import id3

# Tratamento incial dos dados 
original_data = pd.read_csv("base_credito_original.csv")
print(original_data.head())

#removendo coluna desnessesária exemplo
original_data = original_data.drop(columns=["Exemplo"])

original_data.to_csv("base_credito_tratada.csv", index = False)
