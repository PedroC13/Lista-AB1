import pandas as pd
import numpy as np
from node import Node

class c45:
    def __init__(self, df: pd.DataFrame, verbose: bool = False) -> None:
        self.df = df
        self.verbose = verbose
        self.target = None
        self.root = None
        self.log = []

    def fit(self, target: str):
        self.target = target
        atributos = [c for c in self.df.columns if c != target]
        self.root = self.build_tree(self.df, atributos)
        return self.root

    def _registrar(self, texto: str) -> None:
        if self.verbose:
            self.log.append(texto)

    def _montar_prova(self, node_label: str, df: pd.DataFrame, impureza: float, gains: dict, escolhido: str) -> None:
        classes = df[self.target]
        contagem = classes.value_counts()
        exemplos = [f"E{i + 1}" for i in df.index]

        linhas = []
        linhas.append(f"┌{'─' * 60}┐")
        linhas.append(f"│ NÓ: {node_label}")
        linhas.append(f"│ Exemplos: n={len(df)}  [{', '.join(exemplos)}]")
        linhas.append(f"│ Classes: " + " | ".join(f"{c}={n}" for c, n in contagem.items()))
        linhas.append("│")
        linhas.append(f"│ Entropia do conjunto: {impureza:.4f}")
        linhas.append("│")
        linhas.append("│ Ganho de Razão por atributo:")
        for attr, gain in gains.items():
            pontos = "." * max(1, 30 - len(attr))
            linhas.append(f"│   {attr} {pontos} {gain:.4f}")
        linhas.append("│")
        linhas.append(f"│ ► Atributo escolhido: {escolhido.upper()} (maior ganho)")
        linhas.append(f"└{'─' * 60}┘")

        self._registrar("\n".join(linhas))

    def predict(self, linha, no: Node = None):
        if no is None:
            no = self.root

        if no.is_leaf():
            return no.classe

        valor = linha[no.atributo]
        if valor not in no.filhos:
            return self.df[self.target].mode()[0]

        return self.predict(linha, no.filhos[valor])

    def classify_dataset(self, df: pd.DataFrame = None, filepath: str = None) -> pd.DataFrame:
        """Classifica a base e retorna (real x predito), pronto para avaliação (ex: f-score)."""
        if df is None:
            df = self.df

        resultado = pd.DataFrame({
            "Real": df[self.target],
            "Predito": df.apply(self.predict, axis=1),
        })

        if filepath:
            resultado.to_csv(filepath, index=False)

        return resultado

    def calc_general_entropy(self, df: pd.DataFrame = None) -> float:
        if df is None:
            df = self.df

        values = df[self.target].value_counts()
        probability = values / values.sum()
        entropy = -sum(probability * np.log2(probability))

        self._registrar(f"Entropia do conjunto atual (n={len(df)}): {entropy:.4f}")

        return entropy

    def calc_attribute_entropy(self, attribute: str, df: pd.DataFrame = None) -> float:
        """Entropia média ponderada resultante de dividir o conjunto pelo atributo."""
        if df is None:
            df = self.df

        counting = df.groupby(attribute)[self.target].value_counts()
        total = len(df)

        self._registrar(f"--- \n Calculando entropia para o atributo: '{attribute}' ---")

        entropy_weighted_sum = 0

        for group in counting.index.get_level_values(0).unique():
            values = counting[group]
            quantity = values.sum()
            probability = values / quantity
            weight = quantity / total

            entropy = -sum(probability * np.log2(probability))
            entropy_weighted_sum += entropy * weight

            self._registrar(f"  Valor '{group}': n={quantity}, entropia={entropy:.4f}, peso={weight:.4f}")

        self._registrar(f"  => Entropia média ponderada de '{attribute}': {entropy_weighted_sum:.4f}")

        return entropy_weighted_sum

    def calc_split_info(self, attribute: str, df: pd.DataFrame = None) -> float:
        """Entropia da distribuição dos exemplos entre os valores do atributo (usada para normalizar o ganho)."""
        if df is None:
            df = self.df

        values = df[attribute].value_counts()
        probability = values / values.sum()
        split_info = -sum(probability * np.log2(probability))

        return split_info

    def calc_gain_ratio(self, attribute: str, df: pd.DataFrame = None) -> float:
        if df is None:
            df = self.df

        general = self.calc_general_entropy(df)
        attr_entropy = self.calc_attribute_entropy(attribute, df)
        gain = general - attr_entropy

        split_info = self.calc_split_info(attribute, df)
        gain_ratio = gain / split_info if split_info != 0 else 0

        self._registrar(f"  => Ganho de '{attribute}': {gain:.4f}, Split Info: {split_info:.4f}, Ganho de Razão: {gain_ratio:.4f}\n")

        return gain_ratio

    def best_attribute(self, attributes: list, df: pd.DataFrame = None) -> tuple:
        if df is None:
            df = self.df

        gains = {attr: self.calc_gain_ratio(attr, df) for attr in attributes}

        self._registrar("Ganhos de Razão calculados:")
        for attr, gain in gains.items():
            self._registrar(f"  {attr}: {gain:.4f}")

        escolhido = max(gains, key=gains.get)

        self._registrar(f"=> Atributo escolhido: '{escolhido}'\n")

        return escolhido, gains

    def build_tree(self, df: pd.DataFrame, attributes: list, depth: int = 0, origem: str = "Raiz") -> Node:
        classes = df[self.target]

        # Caso 1: todos os exemplos são da mesma classe -> folha
        if classes.nunique() == 1:
            self._registrar("  " * depth + f"[Folha] Classe única: {classes.iloc[0]}")
            return Node(classe=classes.iloc[0])

        # Caso 2: não sobraram atributos -> folha com classe majoritária
        if len(attributes) == 0:
            majoritaria = classes.mode()[0]
            self._registrar("  " * depth + f"[Folha] Sem atributos restantes. Classe majoritária: {majoritaria}")
            return Node(classe=majoritaria)

        escolhido, gains = self.best_attribute(attributes, df)
        no = Node(atributo=escolhido)

        self._registrar("  " * depth + f"[Nó] Dividindo por: '{escolhido}'")
        self._montar_prova(origem, df, self.calc_general_entropy(df), gains, escolhido)

        atributos_restantes = [a for a in attributes if a != escolhido]

        for valor in df[escolhido].unique():
            subgrupo = df[df[escolhido] == valor]

            if len(subgrupo) == 0:
                majoritaria = classes.mode()[0]
                no.filhos[valor] = Node(classe=majoritaria)
            else:
                self._registrar("  " * (depth + 1) + f"-> Valor '{valor}' (n={len(subgrupo)})")
                novo_origem = f"Ramo \"{valor}\" → nível {depth + 1}"
                no.filhos[valor] = self.build_tree(subgrupo, atributos_restantes, depth + 1, novo_origem)

        return no

    def _arvore_texto(self, no: Node = None, depth: int = 0, linhas: list = None) -> list:
        raiz = linhas is None
        if raiz:
            linhas = []
        if no is None:
            no = self.root

        prefixo = "  " * depth
        if no.is_leaf():
            linhas.append(f"{prefixo}[Folha] Classe: {no.classe}")
        else:
            linhas.append(f"{prefixo}[Nó] Atributo: {no.atributo}")
            for valor, filho in no.filhos.items():
                linhas.append(f"{prefixo}  -> Valor '{valor}':")
                self._arvore_texto(filho, depth + 2, linhas)

        return linhas

    def export_tree(self, filepath: str) -> None:
        """Escreve toda a saída (log detalhado + árvore final) em um único arquivo de texto.

        Só gera o arquivo se verbose=True; sem verbose não há nada a exportar."""
        if not self.verbose:
            return

        conteudo = []
        conteudo.append("===== LOG DE CONSTRUÇÃO (C4.5) =====\n")
        conteudo.extend(self.log)
        conteudo.append("\n===== ÁRVORE FINAL (C4.5) =====\n")
        conteudo.extend(self._arvore_texto())

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(conteudo))
