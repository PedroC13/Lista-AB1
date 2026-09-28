import pandas as pd
import numpy as np
from node import Node

class id3:
    def __init__(self, df: pd.DataFrame, verbose: bool = False) -> None:
        self.df = df
        self.verbose = verbose
        self.target = None
        self.root = None
        self.log = []

    def fit(self, target: str):
        self.target = target
        attributes = [c for c in self.df.columns if c != target]
        self.root = self.build_tree(self.df, attributes)
        return self.root

    def _record(self, text: str) -> None:
        if self.verbose:
            self.log.append(text)

    def _build_proof_box(self, node_label: str, df: pd.DataFrame, impurity: float, gains: dict, chosen: str) -> None:
        classes = df[self.target]
        counts = classes.value_counts()
        examples = [f"E{i + 1}" for i in df.index]

        lines = []
        lines.append(f"┌{'─' * 60}┐")
        lines.append(f"│ NÓ: {node_label}")
        lines.append(f"│ Exemplos: n={len(df)}  [{', '.join(examples)}]")
        lines.append(f"│ Classes: " + " | ".join(f"{c}={n}" for c, n in counts.items()))
        lines.append("│")
        lines.append(f"│ Entropia do conjunto: {impurity:.4f}")
        lines.append("│")
        lines.append("│ Ganho de Informação por atributo:")
        for attr, gain in gains.items():
            dots = "." * max(1, 30 - len(attr))
            lines.append(f"│   {attr} {dots} {gain:.4f}")
        lines.append("│")
        lines.append(f"│ ► Atributo escolhido: {chosen.upper()} (maior ganho)")
        lines.append(f"└{'─' * 60}┘")

        self._record("\n".join(lines))

    def predict(self, row, node: Node = None):
        if node is None:
            node = self.root

        if node.is_leaf():
            return node.label

        value = row[node.attribute]
        if value not in node.children:
            return self.df[self.target].mode()[0]

        return self.predict(row, node.children[value])

    def classify_dataset(self, df: pd.DataFrame = None, filepath: str = None) -> pd.DataFrame:
        """Classifica a base e retorna (real x predito), pronto para avaliação (ex: f-score)."""
        if df is None:
            df = self.df

        result = pd.DataFrame({
            "Real": df[self.target],
            "Predito": df.apply(self.predict, axis=1),
        })

        if filepath:
            result.to_csv(filepath, index=False)

        return result

    def calc_general_entropy(self, df: pd.DataFrame = None) -> float:
        if df is None:
            df = self.df

        values = df[self.target].value_counts()
        probability = values / values.sum()
        entropy = -sum(probability * np.log2(probability))

        self._record(f"Entropia do conjunto atual (n={len(df)}): {entropy:.4f}")

        return entropy

    def calc_attribute_entropy(self, attribute: str, df: pd.DataFrame = None) -> float:
        """Entropia média ponderada resultante de dividir o conjunto pelo atributo."""
        if df is None:
            df = self.df

        counting = df.groupby(attribute)[self.target].value_counts()
        total = len(df)

        self._record(f"--- \n Calculando entropia para o atributo: '{attribute}' ---")

        entropy_weighted_sum = 0

        for group in counting.index.get_level_values(0).unique():
            values = counting[group]
            quantity = values.sum()
            probability = values / quantity
            weight = quantity / total

            entropy = -sum(probability * np.log2(probability))
            entropy_weighted_sum += entropy * weight

            self._record(f"  Valor '{group}': n={quantity}, entropia={entropy:.4f}, peso={weight:.4f}")

        self._record(f"  => Entropia média ponderada de '{attribute}': {entropy_weighted_sum:.4f}")

        return entropy_weighted_sum

    def calc_entropy_gain(self, attribute: str, df: pd.DataFrame = None) -> float:
        if df is None:
            df = self.df

        general = self.calc_general_entropy(df)
        attr_entropy = self.calc_attribute_entropy(attribute, df)
        gain = general - attr_entropy

        self._record(f"  => Ganho de Informação de '{attribute}': {gain:.4f}\n")

        return gain

    def best_attribute(self, attributes: list, df: pd.DataFrame = None) -> tuple:
        if df is None:
            df = self.df

        gains = {attr: self.calc_entropy_gain(attr, df) for attr in attributes}

        self._record("Ganhos calculados:")
        for attr, gain in gains.items():
            self._record(f"  {attr}: {gain:.4f}")

        chosen = max(gains, key=gains.get)

        self._record(f"=> Atributo escolhido: '{chosen}'\n")

        return chosen, gains

    def build_tree(self, df: pd.DataFrame, attributes: list, depth: int = 0, origin: str = "Raiz") -> Node:
        classes = df[self.target]

        # Caso 1: todos os exemplos são da mesma classe -> folha
        if classes.nunique() == 1:
            self._record("  " * depth + f"[Folha] Classe única: {classes.iloc[0]}")
            return Node(label=classes.iloc[0])

        # Caso 2: não sobraram atributos -> folha com classe majoritária
        if len(attributes) == 0:
            majority_class = classes.mode()[0]
            self._record("  " * depth + f"[Folha] Sem atributos restantes. Classe majoritária: {majority_class}")
            return Node(label=majority_class)

        chosen, gains = self.best_attribute(attributes, df)
        node = Node(attribute=chosen)

        self._record("  " * depth + f"[Nó] Dividindo por: '{chosen}'")
        self._build_proof_box(origin, df, self.calc_general_entropy(df), gains, chosen)

        remaining_attributes = [a for a in attributes if a != chosen]

        for value in df[chosen].unique():
            subgroup = df[df[chosen] == value]

            if len(subgroup) == 0:
                majority_class = classes.mode()[0]
                node.children[value] = Node(label=majority_class)
            else:
                self._record("  " * (depth + 1) + f"-> Valor '{value}' (n={len(subgroup)})")
                new_origin = f"Ramo \"{value}\" → nível {depth + 1}"
                node.children[value] = self.build_tree(subgroup, remaining_attributes, depth + 1, new_origin)

        return node

    def _tree_text(self, node: Node = None, depth: int = 0, lines: list = None) -> list:
        is_root = lines is None
        if is_root:
            lines = []
        if node is None:
            node = self.root

        prefix = "  " * depth
        if node.is_leaf():
            lines.append(f"{prefix}[Folha] Classe: {node.label}")
        else:
            lines.append(f"{prefix}[Nó] Atributo: {node.attribute}")
            for value, child in node.children.items():
                lines.append(f"{prefix}  -> Valor '{value}':")
                self._tree_text(child, depth + 2, lines)

        return lines

    def export_tree(self, filepath: str) -> None:
        """Escreve toda a saída (log detalhado + árvore final) em um único arquivo de texto.

        Só gera o arquivo se verbose=True; sem verbose não há nada a exportar."""
        if not self.verbose:
            return

        content = []
        content.append("===== LOG DE CONSTRUÇÃO (ID3) =====\n")
        content.extend(self.log)
        content.append("\n===== ÁRVORE FINAL (ID3) =====\n")
        content.extend(self._tree_text())

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

    def export_pure_tree(self, filepath: str = None) -> str:
        """Gera apenas a árvore em texto (sem o log de construção).

        Útil como uma das 3 formas de saída/entrada do pipeline: (1) log completo
        via export_tree, (2) árvore pura via este método, (3) csv de predições
        via classify_dataset."""
        text = "\n".join(self._tree_text())

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

        return text
