import pandas as pd
from node import Node

class cart:
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
        linhas.append(f"│ Gini do conjunto: {impureza:.4f}")
        linhas.append("│")
        linhas.append("│ Ganho de Gini por atributo:")
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

    def calc_gini(self, df: pd.DataFrame = None) -> float:
        if df is None:
            df = self.df

        values = df[self.target].value_counts()
        probability = values / values.sum()
        gini = 1 - sum(probability ** 2)

        self._registrar(f"Gini do conjunto atual (n={len(df)}): {gini:.4f}")

        return gini

    def calc_attribute_gini(self, attribute: str, df: pd.DataFrame = None) -> float:
        """Gini médio ponderado resultante de dividir o conjunto pelo atributo."""
        if df is None:
            df = self.df

        counting = df.groupby(attribute)[self.target].value_counts()
        total = len(df)

        self._registrar(f"--- \n Calculando Gini para o atributo: '{attribute}' ---")

        gini_weighted_sum = 0

        for group in counting.index.get_level_values(0).unique():
            values = counting[group]
            quantity = values.sum()
            probability = values / quantity
            weight = quantity / total

            gini = 1 - sum(probability ** 2)
            gini_weighted_sum += gini * weight

            self._registrar(f"  Valor '{group}': n={quantity}, gini={gini:.4f}, peso={weight:.4f}")

        self._registrar(f"  => Gini médio ponderado de '{attribute}': {gini_weighted_sum:.4f}")

        return gini_weighted_sum

    def calc_gini_gain(self, attribute: str, df: pd.DataFrame = None) -> float:
        if df is None:
            df = self.df

        general = self.calc_gini(df)
        attr_gini = self.calc_attribute_gini(attribute, df)
        gain = general - attr_gini

        self._registrar(f"  => Ganho de Gini de '{attribute}': {gain:.4f}\n")

        return gain

    def best_attribute(self, attributes: list, df: pd.DataFrame = None) -> tuple:
        if df is None:
            df = self.df

        gains = {attr: self.calc_gini_gain(attr, df) for attr in attributes}

        self._registrar("Ganhos de Gini calculados:")
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
        self._montar_prova(origem, df, self.calc_gini(df), gains, escolhido)

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
        conteudo.append("===== LOG DE CONSTRUÇÃO (CART) =====\n")
        conteudo.extend(self.log)
        conteudo.append("\n===== ÁRVORE FINAL (CART) =====\n")
        conteudo.extend(self._arvore_texto())

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(conteudo))
