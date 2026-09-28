import pandas as pd

class prism:
    def __init__(self, df: pd.DataFrame, verbose: bool = False) -> None:
        self.df = df
        self.verbose = verbose
        self.target = None
        self.rules = []
        self.default_class = None
        self.log = []

    def _record(self, text: str) -> None:
        if self.verbose:
            self.log.append(text)

    def fit(self, target: str):
        self.target = target
        self.default_class = self.df[target].mode()[0]
        attributes = [c for c in self.df.columns if c != target]

        for class_label in self.df[target].unique():
            self._record(f"\n=== Gerando regras para a classe '{class_label}' ===")
            self.rules.extend(self._rules_for_class(class_label, attributes))

        return self.rules

    def _rules_for_class(self, class_label: str, attributes: list) -> list:
        rules = []
        pending = self.df.copy()

        while (pending[self.target] == class_label).any():
            conditions = {}
            available = list(attributes)
            subgroup = pending

            while subgroup[self.target].nunique() > 1 and available:
                best = None
                for attr in available:
                    for value in subgroup[attr].unique():
                        slice_ = subgroup[subgroup[attr] == value]
                        t = len(slice_)
                        p = (slice_[self.target] == class_label).sum()
                        score = p / t
                        if best is None or score > best[0] or (score == best[0] and p > best[1]):
                            best = (score, p, attr, value)

                _, p, attr, value = best
                conditions[attr] = value
                available.remove(attr)
                subgroup = subgroup[subgroup[attr] == value]

                self._record(f"  Adiciona '{attr} = {value}' (p={p}, t={len(subgroup)})")

            rules.append((conditions, class_label))
            self._record(f"  => Regra: SE {conditions} ENTAO {self.target} = {class_label} (cobre {len(subgroup)} exemplos)")

            pending = pending.drop(subgroup.index)

        return rules

    def predict(self, row):
        for conditions, class_label in self.rules:
            if all(row[attr] == value for attr, value in conditions.items()):
                return class_label
        return self.default_class

    def classify_dataset(self, df: pd.DataFrame = None, filepath: str = None) -> pd.DataFrame:
        if df is None:
            df = self.df

        result = pd.DataFrame({
            "Real": df[self.target],
            "Predito": df.apply(self.predict, axis=1),
        })

        if filepath:
            result.to_csv(filepath, index=False)

        return result

    def export_rules(self, filepath: str = None) -> str:
        lines = []
        for conditions, class_label in self.rules:
            condition_text = " E ".join(f"{attr} = {value}" for attr, value in conditions.items())
            lines.append(f"SE {condition_text} ENTAO {self.target} = {class_label}")
        lines.append(f"SENAO {self.target} = {self.default_class} (regra padrao)")

        text = "\n".join(lines)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

        return text

    def export_log(self, filepath: str) -> None:
        if not self.verbose:
            return

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(self.log))
