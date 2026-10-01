class InferenceEngine:
    """Engenho de inferência: encadeamento pra frente, pra trás e misto, mais explicação (por que / como)."""

    def __init__(self, base, ask_fact=None):
        self.base = base
        self.facts = dict(base.initial_facts)
        self.trace = []  # regras que dispararam, na ordem, para o "como"
        self.questions = []  # (atributo, regra_que_motivou) para o "por que"
        self.ask_fact = ask_fact or (lambda attribute: input(f"Qual o valor de '{attribute}'? "))

    def _rule_satisfied(self, rule):
        return all(self.facts.get(a) == v for a, v in rule["conditions"])

    def forward_chaining(self):
        """Aplica todas as regras que já podem disparar, repetindo até não sobrar novidade."""
        changed = True
        while changed:
            changed = False
            for rule in self.base.rules:
                attr, value = rule["conclusion"]
                if self.facts.get(attr) == value:
                    continue
                if self._rule_satisfied(rule):
                    self.facts[attr] = value
                    self.trace.append(rule)
                    changed = True
        return self.facts

    def backward_chaining(self, goal):
        """Tenta provar o atributo 'goal', perguntando ao usuário só o que não dá pra deduzir."""
        return self._prove(goal, stack=[])

    def _prove(self, attribute, stack):
        if attribute in self.facts:
            return self.facts[attribute]

        candidates = self.base.rules_concluding(attribute)
        for rule in candidates:
            ok = True
            for a, v in rule["conditions"]:
                self.questions.append((a, rule["id"]))
                obtained_value = self._prove(a, stack + [rule["id"]])
                if obtained_value != v:
                    ok = False
                    break
            if ok:
                self.facts[attribute] = rule["conclusion"][1]
                self.trace.append(rule)
                return self.facts[attribute]

        # não tem regra que resolva -> é um fato primitivo, pergunta pro usuário
        value = self.ask_fact(attribute)
        self.facts[attribute] = value
        return value

    def mixed_chaining(self, goal):
        """Primeiro deduz tudo que der pra frente, depois só completa o que falta pra trás."""
        self.forward_chaining()
        return self._prove(goal, stack=[])

    def explain_how(self):
        if not self.trace:
            return "Nenhuma regra disparou ainda."
        lines = []
        for rule in self.trace:
            condition = " E ".join(f"{a} = {v}" for a, v in rule["conditions"])
            attr, value = rule["conclusion"]
            lines.append(f"{rule['id']}: SE {condition} ENTAO {attr} = {value}")
        return "\n".join(lines)

    def explain_why(self, attribute):
        reasons = [r for a, r in self.questions if a == attribute]
        if not reasons:
            return f"'{attribute}' não foi perguntado por causa de nenhuma regra (é um dado de entrada direto)."
        rule = next(r for r in self.base.rules if r["id"] == reasons[-1])
        condition = " E ".join(f"{a} = {v}" for a, v in rule["conditions"])
        return f"Perguntando '{attribute}' para tentar a regra {rule['id']}: SE {condition} ENTAO {rule['conclusion'][0]} = {rule['conclusion'][1]}"
