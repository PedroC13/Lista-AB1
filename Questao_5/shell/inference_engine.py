class InferenceEngine:
    """Engenho de inferência: encadeamento pra frente, pra trás e misto, mais explicação (por que / como)."""

    def __init__(self, base, ask_fact=None):
        self.base = base
        self.facts = dict(base.initial_facts)
        self.trace = []  # regras que dispararam, na ordem, para o "como"
        self.questions = []  # (atributo, pilha_de_regras_ate_aqui) para o "por que"
        self._why_cursor = {}  # quantos níveis da pilha já mostrei pra cada atributo
        self.ask_fact = ask_fact or (lambda attribute: input(f"Qual o valor de '{attribute}'? "))

    def _rule_satisfied(self, rule):
        return all(self.facts.get(a) == v for a, v in rule["conditions"])

    def _rule_by_id(self, rule_id):
        return next((r for r in self.base.rules if r["id"] == rule_id), None)

    # ---------- encadeamento para frente ----------

    def forward_chaining(self):
        """Aplica todas as regras que já podem disparar, repetindo até não sobrar novidade."""
        changed = True
        fired_now = []
        while changed:
            changed = False
            for rule in self.base.rules:
                attr, value = rule["conclusion"]
                if self.facts.get(attr) == value:
                    continue
                if self._rule_satisfied(rule):
                    self.facts[attr] = value
                    self.trace.append(rule)
                    fired_now.append(rule["id"])
                    changed = True
        return self.facts

    # ---------- encadeamento para trás (com detecção de ciclo) ----------

    def backward_chaining(self, goal):
        """Tenta provar o atributo 'goal', perguntando ao usuário só o que não dá pra deduzir."""
        return self._prove(goal, stack=[], interleave_forward=False, visiting=set())

    def mixed_chaining(self, goal):
        """Intercala: tenta deduzir pra frente, e só pergunta/retrocede no que falta — e cada
        fato novo obtido (perguntado) dispara uma nova rodada de encadeamento pra frente,
        porque ele pode servir pra outras regras também, não só pra provar o objetivo atual."""
        self.forward_chaining()
        if goal in self.facts:
            return self.facts[goal]
        return self._prove(goal, stack=[], interleave_forward=True, visiting=set())

    def _prove(self, attribute, stack, interleave_forward, visiting):
        if attribute in self.facts:
            return self.facts[attribute]

        if attribute in visiting:
            # ciclo: essa regra depende (direta ou indiretamente) de si mesma
            return None
        visiting = visiting | {attribute}

        for rule in self.base.rules_concluding(attribute):
            ok = True
            for a, v in rule["conditions"]:
                self.questions.append((a, stack + [rule["id"]]))
                obtained = self._prove(a, stack + [rule["id"]], interleave_forward, visiting)
                if obtained != v:
                    ok = False
                    break
            if ok:
                self.facts[attribute] = rule["conclusion"][1]
                self.trace.append(rule)
                if interleave_forward:
                    self.forward_chaining()
                return self.facts[attribute]

        # não tem regra que resolva -> é um fato primitivo, pergunta pro usuário
        value = self.ask_fact(attribute)
        self.facts[attribute] = value
        if interleave_forward:
            self.forward_chaining()
        return value

    # ---------- explicação ----------

    def explain_why(self, attribute):
        """'Por quê' no estilo pergunta repetida: cada chamada sobe mais um nível na
        cadeia de regras até chegar no objetivo principal da consulta."""
        occurrences = [s for a, s in self.questions if a == attribute]
        if not occurrences:
            return f"'{attribute}' não foi perguntado por causa de nenhuma regra (é um dado de entrada direto)."

        stack = occurrences[-1]
        cursor = self._why_cursor.get(attribute, 0)
        if cursor >= len(stack):
            self._why_cursor[attribute] = 0
            return "Já expliquei até o objetivo principal desta consulta (voltando ao começo)."

        rule_id = stack[-(cursor + 1)]
        rule = self._rule_by_id(rule_id)
        self._why_cursor[attribute] = cursor + 1

        condition = " E ".join(f"{a} = {v}" for a, v in rule["conditions"])
        aviso = " (pergunte 'por que' de novo para subir mais um nível)" if cursor + 1 < len(stack) else ""
        return (f"Perguntando '{attribute}' para tentar a regra {rule['id']}: "
                f"SE {condition} ENTAO {rule['conclusion'][0]} = {rule['conclusion'][1]}{aviso}")

    def explain_how(self, attribute=None):
        """'Como': mostra a árvore de regras que levou a um fato concluído, recursivamente."""
        if attribute is None:
            if not self.trace:
                return "Nenhuma regra disparou ainda."
            return "\n".join(self._how_lines(r, 0) for r in self.trace if self._is_top_level(r))

        rule = next((r for r in self.trace if r["conclusion"][0] == attribute), None)
        if rule is None:
            if attribute in self.facts:
                return f"'{attribute}' = {self.facts[attribute]} foi informado direto (não veio de regra)."
            return f"'{attribute}' ainda não foi concluído nesta consulta."
        return self._how_lines(rule, 0)

    def _is_top_level(self, rule):
        # uma regra é "de topo" se nenhuma outra regra disparada usa sua conclusão como condição
        attr = rule["conclusion"][0]
        return not any(attr in [a for a, _ in r["conditions"]] for r in self.trace)

    def _how_lines(self, rule, depth):
        indent = "  " * depth
        condition = " E ".join(f"{a} = {v}" for a, v in rule["conditions"])
        attr, value = rule["conclusion"]
        lines = [f"{indent}{rule['id']}: SE {condition} ENTAO {attr} = {value}"]
        for a, v in rule["conditions"]:
            sub_rule = next((r for r in self.trace if r["conclusion"][0] == a), None)
            if sub_rule:
                lines.append(self._how_lines(sub_rule, depth + 1))
            else:
                lines.append(f"{indent}  (fato dado: {a} = {self.facts.get(a)})")
        return "\n".join(lines)
