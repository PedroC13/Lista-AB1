import json

class KnowledgeBase:
    """Guarda fatos e regras SE...ENTAO de um domínio qualquer, sem saber nada do domínio em si."""

    def __init__(self, initial_facts=None, rules=None):
        self.initial_facts = initial_facts or {}
        self.rules = rules or []  # cada regra: {"id": str, "conditions": [[attr, value], ...], "conclusion": [attr, value]}

    def add_rule(self, rule_id, conditions, conclusion):
        if any(r["id"] == rule_id for r in self.rules):
            raise ValueError(f"já existe uma regra com id '{rule_id}'")
        self.rules.append({"id": rule_id, "conditions": conditions, "conclusion": list(conclusion)})

    def remove_rule(self, rule_id):
        self.rules = [r for r in self.rules if r["id"] != rule_id]

    def add_fact(self, attribute, value):
        self.initial_facts[attribute] = value

    def remove_fact(self, attribute):
        self.initial_facts.pop(attribute, None)

    def rules_concluding(self, attribute):
        return [r for r in self.rules if r["conclusion"][0] == attribute]

    def all_attributes(self):
        """Todos os atributos citados na base (em condições ou conclusões), pra ajudar o usuário/a interface em LN."""
        attrs = set(self.initial_facts.keys())
        for r in self.rules:
            attrs.update(a for a, _ in r["conditions"])
            attrs.add(r["conclusion"][0])
        return sorted(attrs)

    def concludable_attributes(self):
        """Só os atributos que alguma regra conclui (bons candidatos a objetivo de consulta)."""
        return sorted({r["conclusion"][0] for r in self.rules})

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"initial_facts": self.initial_facts, "rules": self.rules}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data.get("initial_facts", {}), data.get("rules", []))
