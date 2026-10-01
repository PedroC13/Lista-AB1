import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.append("shell")
from knowledge_base import KnowledgeBase
from inference_engine import InferenceEngine

BASES_FOLDER = "bases"


def choose_base():
    files = [f for f in os.listdir(BASES_FOLDER) if f.endswith(".json")]
    print("\nBases de conhecimento disponíveis:")
    for i, name in enumerate(files, 1):
        print(f"  {i}. {name}")
    choice = input("Escolha o número (ou ENTER para a primeira): ").strip()
    idx = int(choice) - 1 if choice else 0
    return os.path.join(BASES_FOLDER, files[idx])


def show_rules(base):
    print("\nRegras:")
    for r in base.rules:
        condition = " E ".join(f"{a} = {v}" for a, v in r["conditions"])
        print(f"  {r['id']}: SE {condition} ENTAO {r['conclusion'][0]} = {r['conclusion'][1]}")

#
def add_rule(base):
    rule_id = input("Id da regra (ex: R10): ").strip()
    n = int(input("Quantas condições? "))
    conditions = []
    for _ in range(n):
        attr = input("  Atributo: ").strip()
        value = input("  Valor: ").strip()
        conditions.append([attr, value])
    concluded_attr = input("Atributo concluído: ").strip()
    concluded_value = input("Valor concluído: ").strip()
    base.add_rule(rule_id, conditions, [concluded_attr, concluded_value])
    print("Regra adicionada.")


def remove_rule(base):
    base.remove_rule(input("Id da regra a remover: ").strip())
    print("Removida (se existia).")


def menu():
    base_path = choose_base()
    base = KnowledgeBase.load(base_path)
    engine = InferenceEngine(base)  # acumula fatos/trilhas durante toda a sessão

    while True:
        print("""
--- Shell de Sistema Baseado em Conhecimento ---
1. Ver fatos
2. Ver regras
3. Adicionar regra
4. Remover regra
5. Encadeamento para frente
6. Encadeamento para trás (perguntar um objetivo)
7. Encadeamento misto
8. Explicação: por que perguntou X? / como chegou em X?
9. Salvar base
f. Definir um fato manualmente (para testar)
r. Reiniciar fatos (nova consulta do zero)
0. Sair""")
        op = input("Opção: ").strip()

        if op == "1":
            print("\nFatos conhecidos até agora:", engine.facts)
        elif op == "2":
            show_rules(base)
        elif op == "3":
            add_rule(base)
        elif op == "4":
            remove_rule(base)
        elif op == "f":
            attr = input("Atributo: ").strip()
            value = input("Valor: ").strip()
            engine.facts[attr] = value
        elif op == "5":
            print("Fatos deduzidos:", engine.forward_chaining())
        elif op == "6":
            goal = input("Qual atributo você quer descobrir? ").strip()
            print(f"{goal} = {engine.backward_chaining(goal)}")
        elif op == "7":
            goal = input("Qual atributo você quer descobrir? ").strip()
            print(f"{goal} = {engine.mixed_chaining(goal)}")
        elif op == "8":
            print("Digite 'p' para 'por quê' ou 'c' para 'como'.")
            kind = input("> ").strip().lower()
            if kind == "p":
                print(engine.explain_why(input("Atributo perguntado: ").strip()))
            else:
                print(engine.explain_how())
        elif op == "9":
            base.save(base_path)
            print("Salvo em", base_path)
        elif op == "r":
            engine = InferenceEngine(base)
            print("Fatos reiniciados.")
        elif op == "0":
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    menu()
