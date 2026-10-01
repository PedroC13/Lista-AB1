import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.append("shell")
from knowledge_base import KnowledgeBase
from inference_engine import InferenceEngine
import nlp

BASES_FOLDER = "bases"


def list_bases():
    return sorted(f for f in os.listdir(BASES_FOLDER) if f.endswith(".json"))


def choose_base():
    files = list_bases()
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
    try:
        base.add_rule(rule_id, conditions, [concluded_attr, concluded_value])
        print("Regra adicionada.")
    except ValueError as e:
        print("Não deu:", e)


def remove_rule(base):
    base.remove_rule(input("Id da regra a remover: ").strip())
    print("Removida (se existia).")


def make_ask_fact(engine):
    """Pergunta um fato ao usuário, mas aceita 'por que'/'como' no meio da pergunta antes de responder de verdade."""
    def ask(attribute):
        while True:
            resp = input(f"  Qual o valor de '{attribute}'? (ou 'por que' / 'como') ").strip()
            low = resp.lower()
            if low in ("por que", "porque", "pq"):
                print("  ->", engine.explain_why(attribute))
                continue
            if low == "como":
                print("  ->", engine.explain_how())
                continue
            return resp
    return ask


HELP_TEXT = """
Você pode digitar em linguagem natural, por exemplo:
  "qual o risco?"            -> consulta o atributo (encadeamento misto)
  "como risco"                -> explica como um fato foi concluído
  "por que"                   -> (durante uma pergunta) explica por que perguntou aquilo
  "fatos" / "regras"           -> mostra o que a base sabe até agora / a base de regras
  "nova consulta"              -> limpa os fatos e começa do zero
  "adicionar regra" / "remover regra" -> edita a base
  "carregar base"               -> troca de base de conhecimento
  "salvar"                      -> grava a base em disco
  "sair"                        -> encerra
Comandos numéricos do menu antigo também funcionam (digite "menu" pra ver)."""

MENU_TEXT = """
--- Menu numérico (alternativa ao diálogo em linguagem natural) ---
1. Ver fatos                 5. Encadeamento para frente
2. Ver regras                6. Encadeamento para trás (pede um objetivo)
3. Adicionar regra           7. Encadeamento misto (pede um objetivo)
4. Remover regra             8. Explicação: p=por quê / c=como
f. Definir fato manualmente   9. Salvar base
r. Reiniciar fatos            0. Sair"""


def run_numeric(op, base, engine, base_path):
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
        kind = input("Digite 'p' (por quê) ou 'c' (como): ").strip().lower()
        if kind == "p":
            print(engine.explain_why(input("Atributo perguntado: ").strip()))
        else:
            print(engine.explain_how())
    elif op == "9":
        base.save(base_path)
        print("Salvo em", base_path)
    elif op == "r":
        return InferenceEngine(base)  # quem chama é responsável por religar o ask_fact
    elif op == "0":
        return "sair"
    else:
        print("Opção inválida.")
    return engine


def repl():
    base_path = choose_base()
    base = KnowledgeBase.load(base_path)
    engine = InferenceEngine(base)
    engine.ask_fact = make_ask_fact(engine)

    print(f"\nBase carregada: {base_path}")
    print(HELP_TEXT)

    while True:
        entrada = input("\nVocê: ").strip()
        if not entrada:
            continue

        if entrada == "menu":
            print(MENU_TEXT)
            op = input("Opção: ").strip()
            resultado = run_numeric(op, base, engine, base_path)
            if resultado == "sair":
                break
            if resultado is not None:
                engine = resultado
                engine.ask_fact = make_ask_fact(engine)
            continue

        intencao, arg = nlp.interpretar(entrada, base.concludable_attributes() + base.all_attributes())

        if intencao == "exit":
            break
        elif intencao == "help":
            print(HELP_TEXT)
        elif intencao == "facts":
            print("Fatos conhecidos até agora:", engine.facts or "(nenhum ainda)")
        elif intencao == "rules":
            show_rules(base)
        elif intencao == "reset":
            engine = InferenceEngine(base)
            engine.ask_fact = make_ask_fact(engine)
            print("Consulta reiniciada (fatos apagados).")
        elif intencao == "save":
            base.save(base_path)
            print("Salvo em", base_path)
        elif intencao == "add_rule":
            add_rule(base)
        elif intencao == "remove_rule":
            remove_rule(base)
        elif intencao == "load":
            base_path = choose_base()
            base = KnowledgeBase.load(base_path)
            engine = InferenceEngine(base)
            engine.ask_fact = make_ask_fact(engine)
            print(f"Base carregada: {base_path}")
        elif intencao == "why":
            print("Essa pergunta só faz sentido durante uma pergunta de fato; tente de novo quando o sistema perguntar algo.")
        elif intencao == "how":
            if arg is None:
                print(engine.explain_how())
            else:
                print(engine.explain_how(arg))
        elif intencao == "query":
            if arg is None:
                print("Não reconheci de qual atributo você está falando. Tente 'regras' ou 'fatos' pra ver os nomes.")
            else:
                valor = engine.mixed_chaining(arg)
                print(f"\n=> {arg} = {valor}")
        else:
            print("Não entendi. Digite 'ajuda' para ver os comandos, ou 'menu' para o menu numérico.")


if __name__ == "__main__":
    repl()
