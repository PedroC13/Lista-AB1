import re
import difflib

# Camada de diálogo em linguagem natural: bem simples (regex + casamento aproximado de
# palavras), sem nenhuma biblioteca de ML — só pra permitir frases soltas em vez de só
# números de menu. Isso também é um "shell genérico": não conhece nenhum atributo
# específico, só recebe a lista de atributos da base carregada e tenta casar.

_PADROES = [
    ("help", r"^(ajuda|help|\?)$"),
    ("facts", r"\b(fatos|o que (voce|você) sabe)\b"),
    ("rules", r"\b(regras|base de regras|mostra(r)? as regras)\b"),
    ("save", r"\b(salv(a|e|ar))\b"),
    ("reset", r"\b(reinici|nova consulta|recome(c|ç)ar|limpar)\b"),
    ("add_rule", r"\b(adicionar|criar|nova) regra\b"),
    ("remove_rule", r"\b(remover|apagar|excluir) regra\b"),
    ("load", r"\b(carregar|trocar|abrir) base\b"),
    ("menu", r"^(menu)$"),
    ("exit", r"^(sair|tchau|fim|exit|quit)$"),
]

_PADRAO_COMO = re.compile(r"\bcomo\b.*\b([a-zA-Z_çãõáéíóúâêô ]+)\??$", re.IGNORECASE)
_PADRAO_PORQUE = re.compile(r"\bpor ?qu[eê]\b")
_PADRAO_CONSULTA = re.compile(
    r"(?:qual|quais|qual é|qual e|descobrir|determinar|quero saber|me diga|diga-me|calcul\w*)\s+(?:o|a|os|as)?\s*([a-zA-Z_çãõáéíóúâêô ]+?)\??$",
    re.IGNORECASE,
)


def _normalizar(texto: str) -> str:
    return texto.strip().lower()


def casar_atributo(trecho: str, atributos: list) -> str:
    """Acha o atributo conhecido mais parecido com o trecho de texto dado (tolera erro de digitação/abreviação)."""
    trecho = trecho.strip().lower()
    if not trecho:
        return None

    candidatos = {a.lower(): a for a in atributos}
    if trecho in candidatos:
        return candidatos[trecho]

    # substring: "risco" casa com "Risco de Credito" etc.
    por_substring = [original for low, original in candidatos.items() if trecho in low or low in trecho]
    if len(por_substring) == 1:
        return por_substring[0]

    aproximados = difflib.get_close_matches(trecho, candidatos.keys(), n=1, cutoff=0.6)
    if aproximados:
        return candidatos[aproximados[0]]

    return None


def interpretar(texto: str, atributos: list) -> tuple:
    """Retorna (intencao, argumento). intencao='unknown' quando não reconhece nada."""
    bruto = texto.strip()
    low = _normalizar(bruto)

    if _PADRAO_PORQUE.search(low):
        return ("why", None)

    m = _PADRAO_COMO.search(low)
    if m:
        atributo = casar_atributo(m.group(1), atributos)
        return ("how", atributo)
    if low == "como":
        return ("how", None)

    for intencao, padrao in _PADROES:
        if re.search(padrao, low):
            return (intencao, None)

    m = _PADRAO_CONSULTA.search(low)
    if m:
        atributo = casar_atributo(m.group(1), atributos)
        if atributo:
            return ("query", atributo)

    # última tentativa: o texto inteiro é (quase) o nome de um atributo
    atributo = casar_atributo(low, atributos)
    if atributo:
        return ("query", atributo)

    return ("unknown", bruto)
