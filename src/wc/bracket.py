"""
Cuadro oficial del Mundial 2026 (partidos 73 a 104).

Por qué NO se separa en "grupos A–F a la izquierda y G–L a la derecha":
el cuadro de FIFA está fijado por POSICIÓN en el grupo, y cada mitad del
cuadro mezcla grupos de las dos "mitades" alfabéticas. Ejemplo real:
España (H) y Francia (I) se cruzaron en semifinal, mientras que Argentina
(J) e Inglaterra (L) lo hicieron en la otra. Con la separación A–F/G–L,
una final España–Argentina, la que se jugó, era imposible.

Tampoco es aleatorio después de grupos: de dieciseisavos a la final el
camino está fijado de antemano. Lo único que depende de la combinación de
resultados es contra qué grupo-ganador cae cada uno de los 8 mejores
terceros (tabla de 495 combinaciones del Anexo C del reglamento).
"""
from __future__ import annotations

# Slots: "1E" = primero del grupo E, "2B" = segundo del B,
# "3:ABCDF" = un tercero que venga de alguno de esos grupos.
ROUND_OF_32 = {
    73: ("2A", "2B"),      74: ("1E", "3:ABCDF"),
    75: ("1F", "2C"),      76: ("1C", "2F"),
    77: ("1I", "3:CDFGH"), 78: ("2E", "2I"),
    79: ("1A", "3:CEFHI"), 80: ("1L", "3:EHIJK"),
    81: ("1D", "3:BEFIJ"), 82: ("1G", "3:AEHIJ"),
    83: ("2K", "2L"),      84: ("1H", "2J"),
    85: ("1B", "3:EFGIJ"), 86: ("1J", "2H"),
    87: ("1K", "3:DEIJL"), 88: ("2D", "2G"),
}
# Ronda siguiente: (partido de origen 1, partido de origen 2)
ROUND_OF_16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
               93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QUARTER_FINALS = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SEMI_FINALS = {101: (97, 98), 102: (99, 100)}
FINAL = {104: (101, 102)}

KNOCKOUT_ROUNDS = [("r16", ROUND_OF_16), ("qf", QUARTER_FINALS),
                   ("sf", SEMI_FINALS), ("final", FINAL)]

# Mitad del cuadro de cada partido de dieciseisavos (útil para la web).
HALF = {m: ("arriba" if m in (73, 74, 75, 77, 81, 82, 83, 84) else "abajo")
        for m in ROUND_OF_32}

# Asignaciones de terceros ya conocidas del Anexo C (clave = grupos de los
# 8 terceros clasificados). Se puede ampliar con la tabla completa.
KNOWN_THIRD_PLACE_ASSIGNMENTS = {
    # Combinación 67, la que se dio en el torneo real.
    "BDEFIJKL": {74: "D", 77: "F", 79: "E", 80: "K",
                 81: "B", 82: "I", 85: "J", 87: "L"},
}


def assign_third_places(third_groups: list[str]) -> dict[int, str]:
    """Asigna los 8 terceros clasificados a los partidos que los esperan.

    Si la combinación está en la tabla conocida se usa tal cual. Si no, se
    busca una asignación válida (cada tercero solo a un slot que lo admita)
    por backtracking, empezando por los slots con menos opciones. Es un
    emparejamiento válido según las reglas, aunque puede no coincidir
    exactamente con el que eligió FIFA para esa combinación.
    """
    key = "".join(sorted(third_groups))
    if key in KNOWN_THIRD_PLACE_ASSIGNMENTS:
        return KNOWN_THIRD_PLACE_ASSIGNMENTS[key]

    slots = {m: set(b.split(":")[1]) & set(third_groups)
             for m, (_, b) in ROUND_OF_32.items() if b.startswith("3:")}
    order = sorted(slots, key=lambda m: len(slots[m]))
    assignment: dict[int, str] = {}

    def solve(i: int, used: set[str]) -> bool:
        if i == len(order):
            return True
        match = order[i]
        for g in sorted(slots[match] - used):
            assignment[match] = g
            if solve(i + 1, used | {g}):
                return True
        assignment.pop(match, None)
        return False

    if not solve(0, set()):
        raise ValueError(f"Combinación de terceros sin asignación válida: {key}")
    return assignment
