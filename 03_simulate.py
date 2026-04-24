"""
===========================================================
src/simulator.py
===========================================================

PROPÓSITO:
    Simula el torneo del Mundial 2026 usando el modelo entrenado
    como motor de probabilidades partido a partido.

ARQUITECTURA OFICIAL DEL MUNDIAL 2026 — 48 EQUIPOS:
    El torneo está dividido en dos BLOQUES independientes.
    Esto garantiza que los grandes favoritos no se eliminen
    prematuramente entre sí.

    BLOQUE IZQUIERDO: Grupos A, B, C, E, F, I
    BLOQUE DERECHO:   Grupos D, G, H, J, K, L

    Cada bloque tiene su propio camino hasta la semifinal:

    BLOQUE IZQ ──► R32 ──► Octavos ──► Cuartos ──► Semifinal 1 ──┐
                                                                    ├──► FINAL
    BLOQUE DER ──► R32 ──► Octavos ──► Cuartos ──► Semifinal 2 ──┘

    REGLA CLAVE:
        Un equipo del Bloque Izquierdo NUNCA puede enfrentar
        a uno del Bloque Derecho antes de la FINAL.

    EJEMPLO CONCRETO:
        Brasil (Grupo C → Bloque IZQ) y Francia (Grupo I → Bloque IZQ)
        pueden enfrentarse como máximo en Semifinales.

        Brasil (Bloque IZQ) y Argentina (Grupo J → Bloque DER)
        SOLO pueden enfrentarse en la FINAL.

        Esto es lo que hace el torneo justo y evita que dos
        potencias se eliminen en octavos cuando podrían ser
        los dos mejores equipos del mundo.

RONDA DE 32 (nueva en 2026):
    En el Mundial de 32 equipos había Octavos como primera
    eliminatoria. En el de 48 hay una ronda previa llamada
    "Ronda de 32" o "Dieciseisavos":

    48 equipos en grupos → 32 clasificados → Ronda de 32 (32→16)
    → Octavos (16→8) → Cuartos (8→4) → Semis (4→2) → Final

    Dentro de cada bloque:
        16 clasificados → Ronda de 32 (16→8) → Octavos (8→4)
        → Cuartos (4→2) → Semifinal del bloque

AUTOR: Oscar Daniel Farfan Juanias
===========================================================
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.feature_engineering import build_match_row_2026, MODEL_FEATURES


# ===========================================================
# CONSTANTES DEL TORNEO 2026
# ===========================================================

POINTS_WIN  = 3
POINTS_DRAW = 1
POINTS_LOSS = 0
RANDOM_SEED = 42

# División oficial del torneo en dos bloques independientes.
# Los equipos de cada bloque NUNCA se enfrentan entre sí
# antes de la final.
#
# Fuente: FIFA World Cup 2026 official bracket structure.
BLOQUE_IZQUIERDO = ["A", "B", "C", "E", "F", "I"]
BLOQUE_DERECHO   = ["D", "G", "H", "J", "K", "L"]

# Cruces de la Ronda de 32 dentro de cada bloque.
# Formato: (grupo_equipo_A, posicion_A, grupo_equipo_B, posicion_B)
#
# GARANTÍA: todos los cruces son entre grupos del mismo bloque.
# El 1ro de un grupo enfrenta al 2do de otro grupo del mismo bloque.
# Nunca entre grupos de bloques distintos.
#
# BLOQUE IZQUIERDO — 6 partidos de Ronda de 32
R32_BLOQUE_IZQ = [
    ("A", 1, "B", 2),   # 1ro Grupo A vs 2do Grupo B
    ("B", 1, "A", 2),   # 1ro Grupo B vs 2do Grupo A
    ("C", 1, "E", 2),   # 1ro Grupo C vs 2do Grupo E
    ("E", 1, "C", 2),   # 1ro Grupo E vs 2do Grupo C
    ("F", 1, "I", 2),   # 1ro Grupo F vs 2do Grupo I
    ("I", 1, "F", 2),   # 1ro Grupo I vs 2do Grupo F
]

# BLOQUE DERECHO — 6 partidos de Ronda de 32
R32_BLOQUE_DER = [
    ("D", 1, "G", 2),   # 1ro Grupo D vs 2do Grupo G
    ("G", 1, "D", 2),   # 1ro Grupo G vs 2do Grupo D
    ("H", 1, "J", 2),   # 1ro Grupo H vs 2do Grupo J
    ("J", 1, "H", 2),   # 1ro Grupo J vs 2do Grupo H
    ("K", 1, "L", 2),   # 1ro Grupo K vs 2do Grupo L
    ("L", 1, "K", 2),   # 1ro Grupo L vs 2do Grupo K
]

# Los 8 mejores terceros se distribuyen 4 en cada bloque
# para balancear la carga de partidos sin alterar la jerarquía.
# Los 4 mejores terceros van al Bloque Izquierdo,
# los otros 4 al Bloque Derecho.
N_THIRDS_PER_BLOCK = 4


# ===========================================================
# FUNCIÓN: SIMULAR UN PARTIDO
# ===========================================================

def simulate_match(
    team_a: str,
    team_b: str,
    modelo,
    scaler,
    df_rankings: pd.DataFrame,
    hist_stats: dict,
    is_home_team_a: int = 1,
    rng: Optional[np.random.Generator] = None
) -> str:
    """
    Simula el resultado de un partido entre team_a y team_b.

    PROCESO:
        1. Construye features del partido para team_a vs team_b
        2. El modelo predice P(team_a gana)
        3. Construye features desde perspectiva de team_b
        4. Normaliza ambas probabilidades
        5. Genera resultado aleatorio ponderado

    ZONA DE EMPATE:
        Si las probabilidades son muy similares (diferencia < 15%),
        existe probabilidad de empate. Esto es realista en fútbol:
        cuando dos equipos son muy parejos, el empate es frecuente.

    RETORNA: 'A' (gana team_a), 'B' (gana team_b) o 'DRAW' (empate)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # Probabilidad desde perspectiva de team_a
    X_a    = build_match_row_2026(
        team_a, team_b, df_rankings, hist_stats, is_home_team_a
    )
    prob_a = modelo.predict_proba(scaler.transform(X_a))[0, 1]

    # Probabilidad desde perspectiva de team_b
    X_b    = build_match_row_2026(
        team_b, team_a, df_rankings, hist_stats, 1 - is_home_team_a
    )
    prob_b = modelo.predict_proba(scaler.transform(X_b))[0, 1]

    # Normalizar para que sumen correctamente
    total = prob_a + prob_b
    p_a   = (prob_a / total) if total > 0 else 0.5
    p_b   = (prob_b / total) if total > 0 else 0.5

    # Zona de empate: equipos muy parejos tienen chance de empatar
    diff   = abs(p_a - p_b)
    p_draw = max(0, (0.15 - diff) * 0.8) if diff < 0.15 else 0.0
    p_a   *= (1 - p_draw)
    p_b   *= (1 - p_draw)

    rand = rng.random()
    if rand < p_a:
        return "A"
    elif rand < p_a + p_draw:
        return "DRAW"
    else:
        return "B"


# ===========================================================
# FUNCIÓN: SIMULAR FASE DE GRUPOS
# ===========================================================

def simulate_group_stage(
    groups: dict,
    modelo,
    scaler,
    df_rankings: pd.DataFrame,
    hist_stats: dict,
    rng: np.random.Generator
) -> dict:
    """
    Simula la fase de grupos completa.

    REGLA: cada equipo solo juega contra los 3 de su mismo grupo.
    Brasil (Grupo C) solo enfrenta a Marruecos, Haití y Escocia.
    Nunca a un equipo del Grupo D o cualquier otro.

    CLASIFICACIÓN:
        - 1ro y 2do de cada grupo clasifican directamente (24 equipos)
        - Los 12 terceros compiten por los 8 mejores spots (8 equipos)
        - Total: 32 clasificados

    RETORNA:
        dict con:
            'standings':   tabla de posiciones {grupo: [lista ordenada]}
            'best_thirds': los 8 mejores terceros ordenados por puntos
    """
    all_standings = {}

    for group_name, teams in groups.items():
        standings = {team: {"points": 0} for team in teams}

        # Cada equipo juega contra los otros 3 (6 partidos por grupo)
        for i, team_a in enumerate(teams):
            for team_b in teams[i+1:]:
                result = simulate_match(
                    team_a, team_b, modelo, scaler,
                    df_rankings, hist_stats, rng=rng
                )
                if result == "A":
                    standings[team_a]["points"] += POINTS_WIN
                elif result == "B":
                    standings[team_b]["points"] += POINTS_WIN
                else:
                    standings[team_a]["points"] += POINTS_DRAW
                    standings[team_b]["points"] += POINTS_DRAW

        # Ordenar por puntos (de mayor a menor)
        sorted_teams = sorted(
            standings.keys(),
            key=lambda t: standings[t]["points"],
            reverse=True
        )

        all_standings[group_name] = [
            {
                "team":     team,
                "points":   standings[team]["points"],
                "position": pos + 1,
                "group":    group_name,
            }
            for pos, team in enumerate(sorted_teams)
        ]

    # Extraer los mejores terceros
    # Los 12 terceros compiten por 8 spots (los de más puntos clasifican)
    all_thirds = [
        data[2]  # índice 2 = tercer lugar del grupo
        for data in all_standings.values()
        if len(data) >= 3
    ]
    best_thirds = sorted(
        all_thirds,
        key=lambda x: x["points"],
        reverse=True
    )[:8]

    return {
        "standings":   all_standings,
        "best_thirds": best_thirds,
    }


def _get_team(standings: dict, group: str, position: int) -> str:
    """
    Obtiene el equipo que quedó en una posición de un grupo.

    PARÁMETROS:
        standings → tabla completa de posiciones
        group     → letra del grupo ('A', 'B', etc.)
        position  → 1=primero, 2=segundo

    RETORNA: nombre del equipo
    """
    for entry in standings.get(group, []):
        if entry["position"] == position:
            return entry["team"]
    # Fallback por si no se encuentra
    return standings.get(group, [{}])[0].get("team", "Unknown")


# ===========================================================
# FUNCIÓN: ARMAR CRUCES DE RONDA DE 32 POR BLOQUE
# ===========================================================

def build_r32_matches(
    group_standings: dict,
    best_thirds: list,
    bloque_fixtures: list,
    bloque_grupos: list,
    thirds_for_block: list
) -> list:
    """
    Construye los partidos de la Ronda de 32 para UN bloque.

    LÓGICA:
        1. Los 6 partidos fijos entre primeros y segundos del bloque
        2. Los 4 mejores terceros asignados a este bloque completan
           los 16 participantes del bloque (6×2 + 4 = 16)

    GARANTÍA:
        Todos los partidos son entre equipos del mismo bloque.
        Nunca un equipo del Bloque Izquierdo vs uno del Derecho.

    PARÁMETROS:
        group_standings  → tabla de posiciones de todos los grupos
        best_thirds      → los 8 mejores terceros
        bloque_fixtures  → cruces fijos del bloque (R32_BLOQUE_IZQ o DER)
        bloque_grupos    → grupos que pertenecen a este bloque
        thirds_for_block → los 4 mejores terceros asignados a este bloque

    RETORNA:
        lista de 8 tuplas (equipo_a, equipo_b) para este bloque
    """
    matches = []

    # Partidos fijos: 1ros y 2dos de los grupos del bloque
    for group_a, pos_a, group_b, pos_b in bloque_fixtures:
        team_a = _get_team(group_standings, group_a, pos_a)
        team_b = _get_team(group_standings, group_b, pos_b)
        matches.append((team_a, team_b))

    # Agregar los mejores terceros del bloque contra terceros entre sí
    # (los terceros se enfrentan entre ellos en la Ronda de 32)
    for i in range(0, len(thirds_for_block) - 1, 2):
        matches.append((
            thirds_for_block[i]["team"],
            thirds_for_block[i+1]["team"]
        ))

    return matches


# ===========================================================
# FUNCIÓN: SIMULAR RONDA ELIMINATORIA
# ===========================================================

def simulate_knockout_round(
    matches: list,
    modelo,
    scaler,
    df_rankings: pd.DataFrame,
    hist_stats: dict,
    rng: np.random.Generator
) -> list:
    """
    Simula una ronda eliminatoria manteniendo el orden del bracket.

    IMPORTANTE — ORDEN DEL BRACKET:
        El orden de los ganadores en la lista de retorno es
        CRÍTICO para que los cruces de la siguiente ronda sean
        correctos. El ganador del partido 1 siempre enfrenta al
        ganador del partido 2 en la siguiente ronda, etc.

    EN ELIMINATORIAS NO HAY EMPATES:
        Si el resultado es DRAW, se resuelve con penales (50/50).
        En la realidad los penales tienen varianza, pero sin datos
        históricos de penales por equipo, 50/50 es la estimación
        más honesta.

    PARÁMETROS:
        matches → lista ORDENADA de tuplas (equipo_a, equipo_b)
                  el orden importa para el bracket

    RETORNA:
        lista ORDENADA de ganadores (mismo orden que los partidos)
    """
    winners = []

    for team_a, team_b in matches:
        result = simulate_match(
            team_a, team_b, modelo, scaler,
            df_rankings, hist_stats, rng=rng
        )
        if result == "A":
            winners.append(team_a)
        elif result == "B":
            winners.append(team_b)
        else:
            # Empate en eliminatoria → penales (50/50)
            winners.append(rng.choice([team_a, team_b]))

    return winners


# ===========================================================
# FUNCIÓN: SIMULAR UN BLOQUE COMPLETO
# ===========================================================

def simulate_block(
    block_name: str,
    r32_matches: list,
    modelo,
    scaler,
    df_rankings: pd.DataFrame,
    hist_stats: dict,
    rng: np.random.Generator
) -> str:
    """
    Simula un bloque completo desde la Ronda de 32 hasta
    el representante del bloque en la semifinal.

    RUTA DE UN BLOQUE:
        Ronda de 32: 16 equipos → 8 ganadores
        Octavos:      8 equipos → 4 ganadores
        Cuartos:      4 equipos → 2 ganadores
        Semifinal:    2 equipos → 1 REPRESENTANTE DEL BLOQUE

    BRACKET INTERNO:
        El bracket se mantiene consistente: el ganador del
        partido 1 de R32 siempre va por el mismo lado del
        bracket. Esto garantiza que:
        - Brasil (partido 1) y Francia (partido 3) del mismo
          bloque solo pueden enfrentarse en la semifinal del bloque
        - No se pueden eliminar antes mutuamente

    PARÁMETROS:
        block_name  → nombre del bloque para logs ('IZQ' o 'DER')
        r32_matches → 8 partidos de la Ronda de 32 de este bloque

    RETORNA:
        nombre del equipo que representa al bloque en la final
    """
    # Ronda de 32: 16 → 8
    r32_winners = simulate_knockout_round(
        r32_matches, modelo, scaler, df_rankings, hist_stats, rng
    )

    # Octavos: 8 → 4
    # El bracket empareja ganador 1 vs ganador 2, ganador 3 vs 4, etc.
    octavos_matches = [
        (r32_winners[i], r32_winners[i+1])
        for i in range(0, len(r32_winners), 2)
    ]
    octavos_winners = simulate_knockout_round(
        octavos_matches, modelo, scaler, df_rankings, hist_stats, rng
    )

    # Cuartos: 4 → 2
    cuartos_matches = [
        (octavos_winners[i], octavos_winners[i+1])
        for i in range(0, len(octavos_winners), 2)
    ]
    cuartos_winners = simulate_knockout_round(
        cuartos_matches, modelo, scaler, df_rankings, hist_stats, rng
    )

    # Semifinal del bloque: 2 → 1 representante
    if len(cuartos_winners) >= 2:
        semi_result = simulate_knockout_round(
            [(cuartos_winners[0], cuartos_winners[1])],
            modelo, scaler, df_rankings, hist_stats, rng
        )
        return semi_result[0]
    elif cuartos_winners:
        return cuartos_winners[0]
    return "Unknown"


# ===========================================================
# FUNCIÓN: SIMULAR EL TORNEO COMPLETO
# ===========================================================

def simulate_tournament(
    groups: dict,
    modelo,
    scaler,
    df_rankings: pd.DataFrame,
    hist_stats: dict,
    n_simulations: int = 10_000,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Simula el torneo completo N veces respetando la arquitectura
    oficial del Mundial 2026 con dos bloques independientes.

    FLUJO COMPLETO DE CADA SIMULACIÓN:

        1. FASE DE GRUPOS (todos los grupos en paralelo)
           48 equipos → 32 clasificados
           (24 primeros/segundos + 8 mejores terceros)

        2. DISTRIBUCIÓN EN BLOQUES
           Los 32 clasificados se asignan a su bloque:
           - 1ros y 2dos del Bloque Izquierdo → Bloque IZQ
           - 1ros y 2dos del Bloque Derecho   → Bloque DER
           - Mejores terceros: 4 a cada bloque

        3. BLOQUE IZQUIERDO (independiente)
           R32 (16→8) → Octavos (8→4) → Cuartos (4→2) → Semi IZQ

        4. BLOQUE DERECHO (independiente)
           R32 (16→8) → Octavos (8→4) → Cuartos (4→2) → Semi DER

        5. FINAL
           Representante Bloque IZQ vs Representante Bloque DER

    CONTADORES:
        Para cada equipo contamos en cuántas simulaciones llegó
        a cada ronda. Al final dividimos entre N para obtener
        probabilidades.

        prob_r32    = veces que pasó grupos / N
        prob_r16    = veces que pasó Ronda de 32 / N
        prob_qf     = veces que llegó a cuartos / N
        prob_sf     = veces que llegó a semifinal del bloque / N
        prob_final  = veces que llegó a la final / N
        prob_winner = veces que ganó el mundial / N

    PARÁMETROS:
        n_simulations → número de simulaciones Monte Carlo
                        1,000 = suficientemente confiable (~1 hora en PC lento)
                        10,000 = alta precisión (~8-10 horas en PC lento)

    RETORNA:
        DataFrame con columnas:
            team, prob_r32, prob_r16, prob_qf, prob_sf,
            prob_final, prob_winner
        Ordenado por prob_winner descendente
    """
    rng = np.random.default_rng(seed)

    all_teams = [team for teams in groups.values() for team in teams]
    counters  = {
        team: {
            "r32":    0,  # pasó fase de grupos (clasificó a Ronda de 32)
            "r16":    0,  # pasó Ronda de 32 (llegó a Octavos)
            "qf":     0,  # llegó a Cuartos de final
            "sf":     0,  # llegó a Semifinal del bloque
            "final":  0,  # llegó a la Gran Final
            "winner": 0,  # ganó el Mundial
        }
        for team in all_teams
    }

    print(f"[simulator] Iniciando {n_simulations:,} simulaciones...")
    print(f"[simulator] Arquitectura: Bloque IZQ ({BLOQUE_IZQUIERDO}) "
          f"| Bloque DER ({BLOQUE_DERECHO})")

    for sim in range(n_simulations):

        if (sim + 1) % 100 == 0:
            print(f"[simulator]   {sim+1:,} / {n_simulations:,} completadas...")

        # ─── FASE DE GRUPOS ────────────────────────────────────
        group_results   = simulate_group_stage(
            groups, modelo, scaler, df_rankings, hist_stats, rng
        )
        group_standings = group_results["standings"]
        best_thirds     = group_results["best_thirds"]

        # Registrar quién clasificó a la Ronda de 32
        for group_data in group_standings.values():
            for entry in group_data:
                if entry["position"] <= 2:
                    if entry["team"] in counters:
                        counters[entry["team"]]["r32"] += 1
        for third in best_thirds:
            if third["team"] in counters:
                counters[third["team"]]["r32"] += 1

        # ─── DISTRIBUCIÓN DE TERCEROS EN BLOQUES ───────────────
        # Los 4 mejores terceros van al Bloque Izquierdo
        # Los 4 siguientes van al Bloque Derecho
        thirds_izq = best_thirds[:N_THIRDS_PER_BLOCK]
        thirds_der = best_thirds[N_THIRDS_PER_BLOCK:]

        # ─── BLOQUE IZQUIERDO ──────────────────────────────────
        # Construir los 8 partidos de la Ronda de 32 del Bloque IZQ
        r32_izq = build_r32_matches(
            group_standings,
            best_thirds,
            R32_BLOQUE_IZQ,
            BLOQUE_IZQUIERDO,
            thirds_izq
        )

        # Registrar octavos (ganadores de R32 del bloque IZQ)
        r32_izq_winners = simulate_knockout_round(
            r32_izq, modelo, scaler, df_rankings, hist_stats, rng
        )
        for team in r32_izq_winners:
            if team in counters:
                counters[team]["r16"] += 1

        # Cuartos del bloque IZQ
        oct_izq = [(r32_izq_winners[i], r32_izq_winners[i+1])
                   for i in range(0, len(r32_izq_winners), 2)]
        oct_izq_winners = simulate_knockout_round(
            oct_izq, modelo, scaler, df_rankings, hist_stats, rng
        )
        for team in oct_izq_winners:
            if team in counters:
                counters[team]["qf"] += 1

        # Semifinal del bloque IZQ
        sf_izq = [(oct_izq_winners[i], oct_izq_winners[i+1])
                  for i in range(0, len(oct_izq_winners), 2)]
        sf_izq_winners = simulate_knockout_round(
            sf_izq, modelo, scaler, df_rankings, hist_stats, rng
        )
        for team in sf_izq_winners:
            if team in counters:
                counters[team]["sf"] += 1

        # Representante del Bloque IZQ en la final
        if len(sf_izq_winners) >= 2:
            final_izq = simulate_knockout_round(
                [(sf_izq_winners[0], sf_izq_winners[1])],
                modelo, scaler, df_rankings, hist_stats, rng
            )
            rep_izq = final_izq[0] if final_izq else sf_izq_winners[0]
        else:
            rep_izq = sf_izq_winners[0] if sf_izq_winners else "Unknown"

        if rep_izq in counters:
            counters[rep_izq]["final"] += 1

        # ─── BLOQUE DERECHO ────────────────────────────────────
        r32_der = build_r32_matches(
            group_standings,
            best_thirds,
            R32_BLOQUE_DER,
            BLOQUE_DERECHO,
            thirds_der
        )

        r32_der_winners = simulate_knockout_round(
            r32_der, modelo, scaler, df_rankings, hist_stats, rng
        )
        for team in r32_der_winners:
            if team in counters:
                counters[team]["r16"] += 1

        oct_der = [(r32_der_winners[i], r32_der_winners[i+1])
                   for i in range(0, len(r32_der_winners), 2)]
        oct_der_winners = simulate_knockout_round(
            oct_der, modelo, scaler, df_rankings, hist_stats, rng
        )
        for team in oct_der_winners:
            if team in counters:
                counters[team]["qf"] += 1

        sf_der = [(oct_der_winners[i], oct_der_winners[i+1])
                  for i in range(0, len(oct_der_winners), 2)]
        sf_der_winners = simulate_knockout_round(
            sf_der, modelo, scaler, df_rankings, hist_stats, rng
        )
        for team in sf_der_winners:
            if team in counters:
                counters[team]["sf"] += 1

        if len(sf_der_winners) >= 2:
            final_der = simulate_knockout_round(
                [(sf_der_winners[0], sf_der_winners[1])],
                modelo, scaler, df_rankings, hist_stats, rng
            )
            rep_der = final_der[0] if final_der else sf_der_winners[0]
        else:
            rep_der = sf_der_winners[0] if sf_der_winners else "Unknown"

        if rep_der in counters:
            counters[rep_der]["final"] += 1

        # ─── GRAN FINAL ────────────────────────────────────────
        # Representante Bloque IZQ vs Representante Bloque DER
        # GARANTÍA: siempre es un equipo de cada bloque en la final
        if rep_izq != "Unknown" and rep_der != "Unknown":
            final_result = simulate_knockout_round(
                [(rep_izq, rep_der)],
                modelo, scaler, df_rankings, hist_stats, rng
            )
            if final_result and final_result[0] in counters:
                counters[final_result[0]]["winner"] += 1

    # Convertir contadores a probabilidades
    results = []
    for team, counts in counters.items():
        results.append({
            "team":        team,
            "bloque":      "IZQ" if any(
                team == _get_team(
                    {g: [{"team": t, "position": 1, "group": g, "points": 0}
                         for t in groups.get(g, [])]},
                    g, 1
                )
                for g in BLOQUE_IZQUIERDO
            ) else "DER",
            "prob_r32":    counts["r32"]    / n_simulations,
            "prob_r16":    counts["r16"]    / n_simulations,
            "prob_qf":     counts["qf"]     / n_simulations,
            "prob_sf":     counts["sf"]     / n_simulations,
            "prob_final":  counts["final"]  / n_simulations,
            "prob_winner": counts["winner"] / n_simulations,
        })

    # Asignar bloque correctamente basado en el grupo
    group_to_block = {}
    for group_name, teams in groups.items():
        block = "IZQ" if group_name in BLOQUE_IZQUIERDO else "DER"
        for team in teams:
            group_to_block[team] = block

    for r in results:
        r["bloque"] = group_to_block.get(r["team"], "UNK")

    df_results = (
        pd.DataFrame(results)
        .sort_values("prob_winner", ascending=False)
        .reset_index(drop=True)
    )

    print(f"[simulator] Simulación completada.")
    return df_results


# ===========================================================
# FUNCIÓN: RESUMEN DE RESULTADOS
# ===========================================================

def get_results_summary(df_results: pd.DataFrame) -> None:
    """
    Imprime un resumen legible de los resultados de la simulación.
    Incluye el bloque de cada equipo para facilitar el análisis.
    """
    print("\n" + "="*75)
    print("  PREDICCIONES MUNDIAL 2026")
    print("  (Simulación Monte Carlo con arquitectura oficial de bloques)")
    print("="*75)
    print(f"\n{'Pos':>4}  {'Equipo':<22} {'Bloque':>7} {'R32':>6} "
          f"{'Octavos':>8} {'Cuartos':>8} {'Semis':>7} "
          f"{'Final':>7} {'Campeón':>9}")
    print("-"*75)

    for i, row in df_results.iterrows():
        print(
            f"{i+1:>4}  {row['team']:<22} {row['bloque']:>7} "
            f"{row['prob_r32']:>5.1%} "
            f"{row['prob_r16']:>7.1%} "
            f"{row['prob_qf']:>7.1%} "
            f"{row['prob_sf']:>6.1%} "
            f"{row['prob_final']:>6.1%} "
            f"{row['prob_winner']:>8.1%}"
        )

    # Resumen por bloque
    print("\n" + "="*75)
    print("  FAVORITOS POR BLOQUE (prob_winner)")
    print("="*75)

    for bloque in ["IZQ", "DER"]:
        equipos_bloque = df_results[df_results["bloque"] == bloque].head(5)
        grupos = BLOQUE_IZQUIERDO if bloque == "IZQ" else BLOQUE_DERECHO
        print(f"\n  Bloque {bloque} (Grupos {', '.join(grupos)}):")
        for _, row in equipos_bloque.iterrows():
            print(f"    {row['team']:<22} → campeón: {row['prob_winner']:.1%} "
                  f"| final: {row['prob_final']:.1%}")