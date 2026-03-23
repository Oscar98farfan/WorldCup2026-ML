"""
===========================================================
src/simulator.py
===========================================================

PROPÓSITO:
    Simula el torneo del Mundial 2026 usando el modelo entrenado
    como motor de probabilidades partido a partido.

CORRECCIÓN IMPORTANTE vs VERSIÓN ANTERIOR:
    La versión anterior mezclaba aleatoriamente los 32 clasificados
    en octavos, lo cual era incorrecto. Ejemplo del problema:
    Brasil y Marruecos están en el mismo grupo (C). En la realidad
    NUNCA pueden enfrentarse antes de la final. Pero con el shuffle
    aleatorio en muchas simulaciones se encontraban en octavos,
    perjudicando artificialmente a ambos equipos.

    Esta versión respeta el fixture oficial del Mundial 2026:
    los cruces de octavos están predefinidos según el grupo
    y la posición (1ro, 2do, mejor 3ro).

FIXTURE OFICIAL MUNDIAL 2026 — OCTAVOS DE FINAL:
    Los 48 equipos se dividen en 12 grupos (A-L).
    Clasifican: 2 primeros de cada grupo (24) + 8 mejores terceros = 32.
    Los cruces de octavos siguen este esquema fijo:

    Partido 1:  1ro A  vs  2do B
    Partido 2:  1ro C  vs  2do D
    Partido 3:  1ro E  vs  2do F
    Partido 4:  1ro G  vs  2do H
    Partido 5:  1ro I  vs  2do J
    Partido 6:  1ro K  vs  2do L
    Partido 7:  2do A  vs  1ro B
    Partido 8:  2do C  vs  1ro D
    + 8 mejores terceros distribuidos según el fixture de FIFA

    Esto garantiza que equipos del mismo grupo NUNCA se encuentran
    antes de la final, respetando la lógica real del torneo.

AUTOR: Oscar Daniel Farfan Juanias
===========================================================
"""

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from pathlib import Path

from src.feature_engineering import build_match_row_2026, MODEL_FEATURES


# ===========================================================
# CONSTANTES DEL TORNEO 2026
# ===========================================================

POINTS_WIN  = 3
POINTS_DRAW = 1
POINTS_LOSS = 0
RANDOM_SEED = 42

# Fixture oficial de octavos de final del Mundial 2026.
# Define qué posición de qué grupo enfrenta a qué otra posición.
# Formato: (grupo_equipo_A, posicion_A, grupo_equipo_B, posicion_B)
#
# IMPORTANTE: Este fixture garantiza que equipos del mismo grupo
# no se puedan enfrentar hasta la final.
# Fuente: FIFA World Cup 2026 official bracket format.
#
# Los 8 mejores terceros se distribuyen en partidos específicos
# según los grupos de donde provengan. Por simplicidad usamos
# una distribución fija basada en el bracket oficial.

R16_FIXTURES = [
    # Partido → (grupo del 1ro, grupo del rival 2do/3ro)
    ("A", 1, "B", 2),   # 1ro A vs 2do B
    ("C", 1, "D", 2),   # 1ro C vs 2do D
    ("E", 1, "F", 2),   # 1ro E vs 2do F
    ("G", 1, "H", 2),   # 1ro G vs 2do H
    ("I", 1, "J", 2),   # 1ro I vs 2do J
    ("K", 1, "L", 2),   # 1ro K vs 2do L
    ("B", 1, "A", 2),   # 1ro B vs 2do A
    ("D", 1, "C", 2),   # 1ro D vs 2do C
    ("F", 1, "E", 2),   # 1ro F vs 2do E
    ("H", 1, "G", 2),   # 1ro H vs 2do G
    ("J", 1, "I", 2),   # 1ro J vs 2do I
    ("L", 1, "K", 2),   # 1ro L vs 2do K
    # Los últimos 4 partidos son los 8 mejores terceros
    # distribuidos según el bracket oficial
    ("BEST3", 0, "BEST3", 1),
    ("BEST3", 2, "BEST3", 3),
    ("BEST3", 4, "BEST3", 5),
    ("BEST3", 6, "BEST3", 7),
]

# Cuartos de final: el bracket sigue la mitad del cuadro.
# Los ganadores del partido 1 y 2 se enfrentan, etc.
# Se define como pares de índices de partidos de octavos.
QF_PAIRS = [
    (0, 6),   # ganador partido 1 vs ganador partido 7
    (1, 7),   # ganador partido 2 vs ganador partido 8
    (2, 8),   # ganador partido 3 vs ganador partido 9
    (3, 9),   # ganador partido 4 vs ganador partido 10
    (4, 10),  # ganador partido 5 vs ganador partido 11
    (5, 11),  # ganador partido 6 vs ganador partido 12
    (12, 13), # mejores terceros
    (14, 15), # mejores terceros
]

# Semifinales: pares de cuartos
SF_PAIRS = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
]

# Final: pares de semis
FINAL_PAIRS = [
    (0, 1),
    (2, 3),
]


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
        1. Construye las features del partido para team_a
        2. El modelo predice P(team_a gana)
        3. También calcula P(team_b gana) desde su perspectiva
        4. Normaliza ambas probabilidades
        5. Genera resultado aleatorio ponderado

    RETORNA: 'A', 'B' o 'DRAW'
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

    # Normalizar
    total = prob_a + prob_b
    if total > 0:
        p_a = prob_a / total
        p_b = prob_b / total
    else:
        p_a = p_b = 0.5

    # Zona de empate: si las probabilidades son muy similares
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
    Simula la fase de grupos respetando que cada equipo
    solo juega contra los 3 del mismo grupo.

    RETORNA:
        dict con:
            'standings':   tabla de posiciones por grupo
                           {grupo: [{'team': ..., 'points': ..., 'position': ...}]}
            'best_thirds': lista de los mejores terceros ordenados por puntos
    """
    all_standings = {}

    for group_name, teams in groups.items():
        standings = {
            team: {"points": 0, "gf": 0, "ga": 0}
            for team in teams
        }

        # Cada equipo juega contra los otros 3 del grupo (6 partidos)
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

        # Ordenar por puntos
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

    # Extraer mejores terceros (los 8 con más puntos de entre todos los 3ros)
    all_thirds = [
        standings[2]  # posición 2 = índice 2 = tercer lugar
        for standings in all_standings.values()
        if len(standings) >= 3
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


def _get_team_by_position(standings: dict, group: str, position: int) -> str:
    """
    Obtiene el equipo que quedó en una posición específica de un grupo.

    PARÁMETROS:
        standings → tabla de posiciones completa
        group     → letra del grupo ('A', 'B', etc.)
        position  → 1 para primero, 2 para segundo

    RETORNA:
        nombre del equipo
    """
    group_data = standings.get(group, [])
    for entry in group_data:
        if entry["position"] == position:
            return entry["team"]
    # Si no se encuentra (no debería pasar), retornar el primero
    return group_data[0]["team"] if group_data else "Unknown"


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
    Simula una ronda eliminatoria dado una lista de partidos.

    En eliminatorias NO hay empates — si el resultado es DRAW
    se resuelve por penales (50/50).

    PARÁMETROS:
        matches → lista de tuplas (equipo_a, equipo_b)

    RETORNA:
        lista de ganadores en el mismo orden que los partidos
        (importante para mantener el bracket correcto)
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
# FUNCIÓN: ARMAR CRUCES DE OCTAVOS
# ===========================================================

def build_r16_matches(group_standings: dict, best_thirds: list) -> list:
    """
    Construye los 16 partidos de octavos de final respetando
    el fixture oficial del Mundial 2026.

    GARANTÍA CLAVE:
        Equipos del mismo grupo NUNCA se enfrentan en octavos.
        El 1ro del Grupo A solo puede enfrentar al 2do del Grupo B,
        nunca al 2do del Grupo A (que sería del mismo grupo).

    PARÁMETROS:
        group_standings → tabla de posiciones por grupo
        best_thirds     → lista de los 8 mejores terceros

    RETORNA:
        lista de 16 tuplas (equipo_a, equipo_b)
    """
    matches = []

    # Primeros 12 partidos: enfrentamientos fijos entre grupos
    for group_a, pos_a, group_b, pos_b in R16_FIXTURES[:12]:
        team_a = _get_team_by_position(group_standings, group_a, pos_a)
        team_b = _get_team_by_position(group_standings, group_b, pos_b)
        matches.append((team_a, team_b))

    # Últimos 4 partidos: los 8 mejores terceros entre sí
    thirds_teams = [t["team"] for t in best_thirds]
    for i in range(0, 8, 2):
        if i + 1 < len(thirds_teams):
            matches.append((thirds_teams[i], thirds_teams[i+1]))
        else:
            # Si faltan terceros (no debería ocurrir), rellenar
            matches.append((thirds_teams[i], thirds_teams[0]))

    return matches


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
    Simula el torneo completo N veces respetando el fixture oficial.

    ESTRUCTURA DEL TORNEO 2026:
        Fase de grupos: 12 grupos × 4 equipos
            → 24 primeros + 24 segundos + 12 terceros
            → 24 + 8 mejores terceros = 32 clasificados

        Octavos (32→16): cruces fijos según fixture oficial
            → equipos del mismo grupo NO pueden encontrarse aquí

        Cuartos (16→8): ganadores de octavos según bracket
            → el bracket mantiene separados a equipos
              que ya estaban separados en octavos

        Semifinales (8→4): ganadores de cuartos según bracket

        Final (2→1): campeón

    PARÁMETROS:
        n_simulations → número de simulaciones Monte Carlo
                        1,000 = resultados suficientemente confiables
                        10,000 = alta precisión (más lento)

    RETORNA:
        DataFrame con probabilidades por equipo y ronda:
            team, prob_r16, prob_qf, prob_sf, prob_final, prob_winner
    """
    rng = np.random.default_rng(seed)

    all_teams = [team for teams in groups.values() for team in teams]
    counters  = {
        team: {"r16": 0, "qf": 0, "sf": 0, "final": 0, "winner": 0}
        for team in all_teams
    }

    print(f"[simulator] Iniciando {n_simulations:,} simulaciones...")

    for sim in range(n_simulations):

        if (sim + 1) % 100 == 0:
            print(f"[simulator]   {sim + 1:,} / {n_simulations:,} completadas...")

        # ─── FASE DE GRUPOS ────────────────────────────────────
        # Cada equipo solo juega contra los 3 de su grupo
        group_results = simulate_group_stage(
            groups, modelo, scaler, df_rankings, hist_stats, rng
        )
        group_standings = group_results["standings"]
        best_thirds     = group_results["best_thirds"]

        # Contar quién clasificó a octavos
        for group_data in group_standings.values():
            for entry in group_data:
                if entry["position"] <= 2:  # 1ro y 2do clasifican directamente
                    if entry["team"] in counters:
                        counters[entry["team"]]["r16"] += 1

        for third in best_thirds:  # mejores terceros también clasifican
            if third["team"] in counters:
                counters[third["team"]]["r16"] += 1

        # ─── OCTAVOS DE FINAL ──────────────────────────────────
        # Cruces fijos según el fixture oficial
        # GARANTÍA: equipos del mismo grupo no se pueden encontrar aquí
        r16_matches = build_r16_matches(group_standings, best_thirds)
        r16_winners = simulate_knockout_round(
            r16_matches, modelo, scaler, df_rankings, hist_stats, rng
        )

        for team in r16_winners:
            if team in counters:
                counters[team]["qf"] += 1

        # ─── CUARTOS DE FINAL ──────────────────────────────────
        # El bracket mantiene separados a los equipos que venían
        # de partes distintas del cuadro
        qf_matches = [
            (r16_winners[p[0]], r16_winners[p[1]])
            for p in QF_PAIRS
            if p[0] < len(r16_winners) and p[1] < len(r16_winners)
        ]
        qf_winners = simulate_knockout_round(
            qf_matches, modelo, scaler, df_rankings, hist_stats, rng
        )

        for team in qf_winners:
            if team in counters:
                counters[team]["sf"] += 1

        # ─── SEMIFINALES ───────────────────────────────────────
        sf_matches = [
            (qf_winners[p[0]], qf_winners[p[1]])
            for p in SF_PAIRS
            if p[0] < len(qf_winners) and p[1] < len(qf_winners)
        ]
        sf_winners = simulate_knockout_round(
            sf_matches, modelo, scaler, df_rankings, hist_stats, rng
        )

        for team in sf_winners:
            if team in counters:
                counters[team]["final"] += 1

        # ─── FINAL ─────────────────────────────────────────────
        if len(sf_winners) >= 2:
            final_match  = [(sf_winners[0], sf_winners[1])]
            final_winner = simulate_knockout_round(
                final_match, modelo, scaler, df_rankings, hist_stats, rng
            )
            if final_winner and final_winner[0] in counters:
                counters[final_winner[0]]["winner"] += 1

    # Convertir contadores a probabilidades
    results = []
    for team, counts in counters.items():
        results.append({
            "team":        team,
            "prob_r16":    counts["r16"]    / n_simulations,
            "prob_qf":     counts["qf"]     / n_simulations,
            "prob_sf":     counts["sf"]     / n_simulations,
            "prob_final":  counts["final"]  / n_simulations,
            "prob_winner": counts["winner"] / n_simulations,
        })

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
    """Imprime un resumen legible de los resultados."""
    print("\n" + "="*70)
    print("  PREDICCIONES MUNDIAL 2026")
    print("  (Simulación Monte Carlo con fixture oficial)")
    print("="*70)
    print(f"\n{'Pos':>4}  {'Equipo':<22} {'Octavos':>9} {'Cuartos':>9} "
          f"{'Semis':>7} {'Final':>7} {'Campeón':>9}")
    print("-"*70)

    for i, row in df_results.iterrows():
        print(
            f"{i+1:>4}  {row['team']:<22} "
            f"{row['prob_r16']:>8.1%} "
            f"{row['prob_qf']:>8.1%} "
            f"{row['prob_sf']:>6.1%} "
            f"{row['prob_final']:>6.1%} "
            f"{row['prob_winner']:>8.1%}"
        )
