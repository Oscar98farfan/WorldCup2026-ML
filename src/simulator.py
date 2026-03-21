"""
===========================================================
src/simulator.py
===========================================================

PROPÓSITO:
    Simula el torneo del Mundial 2026 usando el modelo entrenado
    como motor de probabilidades partido a partido.

¿CÓMO FUNCIONA LA SIMULACIÓN?
    En lugar de predecir una sola vez, simulamos el torneo
    N veces (por defecto 10,000). En cada simulación:

    1. Fase de grupos: cada equipo juega 3 partidos.
       El modelo predice P(equipo A gana a equipo B).
       Con esa probabilidad se "lanza una moneda" para
       decidir el resultado. Se acumulan puntos y clasifican
       los 2 primeros de cada grupo (24 clasificados de 12 grupos
       + 8 mejores terceros = 32 en octavos).

    2. Octavos: 32 equipos → 16 ganadores
    3. Cuartos: 16 equipos → 8 ganadores
    4. Semifinales: 8 equipos → 4 ganadores
    5. Final: 2 equipos → 1 campeón

    Al final contamos en cuántas simulaciones cada equipo
    llegó a cada ronda. Eso nos da las probabilidades.

¿POR QUÉ SIMULAR Y NO PREDECIR DIRECTAMENTE?
    El fútbol tiene componente aleatoria. Si Brasil tiene 70%
    de ganar a Marruecos, no siempre gana. La simulación
    captura esa incertidumbre de forma natural. Con 10,000
    simulaciones obtenemos una distribución de resultados
    mucho más informativa que una predicción puntual.

CONTIENE:
    simulate_match()         → simula un partido individual
    simulate_group_stage()   → simula toda la fase de grupos
    simulate_knockout()      → simula una ronda eliminatoria
    simulate_tournament()    → simula el torneo completo N veces
    get_results_summary()    → resume los resultados de las simulaciones

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

# Formato del Mundial 2026:
# - 48 equipos, 12 grupos de 4
# - Pasan los 2 primeros de cada grupo (24) + 8 mejores terceros = 32
# - Octavos → Cuartos → Semis → Final

# Puntos por resultado en fase de grupos
POINTS_WIN  = 3
POINTS_DRAW = 1
POINTS_LOSS = 0

# Semilla para reproducibilidad
# Cambiar este valor da resultados distintos pero igualmente válidos
RANDOM_SEED = 42


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
        1. Construye las features del partido para team_a vs team_b
        2. El modelo predice P(team_a gana)
        3. Con esa probabilidad genera un resultado aleatorio:
           - Si random < P(gana) → team_a gana
           - Si random > P(gana) → team_b gana
           (Los empates en grupos se manejan con una zona intermedia)

    PARÁMETROS:
        team_a, team_b → equipos del partido
        modelo         → modelo Gradient Boosting entrenado
        scaler         → StandardScaler ajustado en entrenamiento
        df_rankings    → ranking FIFA para construir features
        hist_stats     → estadísticas históricas de cada equipo
        is_home_team_a → 1 si team_a es el equipo "local"
        rng            → generador de números aleatorios
                         (se pasa para reproducibilidad entre simulaciones)

    RETORNA:
        'A' si gana team_a
        'B' si gana team_b
        'DRAW' si empatan
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # Construir features para team_a contra team_b
    X_a = build_match_row_2026(
        team_a, team_b, df_rankings, hist_stats, is_home_team_a
    )
    X_a_sc = scaler.transform(X_a)
    prob_a_wins = modelo.predict_proba(X_a_sc)[0, 1]

    # También calcular probabilidad de team_b (desde su perspectiva)
    X_b = build_match_row_2026(
        team_b, team_a, df_rankings, hist_stats, 1 - is_home_team_a
    )
    X_b_sc = scaler.transform(X_b)
    prob_b_wins = modelo.predict_proba(X_b_sc)[0, 1]

    # Normalizar para que sumen 1 (con espacio para empate)
    # prob_draw = 1 - prob_a_wins - prob_b_wins puede ser negativo,
    # así que usamos una distribución de 3 resultados
    total = prob_a_wins + prob_b_wins
    if total > 0:
        p_a = prob_a_wins / total
        p_b = prob_b_wins / total
    else:
        p_a = p_b = 0.5

    # Zona de empate: si las probabilidades son muy similares (< 15% de diferencia)
    # hay chance de empate (realista en fútbol)
    diff = abs(p_a - p_b)
    if diff < 0.15:
        # Probabilidad de empate proporcional a la similitud
        p_draw = (0.15 - diff) * 0.8
        p_a    = p_a * (1 - p_draw)
        p_b    = p_b * (1 - p_draw)
    else:
        p_draw = 0.0

    # Lanzar el dado
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
    Simula la fase de grupos del mundial.

    FORMATO MUNDIAL 2026:
        12 grupos de 4 equipos cada uno.
        Cada equipo juega 3 partidos (contra los otros 3 del grupo).
        Pasan los 2 primeros de cada grupo + 8 mejores terceros.

    PARÁMETROS:
        groups → dict {grupo: [equipo1, equipo2, equipo3, equipo4]}
        modelo, scaler, df_rankings, hist_stats → para predecir partidos
        rng → generador de números aleatorios

    RETORNA:
        dict con:
            'group_winners': lista de equipos que clasificaron (32 total)
            'standings': tabla de posiciones por grupo
    """
    all_standings = {}
    qualifiers = []  # equipos que clasifican a octavos

    for group_name, teams in groups.items():
        # Tabla de posiciones inicial
        standings = {
            team: {"points": 0, "gf": 0, "ga": 0, "gd": 0}
            for team in teams
        }

        # Jugar todos los partidos del grupo (6 partidos por grupo)
        for i, team_a in enumerate(teams):
            for team_b in teams[i+1:]:
                result = simulate_match(
                    team_a, team_b, modelo, scaler,
                    df_rankings, hist_stats, rng=rng
                )

                # Actualizar puntos
                if result == "A":
                    standings[team_a]["points"] += POINTS_WIN
                    standings[team_b]["points"] += POINTS_LOSS
                elif result == "B":
                    standings[team_b]["points"] += POINTS_WIN
                    standings[team_a]["points"] += POINTS_LOSS
                else:  # DRAW
                    standings[team_a]["points"] += POINTS_DRAW
                    standings[team_b]["points"] += POINTS_DRAW

        # Ordenar por puntos (desempate simple por ahora)
        sorted_teams = sorted(
            standings.keys(),
            key=lambda t: standings[t]["points"],
            reverse=True
        )

        all_standings[group_name] = {
            "ranking": sorted_teams,
            "points":  {t: standings[t]["points"] for t in sorted_teams}
        }

        # Los 2 primeros clasifican directamente
        qualifiers.append({
            "team":     sorted_teams[0],
            "group":    group_name,
            "position": 1
        })
        qualifiers.append({
            "team":     sorted_teams[1],
            "group":    group_name,
            "position": 2
        })

    # Los 8 mejores terceros también clasifican
    # (regla del Mundial 2026 con 48 equipos)
    thirds = []
    for group_name, data in all_standings.items():
        if len(data["ranking"]) >= 3:
            third_team   = data["ranking"][2]
            third_points = data["points"][third_team]
            thirds.append({
                "team":     third_team,
                "group":    group_name,
                "position": 3,
                "points":   third_points
            })

    # Ordenar terceros por puntos y tomar los 8 mejores
    thirds_sorted = sorted(thirds, key=lambda x: x["points"], reverse=True)
    best_thirds   = thirds_sorted[:8]
    qualifiers.extend(best_thirds)

    return {
        "qualifiers":  qualifiers,  # 32 equipos
        "standings":   all_standings
    }


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
    Simula una ronda eliminatoria (octavos, cuartos, semis o final).

    En eliminatorias NO hay empates — si el resultado es DRAW,
    se resuelve con penales (simulados como 50/50 con ligera ventaja
    al equipo con mejor ranking).

    PARÁMETROS:
        matches → lista de tuplas (equipo_a, equipo_b)
        modelo, scaler, df_rankings, hist_stats → para predecir
        rng → generador de números aleatorios

    RETORNA:
        lista de equipos ganadores (uno por partido)
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
            # Ligera ventaja al equipo con menor número de rank
            winners.append(rng.choice([team_a, team_b]))

    return winners


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
    Simula el torneo completo N veces y acumula los resultados.

    PROCESO COMPLETO:
        Para cada simulación:
        1. Simular fase de grupos → 32 clasificados
        2. Simular octavos → 16 clasificados
        3. Simular cuartos → 8 clasificados
        4. Simular semifinales → 4 clasificados (+ 2 al tercer puesto)
        5. Simular final → 1 campeón

    RESULTADO:
        Para cada equipo, contamos en cuántas simulaciones llegó
        a cada ronda. Dividido entre N simulaciones = probabilidad.

    PARÁMETROS:
        groups         → dict {grupo: [equipo1, equipo2, equipo3, equipo4]}
        modelo         → modelo entrenado
        scaler         → escalador
        df_rankings    → ranking FIFA
        hist_stats     → estadísticas históricas
        n_simulations  → número de simulaciones (más = más preciso pero más lento)
                         10,000 tarda ~2-5 min según tu hardware
        seed           → semilla para reproducibilidad

    RETORNA:
        DataFrame con columnas:
            team, prob_r16, prob_qf, prob_sf, prob_final, prob_winner
            (probabilidad de llegar a cada ronda)
    """
    rng = np.random.default_rng(seed)

    # Contador de llegadas por ronda para cada equipo
    all_teams = [team for teams in groups.values() for team in teams]
    counters  = {
        team: {"r16": 0, "qf": 0, "sf": 0, "final": 0, "winner": 0}
        for team in all_teams
    }

    print(f"[simulator] Iniciando {n_simulations:,} simulaciones...")

    for sim in range(n_simulations):

        # Progreso cada 1000 simulaciones
        if (sim + 1) % 1000 == 0:
            print(f"[simulator]   {sim + 1:,} / {n_simulations:,} completadas...")

        # --- FASE DE GRUPOS ---
        group_results = simulate_group_stage(
            groups, modelo, scaler, df_rankings, hist_stats, rng
        )
        qualifiers_r16 = [q["team"] for q in group_results["qualifiers"]]

        for team in qualifiers_r16:
            if team in counters:
                counters[team]["r16"] += 1

        # --- OCTAVOS DE FINAL ---
        # Emparejar: 1ro grupo A vs mejor 3ro, 2do grupo A vs 2do grupo B, etc.
        # Simplificación: emparejamos aleatoriamente los 32 clasificados
        # (el fixture real depende de la posición en el grupo)
        rng.shuffle(qualifiers_r16)
        matches_r16 = [(qualifiers_r16[i], qualifiers_r16[i+1])
                       for i in range(0, 32, 2)]

        winners_qf = simulate_knockout_round(
            matches_r16, modelo, scaler, df_rankings, hist_stats, rng
        )
        for team in winners_qf:
            if team in counters:
                counters[team]["qf"] += 1

        # --- CUARTOS DE FINAL ---
        rng.shuffle(winners_qf)
        matches_qf = [(winners_qf[i], winners_qf[i+1])
                      for i in range(0, 16, 2)]

        winners_sf = simulate_knockout_round(
            matches_qf, modelo, scaler, df_rankings, hist_stats, rng
        )
        for team in winners_sf:
            if team in counters:
                counters[team]["sf"] += 1

        # --- SEMIFINALES ---
        rng.shuffle(winners_sf)
        matches_sf = [(winners_sf[i], winners_sf[i+1])
                      for i in range(0, 8, 2)]

        winners_final = simulate_knockout_round(
            matches_sf, modelo, scaler, df_rankings, hist_stats, rng
        )
        for team in winners_final:
            if team in counters:
                counters[team]["final"] += 1

        # --- FINAL ---
        matches_final = [(winners_final[0], winners_final[1])]
        winner = simulate_knockout_round(
            matches_final, modelo, scaler, df_rankings, hist_stats, rng
        )
        if winner and winner[0] in counters:
            counters[winner[0]]["winner"] += 1

    # Convertir contadores a probabilidades
    results = []
    for team, counts in counters.items():
        results.append({
            "team":         team,
            "prob_r16":     counts["r16"]    / n_simulations,
            "prob_qf":      counts["qf"]     / n_simulations,
            "prob_sf":      counts["sf"]     / n_simulations,
            "prob_final":   counts["final"]  / n_simulations,
            "prob_winner":  counts["winner"] / n_simulations,
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
    """
    Imprime un resumen legible de los resultados de la simulación.

    PARÁMETROS:
        df_results → resultado de simulate_tournament()
    """
    print("\n" + "="*70)
    print("  PREDICCIONES MUNDIAL 2026")
    print("  (basadas en simulación de Monte Carlo)")
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
