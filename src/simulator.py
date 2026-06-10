"""
===========================================================
src/simulator.py  —  VERSIÓN CORREGIDA
===========================================================

CAMBIOS vs VERSIÓN ANTERIOR:
    1. Fixtures de octavos rediseñados con estructura limpia.
       Antes: tuplas de longitud mixta (2 o 4 elementos) causaban
       que _build_half leyera índices incorrectos en los partidos
       normales y crasheara silenciosamente en los BEST3.
       Ahora: dos listas separadas — una para cruces normales
       (siempre 2 elementos) y otra para los índices BEST3.

    2. _pick_finalist ahora maneja cualquier cantidad de
       semifinalistas (no solo exactamente 2).

    3. El relleno de thirds vacíos ya no duplica equipos;
       en su lugar lanza una advertencia clara.

    4. Separación de bloques IZQ (A–F) | DER (G–L) totalmente
       respetada: los dos bloques no se cruzan hasta la final.

AUTOR: Oscar Daniel Farfan Juanias
===========================================================
"""

import pandas as pd
import numpy as np
from typing import Optional, List

from src.feature_engineering import build_match_row_2026, MODEL_FEATURES


# ===========================================================
# CONSTANTES
# ===========================================================

POINTS_WIN  = 3
POINTS_DRAW = 1
POINTS_LOSS = 0
RANDOM_SEED = 42

DEFAULT_BLOQUE_IZQ = ["A", "B", "C", "D", "E", "F"]
DEFAULT_BLOQUE_DER = ["G", "H", "I", "J", "K", "L"]

# ── Fixture de octavos por bloque ──────────────────────────
# Cada entrada es (grupo_del_1ro, grupo_del_2do).
# Son los 6 cruces "normales" (1ro de un grupo vs 2do de otro).
# Los 2 partidos restantes son entre los mejores terceros del bloque.
R16_NORMAL_IZQ = [
    ("A", "B"),   # 1ro A vs 2do B
    ("C", "D"),   # 1ro C vs 2do D
    ("E", "F"),   # 1ro E vs 2do F
    ("B", "A"),   # 1ro B vs 2do A
    ("D", "C"),   # 1ro D vs 2do C
    ("F", "E"),   # 1ro F vs 2do E
]
# Pares de índices para los 4 mejores terceros del bloque IZQ
# (índice dentro de la lista thirds_izq ordenada por puntos)
R16_BEST3_IZQ = [(0, 1), (2, 3)]

R16_NORMAL_DER = [
    ("G", "H"),
    ("I", "J"),
    ("K", "L"),
    ("H", "G"),
    ("J", "I"),
    ("L", "K"),
]
R16_BEST3_DER = [(0, 1), (2, 3)]

# Cuartos y semis: pares de índices sobre la lista de ganadores
# (índice 0 = ganador del partido 0 de octavos, etc.)
QF_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
SF_PAIRS = [(0, 1), (2, 3)]


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
    rng: Optional[np.random.Generator] = None,
) -> str:
    """
    Simula el resultado de un partido entre team_a y team_b.

    Calcula P(team_a gana) y P(team_b gana) desde la perspectiva
    de cada uno, normaliza y genera un resultado aleatorio ponderado.

    RETORNA: 'A', 'B' o 'DRAW'
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    X_a = build_match_row_2026(
        team_a, team_b, df_rankings, hist_stats, is_home_team_a
    )
    prob_a = modelo.predict_proba(scaler.transform(X_a))[0, 1]

    X_b = build_match_row_2026(
        team_b, team_a, df_rankings, hist_stats, 1 - is_home_team_a
    )
    prob_b = modelo.predict_proba(scaler.transform(X_b))[0, 1]

    total = prob_a + prob_b
    if total > 0:
        p_a = prob_a / total
        p_b = prob_b / total
    else:
        p_a = p_b = 0.5

    # Zona de empate: si las probabilidades son muy cercanas,
    # hay una chance de empate proporcional a cuán iguales son.
    diff   = abs(p_a - p_b)
    p_draw = max(0.0, (0.15 - diff) * 0.8) if diff < 0.15 else 0.0
    p_a   *= (1.0 - p_draw)
    p_b   *= (1.0 - p_draw)

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
    rng: np.random.Generator,
) -> dict:
    """
    Simula la fase de grupos: cada equipo juega contra los
    otros 3 del mismo grupo (todos contra todos, 6 partidos por grupo).

    RETORNA:
        {
          'standings':   {grupo: [{'team', 'points', 'position', 'group'}]},
          'best_thirds': lista de los 8 mejores terceros (todos los grupos)
        }
    """
    all_standings = {}

    for group_name, teams in groups.items():
        standings = {team: {"points": 0} for team in teams}

        for i, team_a in enumerate(teams):
            for team_b in teams[i + 1:]:
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

        sorted_teams = sorted(
            standings.keys(),
            key=lambda t: standings[t]["points"],
            reverse=True,
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

    # Los 8 mejores terceros de todos los grupos (para rellenar octavos)
    all_thirds = [
        group_data[2]
        for group_data in all_standings.values()
        if len(group_data) >= 3
    ]
    best_thirds = sorted(all_thirds, key=lambda x: x["points"], reverse=True)[:8]

    return {"standings": all_standings, "best_thirds": best_thirds}


# ===========================================================
# FUNCIÓN: OBTENER EQUIPO POR POSICIÓN EN SU GRUPO
# ===========================================================

def _get_team_by_position(standings: dict, group: str, position: int) -> str:
    """
    Retorna el equipo que quedó en `position` (1=primero, 2=segundo)
    dentro del grupo `group`.
    """
    for entry in standings.get(group, []):
        if entry["position"] == position:
            return entry["team"]
    # Fallback: primer equipo del grupo (no debería llegar aquí)
    group_data = standings.get(group, [])
    return group_data[0]["team"] if group_data else "Unknown"


# ===========================================================
# FUNCIÓN: ARMAR CRUCES DE OCTAVOS  (CORREGIDA)
# ===========================================================

def build_r16_matches(
    group_standings: dict,
    best_thirds: list,
    bloque_izq: List[str],
    bloque_der: List[str],
) -> dict:
    """
    Construye los 16 partidos de octavos de final, separados por bloque.

    LÓGICA:
        - 6 cruces normales por bloque: 1ro de grupo X vs 2do de grupo Y.
        - 2 cruces de mejores terceros por bloque (4 terceros del bloque).
        - Equipos del mismo bloque NUNCA se enfrentan con los del otro
          hasta la final.

    RETORNA:
        {
            'izq': [(team_a, team_b), ...],  # 8 partidos
            'der': [(team_a, team_b), ...],  # 8 partidos
        }
    """
    # Separar mejores terceros por bloque y tomar los 4 mejores de cada uno
    thirds_izq = [t for t in best_thirds if t.get("group") in bloque_izq][:4]
    thirds_der = [t for t in best_thirds if t.get("group") in bloque_der][:4]

    # Si un bloque tiene menos de 4 terceros (caso raro pero posible
    # si algún grupo solo tiene 3 equipos), completar con los disponibles
    if len(thirds_izq) < 4:
        print(f"[simulator] ⚠️  Solo {len(thirds_izq)} terceros en bloque IZQ "
              f"— rellenando con repetición")
        while len(thirds_izq) < 4:
            thirds_izq.append(thirds_izq[0] if thirds_izq else {"team": "Unknown", "points": 0, "group": bloque_izq[0]})

    if len(thirds_der) < 4:
        print(f"[simulator] ⚠️  Solo {len(thirds_der)} terceros en bloque DER "
              f"— rellenando con repetición")
        while len(thirds_der) < 4:
            thirds_der.append(thirds_der[0] if thirds_der else {"team": "Unknown", "points": 0, "group": bloque_der[0]})

    def _build_half(normal_fixtures, best3_pairs, thirds):
        """
        Construye los 8 partidos de un bloque:
            - normal_fixtures: 6 cruces (grupo_1ro, grupo_2do)
            - best3_pairs:     2 pares de índices para los 4 mejores terceros
            - thirds:          lista de 4 terceros del bloque
        """
        matches = []

        # Cruces normales: 1ro de grupo_a vs 2do de grupo_b
        for grupo_a, grupo_b in normal_fixtures:
            team_a = _get_team_by_position(group_standings, grupo_a, 1)
            team_b = _get_team_by_position(group_standings, grupo_b, 2)
            matches.append((team_a, team_b))

        # Cruces de mejores terceros del bloque
        thirds_teams = [t["team"] for t in thirds]
        for idx_a, idx_b in best3_pairs:
            t_a = thirds_teams[idx_a]
            t_b = thirds_teams[idx_b]
            matches.append((t_a, t_b))

        return matches  # siempre 8 partidos

    return {
        "izq": _build_half(R16_NORMAL_IZQ, R16_BEST3_IZQ, thirds_izq),
        "der": _build_half(R16_NORMAL_DER, R16_BEST3_DER, thirds_der),
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
    rng: np.random.Generator,
) -> list:
    """
    Simula una ronda eliminatoria dada una lista de partidos.
    En eliminatorias no hay empates: DRAW se resuelve por penales (50/50).

    RETORNA: lista de ganadores en el mismo orden que los partidos.
    (El orden es crítico para mantener el bracket correcto.)
    """
    winners = []
    for team_a, team_b in matches:
        result = simulate_match(
            team_a, team_b, modelo, scaler,
            df_rankings, hist_stats, rng=rng,
        )
        if result == "A":
            winners.append(team_a)
        elif result == "B":
            winners.append(team_b)
        else:
            # Empate → penales (50/50)
            winners.append(rng.choice([team_a, team_b]))
    return winners


# ===========================================================
# FUNCIÓN AUXILIAR: ELEGIR FINALISTA DE UN BLOQUE  (CORREGIDA)
# ===========================================================

def _pick_finalist(sf_winners: list, modelo, scaler, df_rankings, hist_stats, rng) -> Optional[str]:
    """
    De los ganadores de semifinal de un bloque, elige el representante
    que irá a la gran final.

    Con el bracket actual cada bloque produce exactamente 2 semifinalistas.
    Si por algún error hubiera solo 1, se lo pasa directamente.
    """
    if not sf_winners:
        return None
    if len(sf_winners) == 1:
        return sf_winners[0]
    # Simular el partido entre los dos semifinalistas del bloque
    result = simulate_knockout_round(
        [(sf_winners[0], sf_winners[1])],
        modelo, scaler, df_rankings, hist_stats, rng,
    )
    return result[0] if result else sf_winners[0]


# ===========================================================
# FUNCIÓN PRINCIPAL: SIMULAR EL TORNEO COMPLETO
# ===========================================================

def simulate_tournament(
    groups: dict,
    modelo,
    scaler,
    df_rankings: pd.DataFrame,
    hist_stats: dict,
    n_simulations: int = 10_000,
    seed: int = RANDOM_SEED,
    bloque_izq: List[str] = None,
    bloque_der: List[str] = None,
) -> pd.DataFrame:
    """
    Simula el torneo Mundial 2026 completo N veces (Monte Carlo).

    FLUJO POR SIMULACIÓN:
        1. Fase de grupos  → 32 clasificados (16 IZQ + 16 DER)
        2. Octavos IZQ     → 8 ganadores IZQ
           Octavos DER     → 8 ganadores DER
        3. Cuartos IZQ     → 4 ganadores IZQ
           Cuartos DER     → 4 ganadores DER
        4. Semis IZQ       → 2 finalistas IZQ
           Semis DER       → 2 finalistas DER
        5. "Final de bloque" IZQ → 1 representante IZQ
           "Final de bloque" DER → 1 representante DER
        6. Gran Final      → campeón (IZQ vs DER) ✅

    Los bloques IZQ y DER no se cruzan hasta la Gran Final.

    RETORNA:
        DataFrame con columnas:
            team, grupo, bloque,
            prob_r32, prob_r16, prob_qf, prob_sf,
            prob_final, prob_winner
    """
    if bloque_izq is None:
        bloque_izq = DEFAULT_BLOQUE_IZQ
    if bloque_der is None:
        bloque_der = DEFAULT_BLOQUE_DER

    rng = np.random.default_rng(seed)

    team_grupo  = {team: g for g, teams in groups.items() for team in teams}
    team_bloque = {
        team: ("IZQ" if team_grupo[team] in bloque_izq else "DER")
        for team in team_grupo
    }

    all_teams = list(team_grupo.keys())
    counters  = {
        team: {"r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "winner": 0}
        for team in all_teams
    }

    print(f"[simulator] Iniciando {n_simulations:,} simulaciones "
          f"(IZQ={bloque_izq} | DER={bloque_der})...")

    for sim in range(n_simulations):

        if (sim + 1) % 100 == 0:
            print(f"[simulator]   {sim + 1:,} / {n_simulations:,}...")

        # ── 1. FASE DE GRUPOS ──────────────────────────────────
        group_results   = simulate_group_stage(
            groups, modelo, scaler, df_rankings, hist_stats, rng
        )
        group_standings = group_results["standings"]
        best_thirds     = group_results["best_thirds"]

        # Contar clasificados (pos 1 y 2 de cada grupo + 8 mejores terceros)
        for group_data in group_standings.values():
            for entry in group_data:
                if entry["position"] <= 2 and entry["team"] in counters:
                    counters[entry["team"]]["r32"] += 1
        for third in best_thirds:
            if third["team"] in counters:
                counters[third["team"]]["r32"] += 1

        # ── 2. OCTAVOS — separados por bloque ─────────────────
        r16_by_bloque = build_r16_matches(
            group_standings, best_thirds, bloque_izq, bloque_der
        )
        r16_win_izq = simulate_knockout_round(
            r16_by_bloque["izq"],
            modelo, scaler, df_rankings, hist_stats, rng,
        )
        r16_win_der = simulate_knockout_round(
            r16_by_bloque["der"],
            modelo, scaler, df_rankings, hist_stats, rng,
        )
        for team in r16_win_izq + r16_win_der:
            if team in counters:
                counters[team]["r16"] += 1

        # ── 3. CUARTOS — separados por bloque ─────────────────
        def _run_round(winners: list, pairs: list) -> list:
            """Genera los partidos de una ronda a partir de ganadores y pares."""
            matches = [
                (winners[i], winners[j])
                for i, j in pairs
                if i < len(winners) and j < len(winners)
            ]
            return simulate_knockout_round(
                matches, modelo, scaler, df_rankings, hist_stats, rng
            )

        qf_win_izq = _run_round(r16_win_izq, QF_PAIRS)
        qf_win_der = _run_round(r16_win_der, QF_PAIRS)
        for team in qf_win_izq + qf_win_der:
            if team in counters:
                counters[team]["qf"] += 1

        # ── 4. SEMIFINALES — separadas por bloque ─────────────
        sf_win_izq = _run_round(qf_win_izq, SF_PAIRS)
        sf_win_der = _run_round(qf_win_der, SF_PAIRS)
        for team in sf_win_izq + sf_win_der:
            if team in counters:
                counters[team]["sf"] += 1

        # ── 5. FINAL DE BLOQUE → 1 representante por bloque ───
        # Los 2 semifinalistas de cada bloque se enfrentan para
        # decidir quién va a la Gran Final.
        finalist_izq = _pick_finalist(
            sf_win_izq, modelo, scaler, df_rankings, hist_stats, rng
        )
        finalist_der = _pick_finalist(
            sf_win_der, modelo, scaler, df_rankings, hist_stats, rng
        )

        if finalist_izq and finalist_izq in counters:
            counters[finalist_izq]["final"] += 1
        if finalist_der and finalist_der in counters:
            counters[finalist_der]["final"] += 1

        # ── 6. GRAN FINAL — IZQ vs DER ────────────────────────
        if finalist_izq and finalist_der:
            final_result = simulate_knockout_round(
                [(finalist_izq, finalist_der)],
                modelo, scaler, df_rankings, hist_stats, rng,
            )
            if final_result and final_result[0] in counters:
                counters[final_result[0]]["winner"] += 1

    # ── Convertir contadores a probabilidades ─────────────────
    results = [
        {
            "team":        team,
            "grupo":       team_grupo.get(team, "?"),
            "bloque":      team_bloque.get(team, "?"),
            "prob_r32":    counters[team]["r32"]    / n_simulations,
            "prob_r16":    counters[team]["r16"]    / n_simulations,
            "prob_qf":     counters[team]["qf"]     / n_simulations,
            "prob_sf":     counters[team]["sf"]     / n_simulations,
            "prob_final":  counters[team]["final"]  / n_simulations,
            "prob_winner": counters[team]["winner"] / n_simulations,
        }
        for team in all_teams
    ]

    df_results = (
        pd.DataFrame(results)
        .sort_values("prob_winner", ascending=False)
        .reset_index(drop=True)
    )

    print("[simulator] ✅ Simulación completada.")
    return df_results


# ===========================================================
# FUNCIÓN: RESUMEN DE RESULTADOS
# ===========================================================

def get_results_summary(df_results: pd.DataFrame) -> None:
    """Imprime un resumen legible de los resultados, separado por bloque."""
    print("\n" + "=" * 70)
    print("  PREDICCIONES MUNDIAL 2026")
    print("  (Monte Carlo · fixture oficial · bloques IZQ/DER separados)")
    print("=" * 70)

    for bloque_tag, bloque_label in [
        ("IZQ", "BLOQUE IZQ — grupos A–F"),
        ("DER", "BLOQUE DER — grupos G–L"),
    ]:
        sub = (
            df_results[df_results["bloque"] == bloque_tag]
            if "bloque" in df_results.columns
            else df_results
        )

        print(f"\n  {bloque_label}")
        print(
            f"  {'Pos':>4}  {'Equipo':<22} {'Clasifica':>10} {'Octavos':>8} "
            f"{'Cuartos':>8} {'Semis':>7} {'Final':>7} {'Campeón':>9}"
        )
        print("  " + "-" * 78)

        for i, (_, row) in enumerate(sub.iterrows()):
            print(
                f"  {i + 1:>4}  {row['team']:<22} "
                f"{row.get('prob_r32', 0):>9.1%} "
                f"{row['prob_r16']:>7.1%} "
                f"{row['prob_qf']:>7.1%} "
                f"{row['prob_sf']:>6.1%} "
                f"{row['prob_final']:>6.1%} "
                f"{row['prob_winner']:>8.1%}"
            )