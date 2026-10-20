"""
Feature engineering SIN fuga de información.

Regla de oro: cada feature de un partido debe poder calcularse ANTES de
que empiece el torneo. Por eso todo se construye en dos pasos:

1. `build_team_snapshots`: una fila por (equipo, año) con el estado del
   equipo el día de inicio del Mundial: Elo, ranking FIFA, tendencia,
   historial en Mundiales anteriores. Nunca se mira el torneo en curso.
2. `build_match_features`: para cada partido, la DIFERENCIA entre las
   instantáneas de los dos equipos. Los goles del partido solo se usan
   para construir la etiqueta (y), jamás como variable.

El mismo código construye las features de 2026 para el simulador, así
que entrenamiento y predicción ven exactamente las mismas variables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Elo mundialista
# ---------------------------------------------------------------------------
def _goal_multiplier(goal_diff: int) -> float:
    """Multiplicador de margen de World Football Elo: golear pesa más."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def compute_pretournament_elo(matches: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Elo de cada selección al INICIO de cada Mundial.

    Se recorre la historia en orden cronológico. Antes de cada torneo se
    toma la instantánea (eso es lo que verá el modelo) y solo después se
    procesan los partidos de ese torneo para actualizar ratings.
    """
    ratings: dict[str, float] = {}
    snapshots = []
    for year in sorted(years):
        # Regresión a la media entre torneos: un equipo cambia mucho en 4 años.
        for team in ratings:
            ratings[team] += config.ELO_REGRESSION * (config.ELO_INIT - ratings[team])

        tour = matches[matches["year"] == year]
        teams = set(tour["home_team"]) | set(tour["away_team"])
        for team in teams | set(ratings):
            snapshots.append({"team": team, "year": year,
                              "elo": ratings.get(team, config.ELO_INIT)})

        hosts = set(config.HOSTS.get(year, []))
        for m in tour.itertuples():
            ra = ratings.get(m.home_team, config.ELO_INIT)
            rb = ratings.get(m.away_team, config.ELO_INIT)
            ha = config.ELO_HOST_BONUS if m.home_team in hosts else 0.0
            hb = config.ELO_HOST_BONUS if m.away_team in hosts else 0.0
            expected_a = 1.0 / (1.0 + 10 ** (-((ra + ha) - (rb + hb)) / 400))
            gd = int(m.home_score - m.away_score)
            score_a = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
            delta = config.ELO_K * _goal_multiplier(gd) * (score_a - expected_a)
            ratings[m.home_team] = ra + delta
            ratings[m.away_team] = rb - delta
    return pd.DataFrame(snapshots).drop_duplicates(["team", "year"])


def _tournament_k(tournament: str) -> float:
    """Peso K según la importancia del partido (esquema World Football Elo)."""
    t = str(tournament).lower()
    if "fifa world cup" in t and "qualification" not in t:
        return 60.0
    if "qualification" in t:
        return 40.0
    if t == "friendly":
        return 20.0
    if any(k in t for k in ("euro", "copa américa", "copa america", "african cup",
                            "asian cup", "gold cup", "confederations")):
        return 50.0
    return 30.0          # Nations League y otros torneos oficiales


def compute_international_elo(results: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Elo con TODOS los partidos internacionales (fuente externa opcional).

    Esta es la mejora de más impacto: el Elo solo-mundialista no ve la
    Eurocopa, la Copa América ni las eliminatorias. Un equipo que ganó la
    Euro 2024 (España) aparecía con Elo mediocre por sus Mundiales 2014–22.

    Se toma la instantánea el día antes de cada Mundial; los partidos del
    propio Mundial solo afectan a instantáneas de torneos posteriores.
    """
    results = results.sort_values("date")
    cutoffs = sorted((pd.Timestamp(config.WORLDCUP_START_DATES[y]), y) for y in years)
    ratings: dict[str, float] = {}
    snapshots, ci = [], 0
    for m in results.itertuples():
        while ci < len(cutoffs) and m.date >= cutoffs[ci][0]:
            snapshots += [{"team": t, "year": cutoffs[ci][1], "elo": r} for t, r in ratings.items()]
            ci += 1
        ra = ratings.get(m.home_team, config.ELO_INIT)
        rb = ratings.get(m.away_team, config.ELO_INIT)
        home_adv = 0.0 if m.neutral else config.ELO_HOST_BONUS
        expected_a = 1.0 / (1.0 + 10 ** (-((ra + home_adv) - rb) / 400))
        gd = int(m.home_score - m.away_score)
        score_a = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        delta = _tournament_k(m.tournament) * _goal_multiplier(gd) * (score_a - expected_a)
        ratings[m.home_team] = ra + delta
        ratings[m.away_team] = rb - delta
    for ts, y in cutoffs[ci:]:                       # torneos posteriores al último dato
        snapshots += [{"team": t, "year": y, "elo": r} for t, r in ratings.items()]
    return pd.DataFrame(snapshots, columns=["team", "year", "elo"])


# ---------------------------------------------------------------------------
# Ranking FIFA
# ---------------------------------------------------------------------------
def _latest_before(rankings: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Última publicación del ranking anterior a `cutoff`, una fila por equipo."""
    prev = rankings[rankings["rank_date"] <= cutoff]
    if prev.empty:
        return prev
    return prev.sort_values("rank_date").groupby("team").tail(1).set_index("team")


def ranking_snapshot(rankings: pd.DataFrame, year: int) -> pd.DataFrame:
    """Ranking, puntos normalizados y tendencia de 12 meses al inicio del torneo."""
    start = pd.Timestamp(config.WORLDCUP_START_DATES[year])
    now = _latest_before(rankings, start)
    if now.empty:
        return pd.DataFrame(columns=["rank", "points_z", "momentum", "confederation"])
    year_ago = _latest_before(rankings, start - pd.DateOffset(months=12))
    out = now[["rank", "points_z", "confederation"]].copy()
    # momentum > 0 → el equipo llega en ascenso (ganó puntos relativos en el año)
    out["momentum"] = (out["points_z"] - year_ago["points_z"].reindex(out.index)).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Historial en Mundiales anteriores
# ---------------------------------------------------------------------------
def history_snapshot(matches: pd.DataFrame, year: int) -> pd.DataFrame:
    """Rendimiento en Mundiales ESTRICTAMENTE anteriores a `year`."""
    past = matches[matches["year"] < year]
    long = pd.concat([
        past[["year", "home_team", "depth"]].rename(columns={"home_team": "team"}),
        past[["year", "away_team", "depth"]].rename(columns={"away_team": "team"}),
    ])
    best = long.groupby(["team", "year"])["depth"].max().reset_index()
    rows = []
    for team, d in best.groupby("team"):
        d = d.sort_values("year")
        rows.append({
            "team": team,
            "hist_stage": d["depth"].mean(),
            "recent_stage": d["depth"].tail(2).mean(),
            "experience": np.log1p(len(d)),
        })
    return pd.DataFrame(rows, columns=["team", "hist_stage", "recent_stage",
                                       "experience"]).set_index("team")


# ---------------------------------------------------------------------------
# Instantánea por (equipo, año)
# ---------------------------------------------------------------------------
def build_team_snapshots(matches: pd.DataFrame, rankings: pd.DataFrame,
                         teams_by_year: dict[int, list[str]],
                         intl_results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una fila por equipo y torneo con todo lo conocido antes del torneo.

    Si se pasa `intl_results` (todos los partidos internacionales), el Elo
    se calcula con esa fuente; si no, con los partidos de Mundiales.
    """
    years = sorted(teams_by_year)
    elo_df = (compute_international_elo(intl_results, years) if intl_results is not None
              else compute_pretournament_elo(matches, years))
    elo = elo_df.drop_duplicates(["team", "year"], keep="last").set_index(["team", "year"])["elo"]
    # Confederación fija por equipo (la del último ranking disponible)
    confed = rankings.sort_values("rank_date").groupby("team")["confederation"].last()

    rows = []
    for year in years:
        rk = ranking_snapshot(rankings, year)
        hist = history_snapshot(matches, year)
        hosts = set(config.HOSTS[year])
        for team in teams_by_year[year]:
            known = team in rk.index
            rows.append({
                "team": team,
                "year": year,
                "elo": elo.get((team, year), config.ELO_INIT),
                "rank_known": int(known),
                "rank": float(rk.at[team, "rank"]) if known else np.nan,
                "points_z": float(rk.at[team, "points_z"]) if known else np.nan,
                "momentum": float(rk.at[team, "momentum"]) if known else 0.0,
                "hist_stage": hist["hist_stage"].get(team, 0.0),
                "recent_stage": hist["recent_stage"].get(team, 0.0),
                "experience": hist["experience"].get(team, 0.0),
                "is_host": int(team in hosts),
                "home_conf": int(confed.get(team) == config.HOST_CONFEDERATION[year]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Features de partido (diferencias A − B)
# ---------------------------------------------------------------------------
def pair_features(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Features de enfrentamiento a partir de dos tablas alineadas de snapshots.

    Si alguno de los dos no tiene ranking FIFA (antes de 1994), las
    diferencias de ranking valen 0 = "sin información", en vez de inventar
    un ranking 75 para todos como hacía la versión anterior.
    """
    both_ranked = (a["rank_known"].values * b["rank_known"].values).astype(bool)
    out = pd.DataFrame(index=a.index)
    out["d_elo"] = (a["elo"].values - b["elo"].values) / 100.0
    out["d_log_rank"] = np.where(both_ranked, np.log(b["rank"].values) - np.log(a["rank"].values), 0.0)
    out["d_points_z"] = np.where(both_ranked, a["points_z"].values - b["points_z"].values, 0.0)
    out["d_momentum"] = np.where(both_ranked, a["momentum"].values - b["momentum"].values, 0.0)
    out["d_hist_stage"] = a["hist_stage"].values - b["hist_stage"].values
    out["d_recent_stage"] = a["recent_stage"].values - b["recent_stage"].values
    out["d_experience"] = a["experience"].values - b["experience"].values
    out["d_host"] = a["is_host"].values - b["is_host"].values
    out["d_home_conf"] = a["home_conf"].values - b["home_conf"].values
    return out[config.ALL_PAIR_FEATURES]


def build_match_dataset(matches: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Dataset de entrenamiento: una fila por partido, features + etiqueta.

    Etiqueta `outcome` (óptica del equipo A = local en la tabla):
        2 = gana, 1 = empate (en 90/120 min), 0 = pierde.
    Los penales NO cuentan como victoria: el simulador los resuelve aparte.
    """
    snap = snapshots.set_index(["team", "year"])
    a = snap.loc[list(zip(matches["home_team"], matches["year"]))].reset_index(drop=True)
    b = snap.loc[list(zip(matches["away_team"], matches["year"]))].reset_index(drop=True)
    X = pair_features(a, b)
    gd = (matches["home_score"] - matches["away_score"]).values
    meta = matches[["year", "date", "round", "is_knockout", "home_team",
                    "away_team"]].reset_index(drop=True)
    meta["outcome"] = np.select([gd > 0, gd == 0], [2, 1], 0)
    meta["both_ranked"] = (a["rank_known"] * b["rank_known"]).values
    return pd.concat([meta, X], axis=1)


def symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    """Duplica cada partido desde la óptica del rival (features con signo
    invertido, gana↔pierde). Elimina el sesgo de "quién aparece primero"
    que en la versión anterior se colaba como `is_home`."""
    flipped = df.copy()
    flipped[config.ALL_PAIR_FEATURES] = -flipped[config.ALL_PAIR_FEATURES]
    flipped["outcome"] = 2 - flipped["outcome"]
    return pd.concat([df, flipped], ignore_index=True)
