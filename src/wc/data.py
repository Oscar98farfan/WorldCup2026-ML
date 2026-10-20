"""
Carga y limpieza de las fuentes de datos.

Fuente 1  matches.csv   → partidos de Mundiales 1930–2022
Fuente 2  rankings.csv  → ranking FIFA mensual 1992–2026
Extra     groups_2026.csv, data/external/wc2026_actual.csv

El paso más importante de la integración es NORMALIZAR NOMBRES:
cada fuente escribe distinto a varios países ("United States" vs "USA",
"Türkiye" vs "Turkey") y además hay selecciones con sucesor histórico
(West Germany → Germany, Zaire → Congo DR). Si esto no se resuelve, el
merge falla en silencio y el equipo recibe valores por defecto.
"""
from __future__ import annotations

import pandas as pd

from . import config

# Nombre en cualquier fuente → nombre canónico (el del ranking FIFA).
CANONICAL_NAMES = {
    "United States": "USA",
    "Türkiye": "Turkey",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "South Korea": "Korea Republic",
    "Iran": "IR Iran",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Curaçao": "Curacao",
    # Sucesores históricos: el historial mundialista pasa al sucesor FIFA.
    "West Germany": "Germany",
    "Zaire": "Congo DR",
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czechia",
    "Yugoslavia": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Dutch East Indies": "Indonesia",
}


def canonical(name: str) -> str:
    """Devuelve el nombre canónico de una selección."""
    name = str(name).strip()
    return CANONICAL_NAMES.get(name, name)


def load_matches() -> pd.DataFrame:
    """Partidos históricos, una fila por partido, con nombres canónicos."""
    df = pd.read_csv(config.RAW_DIR / "matches.csv", low_memory=False)
    df = df.rename(columns={"Year": "year", "Round": "round", "Date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["home_team", "away_team"]:
        df[col] = df[col].map(canonical)
    for col in ["home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score", "date"])
    df["is_knockout"] = (~df["round"].isin(config.GROUP_ROUNDS)).astype(int)
    df["depth"] = df["round"].map(config.ROUND_DEPTH).fillna(0).astype(int)
    keep = ["date", "year", "round", "depth", "is_knockout", "home_team",
            "away_team", "home_score", "away_score", "home_penalty",
            "away_penalty"]
    return df[keep].sort_values("date").reset_index(drop=True)


def load_rankings() -> pd.DataFrame:
    """Ranking FIFA con fecha parseada y puntos normalizados por fecha.

    Los puntos FIFA cambiaron de escala varias veces (1993, 1999, 2006,
    2018), así que los puntos crudos NO son comparables entre épocas.
    Por eso se calcula `points_z`: z-score de los puntos dentro de cada
    publicación del ranking.
    """
    df = pd.read_csv(config.RAW_DIR / "rankings.csv")
    df["rank_date"] = pd.to_datetime(df["rank_date"], dayfirst=True, errors="coerce")
    df["team"] = df["country_full"].map(canonical)
    df = df.dropna(subset=["rank", "rank_date"])
    df = _repair_duplicate_names(df)
    grp = df.groupby("rank_date")["total_points"]
    df["points_z"] = (df["total_points"] - grp.transform("mean")) / grp.transform("std")
    return df[["rank_date", "team", "rank", "total_points", "points_z", "confederation"]]


def load_international_results() -> pd.DataFrame | None:
    """Fuente externa OPCIONAL: todos los partidos internacionales desde 1872.

    Formato esperado (dataset público "International football results",
    github.com/martj42/international_results, también en Kaggle):
        date, home_team, away_team, home_score, away_score,
        tournament, city, country, neutral
    Guardar como data/raw/international_results.csv. Si no existe, el
    pipeline usa el Elo calculado solo con partidos de Mundiales.
    """
    path = config.RAW_DIR / "international_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["home_team", "away_team"]:
        df[col] = df[col].map(canonical)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df.dropna(subset=["date", "home_score", "away_score"])


def _repair_duplicate_names(df: pd.DataFrame, max_gap: float = 100.0) -> pd.DataFrame:
    """Corrige filas con el nombre equivocado en el ranking.

    En las publicaciones de dic-2025 y abr-2026 del CSV hay equipos
    repetidos: p. ej. dos "Korea DPR" (puestos 25 y 118), y falta
    "Korea Republic". El puesto 25 es Corea del Sur. Sin esta corrección
    Corea del Sur usaría un ranking de 2024 y Nueva Zelanda podría tomar
    el puesto de otro país.

    Regla: en cada grupo duplicado, la fila cuyos puntos están más cerca
    de los de ese equipo en la publicación anterior se queda con el nombre;
    las demás se reasignan al equipo AUSENTE en esa fecha con puntos
    previos más parecidos (si la diferencia es < `max_gap`), o se descartan.
    """
    df = df.sort_values(["rank_date", "rank"]).reset_index(drop=True)
    dates = sorted(df["rank_date"].unique())
    drop = []
    for i, date in enumerate(dates[1:], start=1):
        cur = df[df["rank_date"] == date]
        dups = cur[cur.duplicated("team", keep=False)]
        if dups.empty:
            continue
        prev = df[df["rank_date"] == dates[i - 1]].drop_duplicates("team").set_index("team")["total_points"]
        missing = prev.drop(cur["team"], errors="ignore")
        for team, rows in dups.groupby("team"):
            ref = prev.get(team, rows["total_points"].max())
            keep = (rows["total_points"] - ref).abs().idxmin()
            for idx in rows.index.drop(keep):
                gaps = (missing - df.at[idx, "total_points"]).abs()
                if len(gaps) and gaps.min() < max_gap:
                    new_name = gaps.idxmin()
                    df.at[idx, "team"] = new_name
                    missing = missing.drop(new_name)
                else:
                    drop.append(idx)
    return df.drop(index=drop)


def load_groups_2026() -> dict[str, list[str]]:
    """Grupos del Mundial 2026 como {'A': [...], ...} con nombres canónicos."""
    path = config.RAW_DIR / "groups_2026.csv"
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:          # el archivo original está en Latin-1
        df = pd.read_csv(path, encoding="cp1252")
    df["team"] = df["team"].map(canonical)
    return {g: list(d["team"]) for g, d in df.groupby("group")}


def load_actual_2026() -> pd.DataFrame:
    """Resultado real del Mundial 2026 (ronda máxima alcanzada por equipo)."""
    df = pd.read_csv(config.EXTERNAL_DIR / "wc2026_actual.csv")
    df["team"] = df["team"].map(canonical)
    return df


def load_actual_2026_knockout() -> pd.DataFrame:
    """Partidos de eliminación directa reales del Mundial 2026."""
    df = pd.read_csv(config.EXTERNAL_DIR / "wc2026_knockout.csv")
    for col in ["team_a", "team_b", "winner"]:
        df[col] = df[col].map(canonical)
    return df
