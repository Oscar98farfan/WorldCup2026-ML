"""
===========================================================
src/feature_engineering.py
===========================================================

PROPÓSITO:
    Contiene TODA la lógica de construcción de features.
    Transforma los datos crudos en las variables que el
    modelo necesita para aprender.

¿POR QUÉ EN UN MÓDULO SEPARADO?
    La construcción de features es la parte más crítica
    del proyecto. Al tenerla aquí:
    - 01_etl.py la usa para construir el dataset de entrenamiento
    - 03_simulate.py la usa para construir features de partidos 2026
    - No hay duplicación de código entre scripts

CONTIENE:
    CONSTANTES:
        WORLDCUP_START_DATES  → fechas de inicio de cada mundial
        ROUND_MAP             → mapeo de nombre de ronda a número
        COUNTRY_NAME_MAP      → normalización de nombres de países

    FUNCIONES:
        get_ranking_before_worldcup() → ranking FIFA previo al torneo
        build_match_features()        → features por partido
        build_historical_features()   → estadísticas históricas del equipo
        build_full_dataset()          → pipeline completo

DECISIONES DE DISEÑO IMPORTANTES:
    1. Para el ranking FIFA siempre buscamos la fecha MÁS RECIENTE
       anterior al inicio del mundial. Esto evita data leakage.
    2. Las features históricas de un equipo para el año X solo
       usan datos de mundiales anteriores a X. Misma razón.
    3. Attendance fue eliminada porque tiene correlación espuria
       con la ronda (las finales siempre tienen más público)
       y no es usable para predicción futura.

AUTOR: Oscar Daniel Farfan Juanias
===========================================================
"""

import pandas as pd
import numpy as np
from typing import Optional


# ===========================================================
# CONSTANTES
# ===========================================================

# Fechas aproximadas de inicio de cada mundial.
# Se usan como "fecha de corte" para buscar el ranking FIFA
# más reciente ANTES de que empiece el torneo.
# Importante: el ranking FIFA existe desde diciembre 1992,
# por lo que mundiales anteriores a 1994 usarán valores por defecto.
WORLDCUP_START_DATES = {
    1930: "1930-07-13", 1934: "1934-05-27", 1938: "1938-06-04",
    1950: "1950-06-24", 1954: "1954-06-16", 1958: "1958-06-08",
    1962: "1962-05-30", 1966: "1966-07-11", 1970: "1970-05-31",
    1974: "1974-06-13", 1978: "1978-06-01", 1982: "1982-06-13",
    1986: "1986-05-31", 1990: "1990-06-08", 1994: "1994-06-17",
    1998: "1998-06-10", 2002: "2002-05-31", 2006: "2006-06-09",
    2010: "2010-06-11", 2014: "2014-06-12", 2018: "2018-06-14",
    2022: "2022-11-20", 2026: "2026-06-11",
}

# Mapeo de nombre de ronda a valor numérico.
# Permite comparar rondas y calcular qué tan lejos llegó un equipo.
# 0 = grupos, 1 = octavos, 2 = cuartos, 3 = semis/3er puesto, 4 = final
ROUND_MAP = {
    "Group stage":        0,
    "Round of 16":        1,
    "Quarter-finals":     2,
    "Semi-finals":        3,
    "Third-place match":  3,
    "Final":              4,
}

# Normalización de nombres de países entre los dos datasets.
# matches.csv usa un nombre, rankings.csv puede usar otro.
# Clave: nombre en matches.csv → valor: nombre en rankings.csv
# COUNTRY_NAME_MAP = {
#     "USA":              "USA",
#     "South Korea":      "Korea Republic",
#     "Korea Republic":   "Korea Republic",
#     "Iran":             "IR Iran",
#     "IR Iran":          "IR Iran",
#     "Ivory Coast":      "Côte d'Ivoire",
#     "Côte d'Ivoire":    "CÃ´te d'Ivoire",
#     "Czech Republic":   "Czechia",
#     "Ireland":          "Republic of Ireland",
#     "Cabo Verde":       "Cabo Verde",
#     "Congo DR":         "Congo DR",
# }

COUNTRY_NAME_MAP = {
    "USA":                    "USA",
    "South Korea":            "Korea Republic",
    "Korea Republic":         "Korea Republic",
    "Iran":                   "IR Iran",
    "IR Iran":                "IR Iran",
    "Ivory Coast":            "CÃ´te d'Ivoire",
    "Côte d'Ivoire":          "CÃ´te d'Ivoire",  # fix encoding
    "Czech Republic":         "Czechia",
    "Ireland":                "Republic of Ireland",
    "Cabo Verde":             "Cabo Verde",
    "Congo DR":               "Congo DR",
    "Türkiye":                "Turkey",           # fix nuevo
    "Turkey":                 "Turkey",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Curacao":                "Curacao",
}

# Features que usa el modelo — definidas aquí para que todos
# los scripts usen exactamente la misma lista.
# Si agregas o quitas una feature, solo la cambias aquí.
MODEL_FEATURES = [
    # --- Del partido ---
    # goals: goles anotados por el equipo en ese partido
    # penalty: penales anotados (no tiros desde el punto penal)
    "goals",
    "penalty",

    # --- Del ranking FIFA (integrado en ETL) ---
    # rank: posición FIFA del equipo antes del mundial
    # points: puntos FIFA acumulados
    # rank_diff: diferencia de posición con el rival (negativo = rival mejor)
    # points_diff: diferencia de puntos con el rival
    # rank_ratio: ratio de posiciones (captura la brecha relativa)
    # same_conf: 1 si ambos equipos son de la misma confederación
    # is_home: 1 si el equipo juega como "local" en el partido
    # is_host: 1 si el equipo es el anfitrión del mundial
    # rank_tier: categoría del equipo (3=élite, 2=fuerte, 1=medio, 0=débil)
    "rank", "points", "rank_diff", "points_diff",
    "rank_ratio", "same_conf", "is_home", "is_host", "rank_tier",

    # --- Históricas del equipo ---
    # hist_mundiales: cuántos mundiales jugó antes de este
    # hist_max_ronda: mejor ronda histórica alcanzada (0-4)
    # hist_avg_ronda: ronda promedio en todos los mundiales previos
    # hist_veces_qf: veces que llegó a cuartos o más
    # hist_veces_sf: veces que llegó a semis o más
    # hist_veces_final: veces que llegó a la final
    # hist_avg_goles: promedio de goles histórico en mundiales
    # hist_tendencia: ronda promedio de los ÚLTIMOS 2 mundiales
    #                 (captura si el equipo está en ascenso o descenso)
    "hist_mundiales", "hist_max_ronda", "hist_avg_ronda",
    "hist_veces_qf", "hist_veces_sf", "hist_veces_final",
    "hist_avg_goles", "hist_tendencia",
]

# Target del modelo
# gana_partido = 1 si el equipo ganó ese partido
# gana_partido = 0 si perdió o empató
# Se usa para predecir partido a partido en el simulador
TARGET = "gana_partido"


# ===========================================================
# FUNCIÓN: RANKING FIFA PREVIO AL MUNDIAL
# ===========================================================

def get_ranking_before_worldcup(
    df_rankings: pd.DataFrame,
    team: str,
    year: int
) -> dict:
    """
    Obtiene el ranking FIFA de un equipo justo ANTES del mundial.

    ¿POR QUÉ "ANTES" Y NO "DURANTE"?
        Si usáramos el ranking durante o después del mundial,
        estaríamos usando información del futuro para predecir.
        Esto se llama data leakage y hace que el modelo parezca
        mejor de lo que realmente es.

    EJEMPLO:
        Para predecir el Mundial 2018, buscamos el último
        ranking disponible anterior al 14/06/2018 (inicio del torneo).

    PARÁMETROS:
        df_rankings → DataFrame del ranking FIFA (ya cargado)
        team        → nombre del equipo (como aparece en matches.csv)
        year        → año del mundial a predecir

    RETORNA:
        dict con: rank, points, rank_change, confederation
        Si no hay datos disponibles, retorna valores por defecto:
        rank=75 (posición media), points=800 (valor conservador)
    """
    # Normalizar nombre del equipo para buscar en el ranking
    team_in_ranking = COUNTRY_NAME_MAP.get(team, team)

    # Fecha de corte: inicio del mundial
    cutoff = pd.Timestamp(
        WORLDCUP_START_DATES.get(year, f"{year}-06-01")
    )

    # Filtrar: mismo equipo + fechas anteriores al mundial
    mask = (
        (df_rankings["country_full"] == team_in_ranking) &
        (df_rankings["rank_date"] <= cutoff)
    )
    team_data = df_rankings[mask]

    if team_data.empty:
        # Sin datos en el ranking FIFA para este equipo y año.
        # Ocurre principalmente en mundiales pre-1994 (el ranking
        # FIFA existe desde diciembre 1992).
        # Valores por defecto: posición 75 (equipos de nivel medio),
        # puntos 800 (valor conservador para esa época)
        return {
            "rank":          75,
            "points":        800.0,
            "rank_change":   0,
            "confederation": "UNK",
        }

    # Tomar el registro MÁS RECIENTE antes del mundial
    latest = team_data.loc[team_data["rank_date"].idxmax()]

    return {
        "rank":          int(latest["rank"]),
        "points":        float(latest["total_points"]),
        "rank_change":   float(latest.get("rank_change", 0)),
        "confederation": str(latest.get("confederation", "UNK")),
    }


# ===========================================================
# FUNCIÓN: FEATURES DE PARTIDO
# ===========================================================

def build_match_features(
    df_matches: pd.DataFrame,
    df_rankings: pd.DataFrame
) -> pd.DataFrame:
    """
    Transforma el dataset de partidos en formato por equipo.

    ENTRADA:
        df_matches  → DataFrame de partidos (una fila = un partido)
        df_rankings → DataFrame del ranking FIFA

    SALIDA:
        DataFrame donde cada fila es UN EQUIPO en UN PARTIDO.
        Un partido genera DOS filas: una para local y otra para visitante.

    ¿POR QUÉ DUPLICAR LAS FILAS?
        El modelo aprende desde la perspectiva de un equipo,
        no desde la perspectiva del partido. Para que aprenda
        "cómo juega Brasil" necesita ver a Brasil tanto como
        local como visitante.

    NUEVO TARGET — gana_partido:
        A diferencia del proyecto anterior (llega_lejos),
        ahora predecimos si el equipo GANÓ ese partido específico.
        Esto permite usar el modelo dentro del simulador para
        predecir el resultado de cualquier enfrentamiento.

    FEATURES AGREGADAS:
        rank, points: del equipo en ese mundial
        rank_diff: diferencia con el rival (positivo = rival mejor)
        points_diff: diferencia de puntos con el rival
        rank_ratio: ratio de posiciones
        same_conf: misma confederación que el rival
        is_home: juega como local
        is_host: es el anfitrión del mundial
        rank_tier: categoría del equipo
        gana_partido: target (1=ganó, 0=no ganó)
        round_num: número de la ronda (para features históricas)
    """
    df = df_matches.copy()

    # Limpiar columnas numéricas
    num_cols = [
        "home_score", "away_score", "home_xg", "away_xg",
        "home_penalty", "away_penalty", "Attendance", "Year"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Mapear rondas a números
    df["round_num"] = df["Round"].map(ROUND_MAP).fillna(0)

    rows = []

    for _, row in df.iterrows():
        year = int(row["Year"])

        # Obtener ranking de ambos equipos antes del mundial
        home_rank_data = get_ranking_before_worldcup(
            df_rankings, row["home_team"], year
        )
        away_rank_data = get_ranking_before_worldcup(
            df_rankings, row["away_team"], year
        )

        # Determinar ganador del partido
        # Si hay penales, los penales deciden el ganador
        home_won = _determine_winner(
            row["home_score"], row["away_score"],
            row.get("home_penalty", 0), row.get("away_penalty", 0)
        )

        # Base común para ambas filas
        base = {
            "year":      year,
            "round_num": row["round_num"],
            "round":     row["Round"],
            "host":      row["Host"],
        }

        # --- Fila del equipo LOCAL ---
        rows.append({
            **base,
            "team":        row["home_team"],
            "goals":       row["home_score"],
            "penalty":     row.get("home_penalty", 0),
            # Ranking del equipo local
            "rank":        home_rank_data["rank"],
            "points":      home_rank_data["points"],
            "confederation": home_rank_data["confederation"],
            # Diferencias respecto al rival
            # Positivo = el rival tiene mejor ranking (número menor)
            "rank_diff":   home_rank_data["rank"] - away_rank_data["rank"],
            "points_diff": home_rank_data["points"] - away_rank_data["points"],
            # rank_ratio: si < 1, el local es mejor que el visitante
            "rank_ratio":  away_rank_data["rank"] / (home_rank_data["rank"] + 1e-6),
            "same_conf":   int(home_rank_data["confederation"] == away_rank_data["confederation"]),
            "is_home":     1,
            "is_host":     int(row["home_team"] == row["Host"]),
            "rank_tier":   _rank_to_tier(home_rank_data["rank"]),
            # Target: ¿ganó el equipo local?
            "gana_partido": int(home_won == "home"),
        })

        # --- Fila del equipo VISITANTE ---
        rows.append({
            **base,
            "team":        row["away_team"],
            "goals":       row["away_score"],
            "penalty":     row.get("away_penalty", 0),
            # Ranking del equipo visitante
            "rank":        away_rank_data["rank"],
            "points":      away_rank_data["points"],
            "confederation": away_rank_data["confederation"],
            # Diferencias desde la perspectiva del visitante (invertidas)
            "rank_diff":   away_rank_data["rank"] - home_rank_data["rank"],
            "points_diff": away_rank_data["points"] - home_rank_data["points"],
            "rank_ratio":  home_rank_data["rank"] / (away_rank_data["rank"] + 1e-6),
            "same_conf":   int(away_rank_data["confederation"] == home_rank_data["confederation"]),
            "is_home":     0,
            "is_host":     int(row["away_team"] == row["Host"]),
            "rank_tier":   _rank_to_tier(away_rank_data["rank"]),
            # Target: ¿ganó el equipo visitante?
            "gana_partido": int(home_won == "away"),
        })

    result = pd.DataFrame(rows)
    print(f"[feature_engineering] build_match_features: {result.shape}")
    return result


def _determine_winner(h_score, a_score, h_pen, a_pen) -> str:
    """
    Determina el ganador de un partido considerando penales.

    RETORNA: 'home', 'away', o 'draw'
    """
    h_score = float(h_score or 0)
    a_score = float(a_score or 0)
    h_pen   = float(h_pen   or 0)
    a_pen   = float(a_pen   or 0)

    if h_score > a_score:
        return "home"
    elif a_score > h_score:
        return "away"
    else:
        # Empate en tiempo regular → revisar penales
        if h_pen > a_pen:
            return "home"
        elif a_pen > h_pen:
            return "away"
        return "draw"


def _rank_to_tier(rank: int) -> int:
    """
    Convierte una posición FIFA en una categoría de 0 a 3.

    ¿POR QUÉ USAR TIERS ADEMÁS DEL RANK NUMÉRICO?
        El rank numérico puede ser muy sensible (posición 10 vs 11).
        Los tiers agrupan equipos en categorías más estables:
        - 3 (élite): Top 10 del mundo
        - 2 (fuerte): Top 11-30
        - 1 (medio): Top 31-60
        - 0 (débil): Fuera del Top 60
    """
    if rank <= 10:   return 3
    elif rank <= 30: return 2
    elif rank <= 60: return 1
    else:            return 0


# ===========================================================
# FUNCIÓN: FEATURES HISTÓRICAS DEL EQUIPO
# ===========================================================

def build_historical_features(df_matches_by_team: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las estadísticas históricas de cada equipo en mundiales previos.

    ENTRADA:
        df_matches_by_team → resultado de build_match_features()
                             (una fila = un equipo en un partido)

    ¿POR QUÉ FEATURES HISTÓRICAS?
        El ranking FIFA mide la fuerza del equipo HOY.
        Las features históricas miden su TRAYECTORIA:
        - ¿Es un equipo con tradición mundialista? (hist_mundiales)
        - ¿Suele llegar lejos? (hist_avg_ronda)
        - ¿Está en ascenso o en descenso? (hist_tendencia)

    REGLA CRÍTICA ANTI-DATA-LEAKAGE:
        Para predecir el mundial del año X, solo usamos datos
        de mundiales ANTERIORES a X. Si un equipo llegó a semis
        en 2018, esa información NO se usa para predecir 2018,
        pero SÍ para predecir 2022 y 2026.

    FEATURES CALCULADAS:
        hist_mundiales:   número de mundiales jugados antes de este
        hist_max_ronda:   mejor ronda alcanzada históricamente (0-4)
        hist_avg_ronda:   promedio de rondas en todos los mundiales previos
        hist_veces_qf:    cuántas veces llegó a cuartos o más
        hist_veces_sf:    cuántas veces llegó a semis o más
        hist_veces_final: cuántas veces llegó a la final
        hist_avg_goles:   promedio de goles por partido en mundiales previos
        hist_tendencia:   promedio de ronda en los ÚLTIMOS 2 mundiales
                          (si es mayor que hist_avg_ronda = equipo en ascenso)

    RETORNA:
        DataFrame con una fila por (team, year) con las features históricas
    """

    # Primero: ronda máxima por equipo y año (cuánto llegó en cada mundial)
    ronda_por_anio = (
        df_matches_by_team
        .groupby(["team", "year"])
        .agg(
            max_round  = ("round_num", "max"),
            avg_goals  = ("goals", "mean"),
        )
        .reset_index()
    )

    records = []
    teams = df_matches_by_team["team"].unique()
    years = sorted(df_matches_by_team["year"].unique())

    for team in teams:
        team_history = ronda_por_anio[
            ronda_por_anio["team"] == team
        ].sort_values("year")

        for year in years:
            # Solo procesar si el equipo participó en este año
            if year not in team_history["year"].values:
                continue

            # Solo datos de mundiales ANTERIORES a este año
            history_before = team_history[team_history["year"] < year]

            if history_before.empty:
                # Primera participación del equipo → sin historial
                records.append({
                    "team":             team,
                    "year":             year,
                    "hist_mundiales":   0,
                    "hist_max_ronda":   0,
                    "hist_avg_ronda":   0.0,
                    "hist_veces_qf":    0,
                    "hist_veces_sf":    0,
                    "hist_veces_final": 0,
                    "hist_avg_goles":   0.0,
                    "hist_tendencia":   0.0,
                })
            else:
                # Tendencia: promedio de los últimos 2 mundiales
                # Si el equipo mejoró en los últimos 2 vs su promedio total,
                # hist_tendencia > hist_avg_ronda → señal positiva
                ultimos_2 = history_before.nlargest(2, "year")["max_round"].mean()

                records.append({
                    "team":             team,
                    "year":             year,
                    "hist_mundiales":   len(history_before),
                    "hist_max_ronda":   history_before["max_round"].max(),
                    "hist_avg_ronda":   history_before["max_round"].mean(),
                    "hist_veces_qf":    int((history_before["max_round"] >= 2).sum()),
                    "hist_veces_sf":    int((history_before["max_round"] >= 3).sum()),
                    "hist_veces_final": int((history_before["max_round"] >= 4).sum()),
                    "hist_avg_goles":   history_before["avg_goals"].mean(),
                    "hist_tendencia":   ultimos_2,
                })

    result = pd.DataFrame(records)
    print(f"[feature_engineering] build_historical_features: {result.shape}")
    return result


# ===========================================================
# FUNCIÓN: PIPELINE COMPLETO
# ===========================================================

def build_full_dataset(
    df_matches: pd.DataFrame,
    df_rankings: pd.DataFrame
) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    Combina build_match_features() y build_historical_features()
    en un único DataFrame listo para entrenar el modelo.

    PASOS:
        1. Construir features de partido (una fila por equipo por partido)
        2. Construir features históricas (una fila por equipo por año)
        3. Hacer merge de ambos DataFrames

    RETORNA:
        DataFrame con todas las features + target gana_partido
        Listo para usar en 02_train.py
    """
    print("[feature_engineering] Construyendo dataset completo...")

    # Paso 1: features de partido
    df_by_team = build_match_features(df_matches, df_rankings)

    # Paso 2: features históricas
    df_hist = build_historical_features(df_by_team)

    # Paso 3: merge
    # Cada partido del equipo en el año X recibe las features
    # históricas calculadas hasta el año X-1
    df_final = df_by_team.merge(df_hist, on=["team", "year"], how="left")

    # Rellenar NaN en features históricas con 0
    # (equipos sin historial previo reciben 0 en todas las features hist_*)
    hist_cols = [c for c in df_final.columns if c.startswith("hist_")]
    df_final[hist_cols] = df_final[hist_cols].fillna(0)

    print(f"[feature_engineering] Dataset final: {df_final.shape}")
    print(f"  Años: {sorted(df_final['year'].unique())}")
    print(f"  Target gana_partido: {df_final['gana_partido'].value_counts().to_dict()}")

    return df_final


# ===========================================================
# FUNCIÓN: FEATURES PARA UN PARTIDO 2026 (uso en simulador)
# ===========================================================

def build_match_row_2026(
    team_a: str,
    team_b: str,
    df_rankings: pd.DataFrame,
    hist_stats: dict,
    is_home_team_a: int = 1,
    host: str = "USA"
) -> pd.DataFrame:
    """
    Construye las features para UN partido específico del Mundial 2026.
    Usado por el simulador para predecir cada enfrentamiento.

    PARÁMETROS:
        team_a         → equipo que queremos predecir (perspectiva)
        team_b         → rival
        df_rankings    → ranking FIFA (para obtener rank actual)
        hist_stats     → dict con estadísticas históricas de todos los equipos
                         (generado en 03_simulate.py desde el dataset)
        is_home_team_a → 1 si team_a juega como "local" en este partido
        host           → país anfitrión del mundial (USA por defecto)

    RETORNA:
        DataFrame con una fila y todas las features del modelo
        Listo para scaler.transform() y modelo.predict_proba()
    """
    rank_a = get_ranking_before_worldcup(df_rankings, team_a, 2026)
    rank_b = get_ranking_before_worldcup(df_rankings, team_b, 2026)

    hist_a = hist_stats.get(team_a, {k: 0 for k in [
        "hist_mundiales", "hist_max_ronda", "hist_avg_ronda",
        "hist_veces_qf", "hist_veces_sf", "hist_veces_final",
        "hist_avg_goles", "hist_tendencia"
    ]})

    row = {
        "goals":       0,   # desconocido antes del partido
        "penalty":     0,
        "rank":        rank_a["rank"],
        "points":      rank_a["points"],
        "rank_diff":   rank_a["rank"] - rank_b["rank"],
        "points_diff": rank_a["points"] - rank_b["points"],
        "rank_ratio":  rank_b["rank"] / (rank_a["rank"] + 1e-6),
        "same_conf":   int(rank_a["confederation"] == rank_b["confederation"]),
        "is_home":     is_home_team_a,
        "is_host":     int(team_a in ["USA", "Mexico", "Canada"]),
        "rank_tier":   _rank_to_tier(rank_a["rank"]),
        **hist_a
    }

    return pd.DataFrame([row])[MODEL_FEATURES]
