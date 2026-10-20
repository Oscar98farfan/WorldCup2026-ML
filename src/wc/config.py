"""
Configuración central del proyecto.

Todo lo que pueda cambiar entre experimentos (rutas, fechas, listas de
features, hiperparámetros) vive aquí. El resto de módulos importa desde
este archivo y nunca define constantes propias.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# ---------------------------------------------------------------------------
# Torneos
# ---------------------------------------------------------------------------
# Fecha de inicio de cada Mundial = fecha de corte para cualquier dato
# "previo al torneo" (ranking FIFA, Elo, historial). Nada posterior entra.
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

# Anfitriones por año (lista porque 2002 y 2026 tienen varios).
HOSTS = {
    1930: ["Uruguay"], 1934: ["Italy"], 1938: ["France"], 1950: ["Brazil"],
    1954: ["Switzerland"], 1958: ["Sweden"], 1962: ["Chile"], 1966: ["England"],
    1970: ["Mexico"], 1974: ["Germany"], 1978: ["Argentina"], 1982: ["Spain"],
    1986: ["Mexico"], 1990: ["Italy"], 1994: ["USA"], 1998: ["France"],
    2002: ["Korea Republic", "Japan"], 2006: ["Germany"], 2010: ["South Africa"],
    2014: ["Brazil"], 2018: ["Russia"], 2022: ["Qatar"],
    2026: ["USA", "Mexico", "Canada"],
}

# Confederación del país anfitrión: sirve para la feature "juega en su
# continente" (las selecciones europeas rinden peor en América y viceversa).
HOST_CONFEDERATION = {
    1930: "CONMEBOL", 1934: "UEFA", 1938: "UEFA", 1950: "CONMEBOL",
    1954: "UEFA", 1958: "UEFA", 1962: "CONMEBOL", 1966: "UEFA",
    1970: "CONCACAF", 1974: "UEFA", 1978: "CONMEBOL", 1982: "UEFA",
    1986: "CONCACAF", 1990: "UEFA", 1994: "CONCACAF", 1998: "UEFA",
    2002: "AFC", 2006: "UEFA", 2010: "CAF", 2014: "CONMEBOL",
    2018: "UEFA", 2022: "AFC", 2026: "CONCACAF",
}

# Profundidad alcanzada (para features históricas).
# 0 = grupos, 1 = 1.ª eliminatoria, 2 = cuartos, 3 = semis, 4 = final.
ROUND_DEPTH = {
    "Group stage": 0, "First group stage": 0, "First round": 0,
    "Group stage play-off": 0, "Second group stage": 1, "Second round": 1,
    "Round of 16": 1, "Quarter-finals": 2, "Semi-finals": 3,
    "Third-place match": 3, "Final stage": 3, "Final": 4,
}
GROUP_ROUNDS = {"Group stage", "First group stage", "First round",
                "Group stage play-off", "Second group stage", "Final stage"}

FIFA_RANKING_START = 1994       # primer Mundial con ranking FIFA real
EVAL_YEARS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]  # walk-forward
TARGET_YEAR = 2026

# ---------------------------------------------------------------------------
# Elo (hiperparámetros fijados a priori, NO ajustados contra el test)
# ---------------------------------------------------------------------------
ELO_INIT = 1500.0
ELO_K = 50.0                 # K alto: el Mundial es la competición de más peso
ELO_HOST_BONUS = 100.0       # ventaja de jugar en casa
ELO_REGRESSION = 0.20        # entre torneos se regresa 20% a la media

# ---------------------------------------------------------------------------
# Features del modelo de partido
# ---------------------------------------------------------------------------
# Todas son DIFERENCIAS equipo A − equipo B, conocidas ANTES del partido.
# Así el modelo es simétrico: invertir A y B invierte el signo.
MATCH_FEATURES = [
    "d_elo",            # diferencia de Elo mundialista
    "d_points_z",       # diferencia de puntos FIFA normalizados por fecha
    "d_momentum",       # diferencia de tendencia FIFA en los últimos 12 meses
    "d_hist_stage",     # diferencia de ronda media histórica
    "d_recent_stage",   # diferencia de ronda media en los 2 últimos mundiales
    "d_experience",     # diferencia de log(1 + mundiales jugados)
    "d_host",           # anfitrión (1, 0, −1)
    "d_home_conf",      # juega en su confederación (1, 0, −1)
]
# d_log_rank (log(rank_B) − log(rank_A)) se calcula pero no entra al modelo
# completo: es casi colineal con d_points_z y en la ablación walk-forward
# quitarlo mejoraba el log loss. Se usa como baseline.
ALL_PAIR_FEATURES = MATCH_FEATURES + ["d_log_rank"]
# Conjunto mínimo de referencia: si el modelo no supera esto, se dice.
BASELINE_FEATURES = ["d_log_rank"]

OUTCOME_LABELS = {0: "loss", 1: "draw", 2: "win"}   # desde la óptica de A

# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------
N_SIMULATIONS = 20_000
RANDOM_SEED = 42
