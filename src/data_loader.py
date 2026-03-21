"""
===========================================================
src/data_loader.py
===========================================================

PROPÓSITO:
    Centraliza la carga de todos los archivos del proyecto.
    Ningún otro script debe leer archivos directamente —
    siempre usan las funciones de este módulo.

¿POR QUÉ HACERLO ASÍ?
    Si mañana cambias el nombre de un archivo o su ubicación,
    solo lo corriges aquí y todo el resto sigue funcionando.
    Sin este módulo tendrías rutas de archivo repetidas en
    5 scripts distintos y habría que cambiarlas en todos.

CONTIENE:
    - load_matches()      → carga matches.csv (partidos históricos)
    - load_rankings()     → carga rankings.csv (ranking FIFA histórico)
    - load_groups_2026()  → carga groups_2026.csv (grupos del mundial)
    - save_processed()    → guarda un DataFrame en data/processed/
    - load_processed()    → carga un archivo de data/processed/

AUTOR: Oscar Daniel Farfan Juanias
===========================================================
"""

import pandas as pd
from pathlib import Path


# ===========================================================
# RUTAS BASE
# ===========================================================
# Path(__file__) es la ruta de este archivo (data_loader.py)
# .resolve().parents[1] sube DOS niveles desde src/ hasta la raíz
# Así las rutas funcionan sin importar desde dónde se ejecute el script

ROOT        = Path(__file__).resolve().parents[1]
RAW_DIR     = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


# ===========================================================
# FUNCIONES DE CARGA
# ===========================================================

def load_matches() -> pd.DataFrame:
    """
    Carga el dataset histórico de partidos de mundiales.

    ARCHIVO: data/raw/matches.csv
    ORIGEN:  Compilación histórica de partidos 1930-2022.
             Incluye goles, xG, penales, asistencia, ronda, año, etc.

    COLUMNAS PRINCIPALES:
        home_team, away_team    → nombres de los equipos
        home_score, away_score  → goles marcados
        home_xg, away_xg        → expected goals (xG)
        home_penalty, away_penalty → penales anotados
        Round                   → ronda del torneo
        Year                    → año del mundial
        Host                    → país anfitrión
        Attendance              → asistencia al estadio

    RETORNA:
        pd.DataFrame con todos los partidos históricos
    """
    path = RAW_DIR / "matches.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró matches.csv en {RAW_DIR}\n"
            f"Asegúrate de haber copiado tu Data.csv como matches.csv"
        )

    df = pd.read_csv(path, sep=",", low_memory=False)
    print(f"[data_loader] matches.csv cargado: {df.shape[0]} partidos, "
          f"{df.shape[1]} columnas")
    return df


def load_rankings() -> pd.DataFrame:
    """
    Carga el ranking FIFA histórico mensual.

    ARCHIVO: data/raw/rankings.csv
    ORIGEN:  Ranking FIFA desde diciembre 1992 hasta 2023.
             ~70,000 registros. Un registro por equipo por fecha.

    ¿POR QUÉ ES IMPORTANTE?
        Permite saber exactamente qué posición tenía cada equipo
        ANTES de cada mundial. Esto evita data leakage (usar
        información del futuro para entrenar el modelo).

    COLUMNAS PRINCIPALES:
        rank            → posición en el ranking FIFA
        country_full    → nombre completo del país
        country_abrv    → abreviatura del país
        total_points    → puntos FIFA acumulados
        previous_points → puntos del período anterior
        rank_change     → cambio respecto al período anterior
        confederation   → confederación (UEFA, CONMEBOL, AFC, etc.)
        rank_date       → fecha del ranking (DD/MM/YYYY)

    TRANSFORMACIONES APLICADAS:
        - rank_date se convierte a datetime
        - Se eliminan filas sin rank (equipos sin clasificación)
        - rank se convierte a entero

    RETORNA:
        pd.DataFrame con el historial completo de rankings FIFA
    """
    path = RAW_DIR / "rankings.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró rankings.csv en {RAW_DIR}\n"
            f"Asegúrate de haber copiado tu DataScore.csv como rankings.csv"
        )

    df = pd.read_csv(path, sep=",", low_memory=False)

    # Convertir fecha a datetime
    # dayfirst=True porque el formato es DD/MM/YYYY
    df["rank_date"] = pd.to_datetime(
        df["rank_date"], dayfirst=True, errors="coerce"
    )

    # Eliminar filas donde rank es NaN (equipos sin clasificación oficial)
    df = df.dropna(subset=["rank"])
    df["rank"] = df["rank"].astype(int)

    print(f"[data_loader] rankings.csv cargado: {df.shape[0]} registros")
    print(f"[data_loader] Rango de fechas: "
          f"{df['rank_date'].min().date()} → {df['rank_date'].max().date()}")
    return df


def load_groups_2026() -> pd.DataFrame:
    """
    Carga los grupos del Mundial 2026.

    ARCHIVO: data/raw/groups_2026.csv
    CONTENIDO:
        Los 48 equipos organizados en 12 grupos (A-L).
        6 equipos están marcados como placeholders (confirmed=no)
        para los play-offs que aún no se han disputado.

    COLUMNAS:
        group           → letra del grupo (A, B, C... L)
        team            → nombre del equipo
        confirmed       → 'yes' si ya está clasificado, 'no' si es placeholder
        placeholder_for → descripción del play-off que representa

    RETORNA:
        pd.DataFrame con los 48 equipos y su grupo asignado
    """
    path = RAW_DIR / "groups_2026.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró groups_2026.csv en {RAW_DIR}"
        )

    df = pd.read_csv(path)

    # Mostrar resumen de equipos confirmados vs pendientes
    confirmados = (df["confirmed"] == "yes").sum()
    pendientes  = (df["confirmed"] == "no").sum()

    print(f"[data_loader] groups_2026.csv cargado: "
          f"{confirmados} confirmados, {pendientes} placeholders")
    return df


# ===========================================================
# FUNCIONES DE GUARDADO Y CARGA DE PROCESADOS
# ===========================================================

def save_processed(df: pd.DataFrame, filename: str) -> None:
    """
    Guarda un DataFrame procesado en data/processed/.

    ¿POR QUÉ GUARDAR EN PROCESSED Y NO EN RAW?
        Los archivos en raw/ nunca se modifican — son la fuente
        original. Los archivos en processed/ son derivados que
        se pueden regenerar ejecutando 01_etl.py en cualquier momento.
        Por eso processed/ está en .gitignore.

    PARÁMETROS:
        df       → DataFrame a guardar
        filename → nombre del archivo (ej: "model_dataset.csv")
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    print(f"[data_loader] Guardado: {path}")


def load_processed(filename: str) -> pd.DataFrame:
    """
    Carga un archivo desde data/processed/.

    PARÁMETROS:
        filename → nombre del archivo (ej: "model_dataset.csv")

    RETORNA:
        pd.DataFrame con los datos procesados
    """
    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {filename} en {PROCESSED_DIR}\n"
            f"Ejecuta primero: python 01_etl.py"
        )

    df = pd.read_csv(path)
    print(f"[data_loader] {filename} cargado: {df.shape}")
    return df


def get_paths() -> dict:
    """
    Retorna un diccionario con todas las rutas del proyecto.
    Útil para scripts que necesitan guardar modelos o outputs.

    RETORNA:
        dict con claves: root, raw, processed, models, outputs
    """
    return {
        "root":      ROOT,
        "raw":       RAW_DIR,
        "processed": PROCESSED_DIR,
        "models":    MODELS_DIR,
        "outputs":   OUTPUTS_DIR,
    }
