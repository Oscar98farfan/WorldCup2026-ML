"""
===========================================================
01_etl.py — ETL: Extracción, Transformación y Carga
===========================================================

PROPÓSITO:
    Lee los datos crudos (raw), los limpia, los transforma
    y genera el dataset final listo para entrenar el modelo.

¿QUÉ SIGNIFICA ETL?
    E → Extract:   leer los archivos originales
    T → Transform: limpiar, unir y construir features
    L → Load:      guardar el resultado en data/processed/

¿POR QUÉ UN SCRIPT SEPARADO PARA ESTO?
    La limpieza de datos es una etapa distinta al entrenamiento.
    Si en el futuro consigues datos nuevos (ej: partidos 2026),
    solo vuelves a correr este script sin tocar el modelo.

ENTRADA:
    data/raw/matches.csv   → partidos históricos de mundiales
    data/raw/rankings.csv  → ranking FIFA histórico

SALIDA:
    data/processed/model_dataset.csv → dataset limpio con todas
                                       las features listo para
                                       usar en 02_train.py

CÓMO EJECUTAR:
    python 01_etl.py

AUTOR: Oscar Daniel Farfan Juanias
===========================================================
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Agregar la raíz del proyecto al path para poder importar src/
# Esto es necesario porque ejecutamos el script desde la raíz
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import load_matches, load_rankings, save_processed
from src.feature_engineering import build_full_dataset


# ===========================================================
# PASO 1: CARGAR DATOS CRUDOS
# ===========================================================

def load_raw_data():
    """
    Carga los dos archivos fuente del proyecto.

    RETORNA:
        df_matches  → DataFrame de partidos históricos
        df_rankings → DataFrame del ranking FIFA histórico
    """
    print("\n" + "="*55)
    print("  PASO 1: Cargando datos crudos")
    print("="*55)

    df_matches  = load_matches()
    df_rankings = load_rankings()

    # Mostrar resumen básico para verificar que cargaron bien
    print(f"\n  Partidos:  {len(df_matches):>6} filas | "
          f"{df_matches['Year'].nunique()} mundiales "
          f"({int(df_matches['Year'].min())}-{int(df_matches['Year'].max())})")
    print(f"  Rankings:  {len(df_rankings):>6} filas | "
          f"{df_rankings['country_full'].nunique()} países")

    return df_matches, df_rankings


# ===========================================================
# PASO 2: LIMPIAR DATOS DE PARTIDOS
# ===========================================================

def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de partidos históricos.

    OPERACIONES:
        1. Seleccionar solo las columnas que necesitamos
        2. Convertir columnas numéricas al tipo correcto
        3. Rellenar valores nulos con 0 donde corresponde
        4. Verificar que no queden nulos críticos

    COLUMNAS QUE CONSERVAMOS:
        Las mínimas necesarias para construir las features.
        Attendance se excluye intencionalmente (ver documentación
        en feature_engineering.py).

    PARÁMETROS:
        df → DataFrame crudo de matches.csv

    RETORNA:
        DataFrame limpio listo para feature engineering
    """
    print("\n" + "="*55)
    print("  PASO 2: Limpiando datos de partidos")
    print("="*55)

    # Columnas que necesitamos
    # Nota: xG se excluye porque tuvo importancia 0.000 en el modelo
    # Attendance se excluye por correlación espuria con la ronda
    cols_needed = [
        "home_team", "away_team",
        "home_score", "away_score",
        "home_penalty", "away_penalty",
        "Round", "Host", "Year",
    ]

    # Verificar que las columnas existen
    missing_cols = [c for c in cols_needed if c not in df.columns]
    if missing_cols:
        print(f"  ⚠️  Columnas no encontradas: {missing_cols}")
        print(f"  Columnas disponibles: {list(df.columns)}")
        # Intentar con las columnas que sí existen
        cols_needed = [c for c in cols_needed if c in df.columns]

    df = df[cols_needed].copy()

    # Convertir columnas numéricas
    num_cols = ["home_score", "away_score", "home_penalty", "away_penalty", "Year"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Limpiar nombres de equipos (eliminar espacios extra)
    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()
    df["Host"]      = df["Host"].str.strip()
    df["Round"]     = df["Round"].str.strip()

    # Eliminar filas sin año o sin equipos (datos corruptos)
    antes = len(df)
    df = df.dropna(subset=["Year", "home_team", "away_team"])
    eliminadas = antes - len(df)
    if eliminadas > 0:
        print(f"  ⚠️  {eliminadas} filas eliminadas por valores nulos críticos")

    # Mostrar resumen de nulos restantes
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    if len(nulos) > 0:
        print(f"\n  Valores nulos restantes:")
        print(nulos)
    else:
        print(f"\n  ✅ Sin valores nulos")

    # Mostrar distribución de rondas
    print(f"\n  Partidos por ronda:")
    round_counts = df["Round"].value_counts()
    for ronda, count in round_counts.items():
        print(f"    {ronda:<25}: {count}")

    print(f"\n  Dataset limpio: {df.shape}")
    return df


# ===========================================================
# PASO 3: VERIFICAR COBERTURA DEL RANKING FIFA
# ===========================================================

def check_ranking_coverage(
    df_matches: pd.DataFrame,
    df_rankings: pd.DataFrame
) -> None:
    """
    Verifica qué porcentaje de equipos tiene datos en el ranking FIFA
    para cada año de mundial.

    ¿POR QUÉ VERIFICAR ESTO?
        El ranking FIFA existe desde diciembre 1992.
        Para mundiales anteriores (1930-1990) no hay datos de ranking.
        El modelo usa valores por defecto para esos casos (rank=75).
        Esta función nos muestra exactamente dónde hay datos reales
        y dónde hay valores por defecto.

    NO RETORNA NADA — solo imprime información diagnóstica.
    """
    print("\n" + "="*55)
    print("  PASO 3: Verificando cobertura del ranking FIFA")
    print("="*55)

    from src.feature_engineering import WORLDCUP_START_DATES

    print(f"\n  {'Año':>6} {'Equipos':>8} {'Con ranking':>12} {'Cobertura':>11}")
    print("  " + "-"*42)

    for year in sorted(df_matches["Year"].unique()):
        teams_in_year = pd.concat([
            df_matches[df_matches["Year"] == year]["home_team"],
            df_matches[df_matches["Year"] == year]["away_team"]
        ]).unique()

        cutoff = pd.Timestamp(
            WORLDCUP_START_DATES.get(int(year), f"{int(year)}-06-01")
        )

        con_ranking = 0
        for team in teams_in_year:
            mask = (
                (df_rankings["country_full"] == team) &
                (df_rankings["rank_date"] <= cutoff)
            )
            if df_rankings[mask].shape[0] > 0:
                con_ranking += 1

        cobertura = con_ranking / len(teams_in_year) * 100
        indicador = "✅" if cobertura >= 80 else "⚠️ " if cobertura > 0 else "❌"
        print(f"  {int(year):>6} {len(teams_in_year):>8} {con_ranking:>12} "
              f"{cobertura:>10.1f}%  {indicador}")

    print(f"\n  ✅ = cobertura ≥ 80% | ⚠️  = cobertura parcial | ❌ = sin datos")
    print(f"  El modelo usa rank=75 como valor por defecto cuando no hay datos.")


# ===========================================================
# PASO 4: CONSTRUIR FEATURES Y DATASET FINAL
# ===========================================================

def build_and_save_dataset(
    df_matches: pd.DataFrame,
    df_rankings: pd.DataFrame
) -> pd.DataFrame:
    """
    Construye el dataset final con todas las features y lo guarda.

    Llama a build_full_dataset() de feature_engineering.py
    que hace todo el trabajo de construcción de features.

    PARÁMETROS:
        df_matches  → partidos limpios (output del paso 2)
        df_rankings → ranking FIFA

    RETORNA:
        DataFrame final guardado en data/processed/model_dataset.csv
    """
    print("\n" + "="*55)
    print("  PASO 4: Construyendo features y dataset final")
    print("="*55)

    # Construir dataset completo con todas las features
    df_final = build_full_dataset(df_matches, df_rankings)

    # Verificación final
    print(f"\n  Resumen del dataset final:")
    print(f"    Filas totales:        {len(df_final):>6}")
    print(f"    Features del modelo:  {len([c for c in df_final.columns if c in __import__('src.feature_engineering', fromlist=['MODEL_FEATURES']).MODEL_FEATURES]):>6}")
    print(f"    Años incluidos:       {sorted(df_final['year'].unique())}")
    print(f"    Target gana_partido:")
    target_counts = df_final["gana_partido"].value_counts()
    print(f"      Ganó  (1): {target_counts.get(1,0):>4}  "
          f"({target_counts.get(1,0)/len(df_final)*100:.1f}%)")
    print(f"      No ganó (0): {target_counts.get(0,0):>4}  "
          f"({target_counts.get(0,0)/len(df_final)*100:.1f}%)")

    # Verificar que no hay nulos en las features del modelo
    from src.feature_engineering import MODEL_FEATURES
    nulos_features = df_final[MODEL_FEATURES].isna().sum().sum()
    if nulos_features > 0:
        print(f"\n  ⚠️  {nulos_features} valores nulos en features — rellenando con 0")
        df_final[MODEL_FEATURES] = df_final[MODEL_FEATURES].fillna(0)
    else:
        print(f"\n  ✅ Sin valores nulos en las features del modelo")

    # Guardar en data/processed/
    save_processed(df_final, "model_dataset.csv")
    print(f"\n  ✅ Dataset guardado: data/processed/model_dataset.csv")

    return df_final


# ===========================================================
# MAIN — EJECUTAR ETL COMPLETO
# ===========================================================

def main():
    print("\n" + "="*55)
    print("  WorldCup2026-ML — ETL")
    print("  Limpieza y preparación de datos")
    print("="*55)

    # Paso 1: cargar datos crudos
    df_matches, df_rankings = load_raw_data()

    # Paso 2: limpiar datos de partidos
    df_matches_clean = clean_matches(df_matches)

    # Paso 3: verificar cobertura del ranking (informativo)
    check_ranking_coverage(df_matches_clean, df_rankings)

    # Paso 4: construir features y guardar
    df_final = build_and_save_dataset(df_matches_clean, df_rankings)

    print("\n" + "="*55)
    print("  ETL COMPLETADO")
    print("  Siguiente paso: python 02_train.py")
    print("="*55 + "\n")

    return df_final


if __name__ == "__main__":
    main()
