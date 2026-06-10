"""
===========================================================
03_simulate.py — Simulación del Torneo Mundial 2026
===========================================================

PROPÓSITO:
    Carga el modelo entrenado y simula el Mundial 2026
    completo usando Monte Carlo: 10,000 simulaciones
    del torneo partido a partido para obtener probabilidades
    de avance por ronda para cada selección.

¿QUÉ ES MONTE CARLO?
    Es una técnica que resuelve problemas usando simulaciones
    aleatorias repetidas. En lugar de calcular directamente
    "¿quién gana el mundial?", simulamos el torneo miles de
    veces y contamos cuántas veces ganó cada equipo.

    Ejemplo: si Brasil gana en 2,847 de 10,000 simulaciones,
    su probabilidad de ser campeón es 28.47%.

    Ventaja: captura la incertidumbre natural del fútbol.
    Si Marruecos tiene 30% de ganar a Francia, en el 30%
    de las simulaciones Marruecos avanza. Eso se propaga
    a través de todo el torneo.

¿POR QUÉ 10,000 SIMULACIONES?
    Con pocas simulaciones (ej: 100) los resultados son
    muy ruidosos. Con 10,000 los porcentajes se estabilizan
    y dan una estimación confiable. Más simulaciones = más
    preciso pero más lento.

FORMATO MUNDIAL 2026 (48 equipos, 12 grupos):
    ┌─────────────────────────────────────────────────────────┐
    │  BLOQUE IZQUIERDO        │  BLOQUE DERECHO              │
    │  Grupos A, B, C, D, E, F │  Grupos G, H, I, J, K, L    │
    └─────────────────────────────────────────────────────────┘

    Fase de grupos:
        - 12 grupos de 4 equipos
        - Clasifican: 2 primeros de cada grupo (24)
                    + 8 mejores terceros (4 del bloque IZQ + 4 del bloque DER)
                    = 32 clasificados en total

    IMPORTANTE — separación de bloques en el bracket:
        Los 16 clasificados del bloque IZQ (grupos A–F) solo pueden
        cruzarse con otros del bloque IZQ hasta semifinales.
        Los 16 clasificados del bloque DER (grupos G–L) ídem.
        Las dos ramas se cruzan ÚNICAMENTE en la final.

        Esto replica el cuadro oficial FIFA 2026:
        https://en.wikipedia.org/wiki/2026_FIFA_World_Cup

    Cruces de octavos de final (bracket oficial):
        Bloque IZQ:
            1A vs 2B  |  1C vs 2D  |  1E vs 2F  |  mejor 3° IZQ vs siguiente
        Bloque DER:
            1G vs 2H  |  1I vs 2J  |  1K vs 2L  |  mejor 3° DER vs siguiente

    Cuartos, semis y final: bracket clásico dentro de cada bloque
    hasta que las dos ramas se unen en la final.

ENTRADA:
    models/match_predictor.pkl → modelo entrenado
    models/scaler.pkl          → escalador
    models/features.pkl        → lista de features
    data/raw/groups_2026.csv   → grupos del mundial
    data/raw/rankings.csv      → ranking FIFA actual
    data/processed/model_dataset.csv → para features históricas

SALIDA:
    outputs/simulation_results.csv → probabilidades por equipo y ronda
                                     incluye columna 'bloque' (IZQ/DER)

CÓMO EJECUTAR:
    python 03_simulate.py

AUTOR: Oscar Daniel Farfan Juanias
===========================================================
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import load_rankings, load_groups_2026, load_processed, get_paths
from src.feature_engineering import MODEL_FEATURES, WORLDCUP_START_DATES
from src.simulator import simulate_tournament, get_results_summary


# ===========================================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# ===========================================================

N_SIMULATIONS = 10_000  # número de simulaciones Monte Carlo
RANDOM_SEED   = 42       # semilla para reproducibilidad
WORLDCUP_YEAR = 2026     # año del mundial a simular

# ── Separación oficial de bloques del Mundial 2026 ─────────
# Los grupos A–F forman el bloque IZQ y los grupos G–L el DER.
# Ambos bloques tienen su propio cuadro hasta la final.
BLOQUE_IZQ = ["A", "B", "C", "D", "E", "F"]
BLOQUE_DER = ["G", "H", "I", "J", "K", "L"]


# ===========================================================
# PASO 1: CARGAR MODELO ENTRENADO
# ===========================================================

def load_model(paths: dict):
    """
    Carga el modelo, scaler y lista de features guardados
    por 02_train.py.

    ¿POR QUÉ CARGAR LOS TRES?
        - modelo:    hace las predicciones
        - scaler:    transforma los datos a la misma escala
                     que se usó en entrenamiento
        - features:  garantiza que las columnas van en el
                     orden exacto que el modelo espera

    RETORNA:
        modelo, scaler, features
    """
    print("\n" + "="*55)
    print("  PASO 1: Cargando modelo entrenado")
    print("="*55)

    models_dir = paths["models"]

    for filename in ["match_predictor.pkl", "scaler.pkl", "features.pkl"]:
        if not (models_dir / filename).exists():
            raise FileNotFoundError(
                f"No se encontró {filename} en models/\n"
                f"Ejecuta primero: python 02_train.py"
            )

    with open(models_dir / "match_predictor.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "features.pkl", "rb") as f:
        features = pickle.load(f)

    print(f"\n  ✅ Modelo cargado: GradientBoostingClassifier")
    print(f"  ✅ Features:       {len(features)} variables")
    return modelo, scaler, features


# ===========================================================
# PASO 2: PREPARAR GRUPOS DEL MUNDIAL 2026
# ===========================================================

def prepare_groups() -> dict:
    """
    Carga y organiza los grupos del Mundial 2026.

    RETORNA:
        dict con formato: {'A': ['Mexico', 'South Africa', ...],
                           'B': ['Canada', 'Qatar', ...], ...}

    Imprime qué grupos pertenecen a cada bloque del cuadro.
    """
    print("\n" + "="*55)
    print("  PASO 2: Preparando grupos del Mundial 2026")
    print("="*55)

    df_groups = load_groups_2026()

    groups = {}
    for group_name, group_df in df_groups.groupby("group"):
        groups[group_name] = group_df["team"].tolist()

    # ── Mostrar agrupado por bloque ────────────────────────
    print(f"\n  {'':─<55}")
    print(f"  BLOQUE IZQUIERDO — grupos {', '.join(BLOQUE_IZQ)}")
    print(f"  {'':─<55}")
    for g in BLOQUE_IZQ:
        if g not in groups:
            continue
        teams = groups[g]
        placeholders = df_groups[
            (df_groups["group"] == g) &
            (df_groups["confirmed"] == "no")
        ]["team"].tolist()
        ph_str = f"  ⚠️  {placeholders}" if placeholders else ""
        print(f"  Grupo {g:<4} {', '.join(teams)}{ph_str}")

    print(f"\n  {'':─<55}")
    print(f"  BLOQUE DERECHO  — grupos {', '.join(BLOQUE_DER)}")
    print(f"  {'':─<55}")
    for g in BLOQUE_DER:
        if g not in groups:
            continue
        teams = groups[g]
        placeholders = df_groups[
            (df_groups["group"] == g) &
            (df_groups["confirmed"] == "no")
        ]["team"].tolist()
        ph_str = f"  ⚠️  {placeholders}" if placeholders else ""
        print(f"  Grupo {g:<4} {', '.join(teams)}{ph_str}")

    total_equipos = sum(len(t) for t in groups.values())
    confirmados   = (df_groups["confirmed"] == "yes").sum()
    pendientes    = (df_groups["confirmed"] == "no").sum()

    print(f"\n  Total equipos:  {total_equipos}")
    print(f"  Confirmados:    {confirmados}")
    print(f"  Placeholders:   {pendientes} (actualizar cuando se confirmen)")
    print(f"\n  ⚙️  Separación de bloques activada:")
    print(f"     IZQ (A–F) y DER (G–L) no se cruzan hasta la final")

    return groups


# ===========================================================
# PASO 3: PREPARAR ESTADÍSTICAS HISTÓRICAS
# ===========================================================

def prepare_historical_stats(groups: dict) -> dict:
    """
    Construye el diccionario de estadísticas históricas
    para todos los equipos del Mundial 2026.

    Precalcularlas aquí evita recalcularlas en cada una
    de las 10,000 simulaciones, lo que haría el proceso
    mucho más lento.

    RETORNA:
        dict con formato:
        {
            'Brazil':    {'hist_mundiales': 22, 'hist_avg_ronda': 2.75, ...},
            'Argentina': {'hist_mundiales': 18, 'hist_avg_ronda': 2.69, ...},
            ...
        }
    """
    print("\n" + "="*55)
    print("  PASO 3: Preparando estadísticas históricas")
    print("="*55)

    df_model = load_processed("model_dataset.csv")

    rendimiento = df_model.groupby(["team", "year"]).agg(
        max_round = ("round_num", "max"),
        avg_goals = ("goals", "mean"),
    ).reset_index()

    all_teams = [team for teams in groups.values() for team in teams]
    hist_stats = {}

    for team in all_teams:
        h = rendimiento[rendimiento["team"] == team]

        if h.empty:
            hist_stats[team] = {
                "hist_mundiales":    0,
                "hist_max_ronda":    0,
                "hist_avg_ronda":    0.0,
                "hist_veces_qf":     0,
                "hist_veces_sf":     0,
                "hist_veces_final":  0,
                "hist_avg_goles":    0.0,
                "hist_tendencia":    0.0,
            }
        else:
            ultimos_2 = h.nlargest(2, "year")["max_round"].mean()
            hist_stats[team] = {
                "hist_mundiales":    len(h),
                "hist_max_ronda":    float(h["max_round"].max()),
                "hist_avg_ronda":    float(h["max_round"].mean()),
                "hist_veces_qf":     int((h["max_round"] >= 2).sum()),
                "hist_veces_sf":     int((h["max_round"] >= 3).sum()),
                "hist_veces_final":  int((h["max_round"] >= 4).sum()),
                "hist_avg_goles":    float(h["avg_goals"].mean()),
                "hist_tendencia":    float(ultimos_2),
            }

    print(f"\n  Estadísticas históricas preparadas para {len(hist_stats)} equipos")
    print(f"\n  Muestra de equipos clave:")
    print(f"  {'Equipo':<20} {'Mundiales':>10} {'Avg ronda':>10} {'Veces QF':>10}")
    print("  " + "-"*55)

    equipos_muestra = ["Brazil", "Argentina", "France", "Germany",
                       "England", "Morocco", "Canada", "Japan"]
    for eq in equipos_muestra:
        if eq in hist_stats:
            h = hist_stats[eq]
            print(f"  {eq:<20} {h['hist_mundiales']:>10} "
                  f"{h['hist_avg_ronda']:>10.2f} "
                  f"{h['hist_veces_qf']:>10}")

    return hist_stats


# ===========================================================
# PASO 4: EJECUTAR SIMULACIÓN
# ===========================================================

def run_simulation(
    groups: dict,
    modelo,
    scaler,
    df_rankings: pd.DataFrame,
    hist_stats: dict,
    paths: dict
) -> pd.DataFrame:
    """
    Ejecuta las N_SIMULATIONS simulaciones del torneo completo
    respetando la separación de bloques IZQ (A–F) y DER (G–L).

    PROCESO INTERNO (ver src/simulator.py para detalle):
        Para cada simulación:
        1. Fase de grupos → 32 clasificados (16 por bloque)
        2. Octavos        → 16 clasificados (8 por bloque)
        3. Cuartos        → 8  clasificados (4 por bloque)
        4. Semifinales    → 4  clasificados (2 por bloque → final cruzada)
        5. Final          → 1 campeón (IZQ vs DER)

        CLAVE: bloques IZQ y DER no se mezclan hasta la final.

    RETORNA:
        DataFrame con probabilidades por equipo, ronda y bloque.
    """
    print("\n" + "="*55)
    print(f"  PASO 4: Ejecutando {N_SIMULATIONS:,} simulaciones")
    print(f"          (separación de bloques A–F | G–L activada)")
    print("="*55)
    print(f"\n  Esto puede tomar 2-5 minutos...")

    df_results = simulate_tournament(
        groups        = groups,
        modelo        = modelo,
        scaler        = scaler,
        df_rankings   = df_rankings,
        hist_stats    = hist_stats,
        n_simulations = N_SIMULATIONS,
        seed          = RANDOM_SEED,
        # ── Parámetros de separación de bloques ────────────
        bloque_izq    = BLOQUE_IZQ,   # grupos A–F
        bloque_der    = BLOQUE_DER,   # grupos G–L
    )

    return df_results


# ===========================================================
# PASO 5: GUARDAR RESULTADOS
# ===========================================================

def save_results(df_results: pd.DataFrame, paths: dict) -> None:
    """
    Guarda los resultados de la simulación en outputs/.

    COLUMNAS DEL ARCHIVO DE SALIDA:
        team        → nombre del equipo
        grupo       → grupo del equipo (A–L)
        bloque      → IZQ (grupos A–F) o DER (grupos G–L)
        prob_r32    → probabilidad de clasificar (pasar fase de grupos)
        prob_r16    → probabilidad de llegar a octavos de final
        prob_qf     → probabilidad de llegar a cuartos de final
        prob_sf     → probabilidad de llegar a semifinales
        prob_final  → probabilidad de llegar a la final
        prob_winner → probabilidad de ser campeón
    """
    outputs_dir = paths["outputs"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_path = outputs_dir / "simulation_results.csv"
    df_results.to_csv(output_path, index=False)

    print(f"\n  ✅ Resultados guardados: outputs/simulation_results.csv")
    print(f"     Columnas: {list(df_results.columns)}")


# ===========================================================
# MAIN
# ===========================================================

def main():
    print("\n" + "="*55)
    print("  WorldCup2026-ML — Simulación del Torneo 2026")
    print("="*55)

    paths = get_paths()

    # Paso 1: cargar modelo
    modelo, scaler, features = load_model(paths)

    # Paso 2: preparar grupos
    groups = prepare_groups()

    # Paso 3: estadísticas históricas
    df_rankings = load_rankings()
    hist_stats  = prepare_historical_stats(groups)

    # Paso 4: simular (con separación de bloques)
    df_results = run_simulation(
        groups, modelo, scaler, df_rankings, hist_stats, paths
    )

    # Paso 5: guardar resultados
    save_results(df_results, paths)

    # Mostrar resumen en consola
    get_results_summary(df_results)

    print("\n" + "="*55)
    print("  SIMULACIÓN COMPLETADA")
    print("  Siguiente paso: python 04_report.py")
    print("="*55 + "\n")

    return df_results


if __name__ == "__main__":
    main()