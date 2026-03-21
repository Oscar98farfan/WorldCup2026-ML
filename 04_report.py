"""
===========================================================
04_report.py — Reportes y Visualizaciones
===========================================================

PROPÓSITO:
    Lee los resultados de la simulación generados por
    03_simulate.py y produce todas las visualizaciones
    y reportes del proyecto.

¿POR QUÉ UN SCRIPT SEPARADO PARA LOS REPORTES?
    Separar la generación de resultados (03_simulate.py)
    de su visualización (este script) permite:
    - Regenerar los reportes sin volver a simular
    - Cambiar el estilo de las gráficas fácilmente
    - Agregar nuevas visualizaciones sin tocar la simulación

SALIDAS:
    outputs/plots/predictions_by_round.png  → gráfica principal
    outputs/plots/top_candidates.png        → top 16 candidatos
    outputs/plots/group_analysis.png        → análisis por grupo
    outputs/report_2026.txt                 → reporte en texto

CÓMO EJECUTAR:
    python 04_report.py

AUTOR: Oscar Daniel Farfan Juanias
===========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import get_paths, load_groups_2026


# ===========================================================
# CONFIGURACIÓN VISUAL
# ===========================================================
# Paleta de colores por ronda
# Verde más intenso = ronda más avanzada
COLORS = {
    "winner": "#1a5276",   # azul oscuro  — campeón
    "final":  "#2980b9",   # azul medio   — final
    "sf":     "#27ae60",   # verde oscuro — semis
    "qf":     "#82e0aa",   # verde claro  — cuartos
    "r16":    "#d5e8d4",   # verde pálido — octavos
    "out":    "#f2f2f2",   # gris         — eliminado
}


# ===========================================================
# PASO 1: CARGAR RESULTADOS
# ===========================================================

def load_results(paths: dict) -> pd.DataFrame:
    """
    Carga los resultados de la simulación.

    ARCHIVO: outputs/simulation_results.csv
    Generado por: 03_simulate.py

    COLUMNAS:
        team        → nombre del equipo
        prob_r16    → probabilidad de pasar grupos (llegar a octavos)
        prob_qf     → probabilidad de llegar a cuartos
        prob_sf     → probabilidad de llegar a semis
        prob_final  → probabilidad de llegar a la final
        prob_winner → probabilidad de ser campeón

    RETORNA:
        DataFrame con resultados ordenados por prob_winner
    """
    print("\n" + "="*55)
    print("  PASO 1: Cargando resultados de la simulación")
    print("="*55)

    results_path = paths["outputs"] / "simulation_results.csv"

    if not results_path.exists():
        raise FileNotFoundError(
            f"No se encontró simulation_results.csv\n"
            f"Ejecuta primero: python 03_simulate.py"
        )

    df = pd.read_csv(results_path)
    df = df.sort_values("prob_winner", ascending=False).reset_index(drop=True)

    print(f"\n  {len(df)} equipos cargados")
    print(f"\n  Top 10 candidatos a campeón:")
    print(f"  {'Equipo':<20} {'Campeón':>9} {'Final':>8} "
          f"{'Semis':>7} {'Cuartos':>9}")
    print("  " + "-"*55)
    for _, row in df.head(10).iterrows():
        print(f"  {row['team']:<20} {row['prob_winner']:>8.1%} "
              f"{row['prob_final']:>7.1%} {row['prob_sf']:>6.1%} "
              f"{row['prob_qf']:>8.1%}")

    return df


# ===========================================================
# PASO 2: GRÁFICA PRINCIPAL — PROBABILIDADES POR RONDA
# ===========================================================

def plot_predictions_by_round(df: pd.DataFrame, paths: dict) -> None:
    """
    Genera la gráfica principal del proyecto: barras apiladas
    mostrando la probabilidad de llegar a cada ronda por equipo.

    ¿POR QUÉ BARRAS APILADAS?
        Permite ver de un vistazo tanto la probabilidad total
        de avanzar como la distribución entre rondas.
        Un equipo con mucho azul oscuro es candidato al título.
        Un equipo con solo verde claro suele llegar a octavos
        pero rara vez va más lejos.

    CÓMO LEER LA GRÁFICA:
        Cada barra representa un equipo.
        El largo total = probabilidad de pasar grupos.
        Los colores muestran qué fracción llega a cada ronda:
        Verde claro → octavos
        Verde medio → cuartos
        Verde oscuro → semis
        Azul medio  → final
        Azul oscuro → campeón
    """
    print("\n[04_report] Generando gráfica de predicciones por ronda...")

    # Ordenar por probabilidad de llegar a semis (más informativo)
    df_plot = df.sort_values("prob_sf", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(14, max(10, len(df_plot) * 0.35)))

    # Construir barras apiladas
    # Cada segmento es la diferencia entre rondas consecutivas
    # para que visualmente cada color represente "llegar a esa ronda
    # pero no a la siguiente"
    p_r16    = df_plot["prob_r16"]
    p_qf     = df_plot["prob_qf"]
    p_sf     = df_plot["prob_sf"]
    p_final  = df_plot["prob_final"]
    p_winner = df_plot["prob_winner"]

    # Diferencias entre rondas
    seg_r16    = p_r16 - p_qf       # llega a octavos pero no a cuartos
    seg_qf     = p_qf  - p_sf       # llega a cuartos pero no a semis
    seg_sf     = p_sf  - p_final    # llega a semis pero no a la final
    seg_final  = p_final - p_winner # llega a la final pero no gana
    seg_winner = p_winner            # gana el mundial

    y = range(len(df_plot))

    # Dibujar segmentos de derecha a izquierda (acumulado)
    ax.barh(y, seg_r16,    left=p_qf,     color=COLORS["r16"],
            label="Octavos de final")
    ax.barh(y, seg_qf,     left=p_sf,     color=COLORS["qf"],
            label="Cuartos de final")
    ax.barh(y, seg_sf,     left=p_final,  color=COLORS["sf"],
            label="Semifinal")
    ax.barh(y, seg_final,  left=p_winner, color=COLORS["final"],
            label="Final")
    ax.barh(y, seg_winner, left=0,        color=COLORS["winner"],
            label="Campeón")

    # Etiquetas de equipo y porcentaje de ganar
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["team"], fontsize=9)

    # Agregar porcentaje del campeón al final de cada barra
    for i, (_, row) in enumerate(df_plot.iterrows()):
        if row["prob_winner"] >= 0.01:  # solo si es > 1%
            ax.text(row["prob_r16"] + 0.005, i,
                    f"{row['prob_winner']:.1%}",
                    va="center", fontsize=7, color="#1a5276")

    ax.set_xlabel("Probabilidad", fontsize=11)
    ax.set_title("Mundial 2026 — Predicciones por Ronda\n"
                 "(Simulación Monte Carlo, 10,000 simulaciones)",
                 fontsize=13, pad=15)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plots_dir = paths["outputs"] / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "predictions_by_round.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✅ outputs/plots/predictions_by_round.png")


# ===========================================================
# PASO 3: GRÁFICA TOP 16 CANDIDATOS
# ===========================================================

def plot_top_candidates(df: pd.DataFrame, paths: dict) -> None:
    """
    Gráfica detallada de los 16 equipos con mayor probabilidad
    de llegar a cuartos de final.

    Muestra 5 barras por equipo (una por ronda) para ver
    claramente cómo se va filtrando cada equipo a medida
    que avanza el torneo.
    """
    print("[04_report] Generando gráfica de top candidatos...")

    top16 = df.head(16).copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    rondas = [
        ("prob_r16",    "Octavos",  COLORS["r16"]),
        ("prob_qf",     "Cuartos",  COLORS["qf"]),
        ("prob_sf",     "Semis",    COLORS["sf"]),
        ("prob_final",  "Final",    COLORS["final"]),
        ("prob_winner", "Campeón",  COLORS["winner"]),
    ]

    x       = np.arange(len(top16))
    n_rondas = len(rondas)
    width   = 0.15

    for i, (col, label, color) in enumerate(rondas):
        offset = (i - n_rondas // 2) * width
        ax.bar(x + offset, top16[col], width,
               label=label, color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(top16["team"], rotation=45,
                       ha="right", fontsize=9)
    ax.set_ylabel("Probabilidad", fontsize=11)
    ax.set_title("Top 16 Candidatos — Probabilidad por Ronda",
                 fontsize=13, pad=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(paths["outputs"] / "plots" / "top_candidates.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✅ outputs/plots/top_candidates.png")


# ===========================================================
# PASO 4: ANÁLISIS POR GRUPO
# ===========================================================

def plot_group_analysis(df: pd.DataFrame, paths: dict) -> None:
    """
    Muestra la probabilidad de pasar grupos para cada equipo
    organizado por su grupo en el torneo.

    Útil para identificar los grupos más competitivos (donde
    ningún equipo tiene probabilidad dominante de pasar) vs
    los grupos más desequilibrados.
    """
    print("[04_report] Generando análisis por grupo...")

    df_groups = load_groups_2026()

    # Unir con probabilidades
    df_merged = df_groups.merge(
        df[["team", "prob_r16", "prob_qf", "prob_winner"]],
        on="team", how="left"
    ).fillna(0)

    groups_sorted = sorted(df_merged["group"].unique())
    n_groups = len(groups_sorted)
    n_cols   = 4
    n_rows   = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for i, group_name in enumerate(groups_sorted):
        ax = axes[i]
        group_data = df_merged[df_merged["group"] == group_name].copy()
        group_data = group_data.sort_values("prob_r16", ascending=False)

        # Color según si es placeholder
        colors = []
        for _, row in group_data.iterrows():
            if row.get("confirmed", "yes") == "no":
                colors.append("#f39c12")  # naranja — placeholder
            else:
                colors.append("#3498db")  # azul    — confirmado

        bars = ax.bar(group_data["team"], group_data["prob_r16"],
                      color=colors, alpha=0.85)

        # Línea de referencia: 50% de probabilidad
        ax.axhline(0.5, color="red", linestyle="--",
                   alpha=0.4, linewidth=1)

        ax.set_title(f"Grupo {group_name}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_xticklabels(group_data["team"], rotation=30,
                           ha="right", fontsize=7)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )
        ax.grid(True, alpha=0.2, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Ocultar ejes sobrantes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Leyenda global
    confirmed_patch   = mpatches.Patch(color="#3498db", label="Confirmado")
    placeholder_patch = mpatches.Patch(color="#f39c12", label="Placeholder (por confirmar)")
    fig.legend(handles=[confirmed_patch, placeholder_patch],
               loc="lower right", fontsize=9)

    fig.suptitle("Probabilidad de pasar la fase de grupos por equipo",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(paths["outputs"] / "plots" / "group_analysis.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✅ outputs/plots/group_analysis.png")


# ===========================================================
# PASO 5: REPORTE EN TEXTO
# ===========================================================

def generate_text_report(df: pd.DataFrame, paths: dict) -> None:
    """
    Genera un reporte en texto plano con los resultados
    completos de la simulación.

    ¿POR QUÉ UN REPORTE EN TEXTO?
        Las gráficas son visuales pero no son fáciles de
        compartir en mensajes o documentos. El reporte .txt
        se puede copiar y pegar en cualquier lugar.
    """
    print("[04_report] Generando reporte en texto...")

    lines = []
    lines.append("=" * 65)
    lines.append("  WORLDCUP2026-ML — PREDICCIONES MUNDIAL 2026")
    lines.append("  Simulación Monte Carlo (10,000 simulaciones)")
    lines.append("=" * 65)

    lines.append(f"\n{'Pos':>4}  {'Equipo':<22} {'Octavos':>9} "
                 f"{'Cuartos':>9} {'Semis':>7} {'Final':>7} {'Campeón':>9}")
    lines.append("  " + "-"*65)

    for i, row in df.iterrows():
        lines.append(
            f"{i+1:>4}  {row['team']:<22} "
            f"{row['prob_r16']:>8.1%} "
            f"{row['prob_qf']:>8.1%} "
            f"{row['prob_sf']:>6.1%} "
            f"{row['prob_final']:>6.1%} "
            f"{row['prob_winner']:>8.1%}"
        )

    # Candidatos a semifinales (prob_sf >= 20%)
    sf_candidates = df[df["prob_sf"] >= 0.20]
    lines.append(f"\n\n{'='*65}")
    lines.append("  CANDIDATOS A SEMIFINALES (prob >= 20%)")
    lines.append("=" * 65)
    for _, row in sf_candidates.iterrows():
        lines.append(f"  {row['team']:<22} → {row['prob_sf']:.1%} "
                     f"(campeón: {row['prob_winner']:.1%})")

    # Candidatos al título (prob_winner >= 5%)
    title_candidates = df[df["prob_winner"] >= 0.05]
    lines.append(f"\n\n{'='*65}")
    lines.append("  CANDIDATOS AL TÍTULO (prob >= 5%)")
    lines.append("=" * 65)
    for _, row in title_candidates.iterrows():
        lines.append(f"  {row['team']:<22} → {row['prob_winner']:.1%}")

    lines.append(f"\n\n{'='*65}")
    lines.append("  NOTAS")
    lines.append("=" * 65)
    lines.append("  - 6 equipos son placeholders (play-offs pendientes)")
    lines.append("  - Actualizar data/raw/groups_2026.csv cuando se")
    lines.append("    confirmen y correr: python 03_simulate.py")
    lines.append("  - El modelo predice partido a partido usando")
    lines.append("    ranking FIFA histórico + estadísticas mundialistas")
    lines.append("  - Walk-Forward AUC promedio: ~0.68 (ver metrics.json)")
    lines.append("=" * 65)

    report_text = "\n".join(lines)

    # Imprimir en consola
    print("\n" + report_text)

    # Guardar en archivo
    report_path = paths["outputs"] / "report_2026.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n  ✅ outputs/report_2026.txt")


# ===========================================================
# MAIN
# ===========================================================

def main():
    print("\n" + "="*55)
    print("  WorldCup2026-ML — Reportes y Visualizaciones")
    print("="*55)

    paths = get_paths()

    # Paso 1: cargar resultados
    df = load_results(paths)

    # Paso 2: gráfica principal
    plot_predictions_by_round(df, paths)

    # Paso 3: top 16 candidatos
    plot_top_candidates(df, paths)

    # Paso 4: análisis por grupo
    plot_group_analysis(df, paths)

    # Paso 5: reporte en texto
    generate_text_report(df, paths)

    print("\n" + "="*55)
    print("  REPORTES GENERADOS")
    print("  outputs/plots/predictions_by_round.png")
    print("  outputs/plots/top_candidates.png")
    print("  outputs/plots/group_analysis.png")
    print("  outputs/report_2026.txt")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
