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
    outputs/plots/predictions_by_round.png  → gráfica principal (barras apiladas)
    outputs/plots/top_candidates.png        → top 16 candidatos por ronda
    outputs/plots/group_analysis.png        → análisis por grupo (12 subplots)
    outputs/plots/bracket_comparison.png    → NUEVA: IZQ vs DER cara a cara
    outputs/report_2026.txt                 → reporte en texto

NOTAS SOBRE LA SEPARACIÓN DE BLOQUES:
    El Mundial 2026 divide los 12 grupos en dos bloques:
        IZQ → grupos A, B, C, D, E, F
        DER → grupos G, H, I, J, K, L
    Los equipos de un bloque NO pueden cruzarse con los del
    otro hasta la final. Esto se refleja en todas las gráficas:
    - Las barras se colorean por bloque (azul=IZQ, naranja=DER)
    - La gráfica de bracket muestra ambas ramas por separado
    - El reporte de texto incluye una sección por bloque

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

# Colores de ronda (verde pálido → azul oscuro conforme avanza)
COLORS = {
    "winner": "#1a5276",   # azul oscuro  — campeón
    "final":  "#2980b9",   # azul medio   — final
    "sf":     "#27ae60",   # verde oscuro — semis
    "qf":     "#82e0aa",   # verde claro  — cuartos
    "r16":    "#d5e8d4",   # verde pálido — octavos
    "r32":    "#f0f7ee",   # casi blanco  — clasifica (fase de grupos)
    "out":    "#f2f2f2",   # gris         — eliminado
}

# Colores de bloque para distinguir IZQ vs DER en gráficas
COLOR_IZQ = "#2e86c1"   # azul  — bloque IZQ (grupos A–F)
COLOR_DER = "#e67e22"   # naranja — bloque DER (grupos G–L)

BLOQUE_IZQ = ["A", "B", "C", "D", "E", "F"]
BLOQUE_DER = ["G", "H", "I", "J", "K", "L"]


# ===========================================================
# HELPERS
# ===========================================================

def _get_bloque(grupo: str) -> str:
    """Devuelve 'IZQ' o 'DER' según el grupo."""
    return "IZQ" if grupo in BLOQUE_IZQ else "DER"


def _bloque_color(bloque) -> str:
    """Devuelve el color de barra según el bloque."""
    return COLOR_IZQ if bloque == "IZQ" else COLOR_DER


# ===========================================================
# PASO 1: CARGAR RESULTADOS
# ===========================================================

def load_results(paths: dict) -> pd.DataFrame:
    """
    Carga los resultados de la simulación.

    Si el CSV ya tiene columna 'bloque' (generado por la versión
    con separación de bloques), la usa directamente.
    Si no, la reconstruye a partir de 'grupo' o de groups_2026.csv.

    RETORNA:
        DataFrame con resultados ordenados por prob_winner,
        siempre con las columnas 'grupo' y 'bloque'.
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

    # ── Asegurar columnas de bloque/grupo ──────────────────
    if "grupo" not in df.columns or df["grupo"].isna().all():
        # Reconstruir desde groups_2026.csv
        try:
            df_groups = load_groups_2026()
            grupo_map = df_groups.set_index("team")["group"].to_dict()
            df["grupo"] = df["team"].map(grupo_map).fillna("?")
        except Exception:
            df["grupo"] = "?"

    if "bloque" not in df.columns or df["bloque"].isna().all():
        df["bloque"] = df["grupo"].apply(
            lambda g: _get_bloque(g) if g != "?" else "?"
        )

    # ── Asegurar prob_r32 (puede llamarse prob_r16 en versiones antiguas) ──
    # En formato 2026: prob_r32 = pasar fase de grupos (clasificar entre 32)
    #                  prob_r16 = pasar octavos (llegar a los 16)
    if "prob_r32" not in df.columns:
        # Versión antigua: prob_r16 era "pasar grupos". Renombramos para
        # mantener consistencia con el nuevo esquema de 5 rondas.
        df["prob_r32"] = df.get("prob_r16", 0.0)
        # prob_r16 real (octavos) no está disponible → estimamos como prob_qf
        # para que las gráficas no se rompan
        if "prob_r16" in df.columns and "prob_qf" in df.columns:
            # Ya existe prob_r16 como octavos: no hacer nada
            pass
        elif "prob_qf" in df.columns:
            df["prob_r16"] = df["prob_qf"]

    # ── Mostrar resumen por bloque ─────────────────────────
    print(f"\n  {len(df)} equipos cargados")
    for bloque_name, grupos_bloque in [("IZQ (A–F)", BLOQUE_IZQ),
                                        ("DER (G–L)", BLOQUE_DER)]:
        sub = df[df["bloque"] == bloque_name.split()[0]]
        if len(sub) == 0:
            continue
        print(f"\n  Bloque {bloque_name} — top 5:")
        print(f"  {'Equipo':<20} {'Campeón':>9} {'Final':>8} {'Semis':>7}")
        print("  " + "-"*46)
        for _, row in sub.head(5).iterrows():
            print(f"  {row['team']:<20} {row['prob_winner']:>8.1%} "
                  f"{row['prob_final']:>7.1%} {row['prob_sf']:>6.1%}")

    return df


# ===========================================================
# PASO 2: GRÁFICA PRINCIPAL — PROBABILIDADES POR RONDA
# ===========================================================

def plot_predictions_by_round(df: pd.DataFrame, paths: dict) -> None:
    """
    Barras apiladas horizontales con probabilidad por ronda.
    Las barras se colorean por bloque: azul=IZQ, naranja=DER.

    CÓMO LEER LA GRÁFICA:
        Largo total   = probabilidad de clasificar (pasar grupos)
        Verde pálido  → clasifica (llega a octavos)
        Verde claro   → cuartos de final
        Verde oscuro  → semifinal
        Azul medio    → final
        Azul oscuro   → campeón
        Borde azul    = equipo del bloque IZQ (grupos A–F)
        Borde naranja = equipo del bloque DER (grupos G–L)
    """
    print("\n[04_report] Generando gráfica de predicciones por ronda...")

    df_plot = df.sort_values("prob_sf", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(14, max(10, len(df_plot) * 0.35)))

    # Columnas de probabilidad disponibles
    p_r32    = df_plot.get("prob_r32",   df_plot.get("prob_r16", 0))
    p_r16    = df_plot.get("prob_r16",   df_plot.get("prob_qf", 0))
    p_qf     = df_plot["prob_qf"]
    p_sf     = df_plot["prob_sf"]
    p_final  = df_plot["prob_final"]
    p_winner = df_plot["prob_winner"]

    # Segmentos (diferencia entre rondas)
    seg_r32    = p_r32 - p_r16        # clasifica pero no pasa octavos
    seg_r16    = p_r16 - p_qf         # octavos pero no cuartos
    seg_qf     = p_qf  - p_sf         # cuartos pero no semis
    seg_sf     = p_sf  - p_final      # semis pero no final
    seg_final  = p_final - p_winner   # final pero no campeón
    seg_winner = p_winner             # campeón

    y = range(len(df_plot))

    ax.barh(y, seg_r32,    left=p_r16,    color=COLORS["r32"],    label="Clasifica")
    ax.barh(y, seg_r16,    left=p_qf,     color=COLORS["r16"],    label="Octavos")
    ax.barh(y, seg_qf,     left=p_sf,     color=COLORS["qf"],     label="Cuartos")
    ax.barh(y, seg_sf,     left=p_final,  color=COLORS["sf"],     label="Semifinal")
    ax.barh(y, seg_final,  left=p_winner, color=COLORS["final"],  label="Final")
    ax.barh(y, seg_winner, left=0,        color=COLORS["winner"], label="Campeón")

    # Borde de color por bloque
    for i, (_, row) in enumerate(df_plot.iterrows()):
        bloque = row.get("bloque", "?")
        bcolor = _bloque_color(bloque)
        ax.barh(i, p_r32.iloc[i], left=0,
                color="none", edgecolor=bcolor, linewidth=1.2)

    # Etiquetas de equipo + porcentaje campeón
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["team"], fontsize=9)

    for i, (_, row) in enumerate(df_plot.iterrows()):
        if row["prob_winner"] >= 0.01:
            ax.text(p_r32.iloc[i] + 0.005, i,
                    f"{row['prob_winner']:.1%}",
                    va="center", fontsize=7, color="#1a5276")

    # Leyenda combinada: rondas + bloques
    handles, labels = ax.get_legend_handles_labels()
    izq_patch = mpatches.Patch(facecolor="white", edgecolor=COLOR_IZQ,
                                linewidth=1.5, label="Bloque IZQ (A–F)")
    der_patch = mpatches.Patch(facecolor="white", edgecolor=COLOR_DER,
                                linewidth=1.5, label="Bloque DER (G–L)")
    ax.legend(handles=handles + [izq_patch, der_patch],
              loc="lower right", fontsize=8)

    ax.set_xlabel("Probabilidad", fontsize=11)
    ax.set_title(
        "Mundial 2026 — Predicciones por Ronda\n"
        "(Monte Carlo 10,000 sim · borde azul = bloque A–F · borde naranja = bloque G–L)",
        fontsize=12, pad=12
    )
    ax.set_xlim(0, 1.08)
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
    Top 16 equipos con mayor prob_winner.
    Barras agrupadas (una barra por ronda por equipo).
    Color de barra según el bloque del equipo.

    La etiqueta del eje X incluye el grupo y el bloque:
        Argentina (G · IZQ)
    """
    print("[04_report] Generando gráfica de top candidatos...")

    top16 = df.head(16).copy()

    fig, ax = plt.subplots(figsize=(14, 8))

    todas_rondas = [
        ("prob_r32",    "Clasifica",  COLORS["r32"]),
        ("prob_r16",    "Octavos",    COLORS["r16"]),
        ("prob_qf",     "Cuartos",    COLORS["qf"]),
        ("prob_sf",     "Semis",      COLORS["sf"]),
        ("prob_final",  "Final",      COLORS["final"]),
        ("prob_winner", "Campeón",    COLORS["winner"]),
    ]
    rondas = [(col, lbl, clr) for col, lbl, clr in todas_rondas
              if col in top16.columns]

    x        = np.arange(len(top16))
    n_rondas = len(rondas)
    width    = 0.12

    for i, (col, label, color) in enumerate(rondas):
        offset = (i - n_rondas / 2 + 0.5) * width
        ax.bar(x + offset, top16[col], width,
               label=label, color=color, alpha=0.92)

    # Etiqueta con nombre, grupo y bloque
    labels = []
    for _, row in top16.iterrows():
        grupo  = row.get("grupo",  "?")
        bloque = row.get("bloque", "?")
        labels.append(f"{row['team']}\n(Grp {grupo} · {bloque})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Probabilidad", fontsize=11)
    ax.set_title(
        "Top 16 Candidatos — Probabilidad por Ronda\n"
        "(IZQ = grupos A–F  ·  DER = grupos G–L)",
        fontsize=12, pad=10
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
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
    12 subplots (uno por grupo A–L) con probabilidad de
    clasificar (pasar fase de grupos) por equipo.

    Los subplots se organizan en dos filas de 6:
        Fila 1: grupos A–F  (bloque IZQ, barras azules)
        Fila 2: grupos G–L  (bloque DER, barras naranjas)

    Línea roja punteada = 50% de probabilidad de referencia.
    Barras naranja opaco = placeholder (equipo por confirmar).
    """
    print("[04_report] Generando análisis por grupo...")

    df_groups = load_groups_2026()

    # Columna de probabilidad de clasificar
    prob_col = "prob_r32" if "prob_r32" in df.columns else "prob_r16"

    df_merged = df_groups.merge(
        df[["team", prob_col, "prob_qf", "prob_winner"]],
        on="team", how="left"
    ).fillna(0)

    # Ordenar grupos: A–F primera fila, G–L segunda fila
    grupos_ordenados = BLOQUE_IZQ + BLOQUE_DER
    grupos_presentes = [g for g in grupos_ordenados
                        if g in df_merged["group"].unique()]

    n_cols = 6
    n_rows = 2  # fila 1: IZQ, fila 2: DER

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4.5))
    axes = axes.flatten()

    for i, group_name in enumerate(grupos_presentes):
        ax = axes[i]
        group_data = df_merged[df_merged["group"] == group_name].copy()
        group_data = group_data.sort_values(prob_col, ascending=False)

        bloque = _get_bloque(group_name)
        base_color = COLOR_IZQ if bloque == "IZQ" else COLOR_DER

        colors = []
        for _, row in group_data.iterrows():
            confirmed = row.get("confirmed", "yes")
            if str(confirmed) == "no":
                colors.append("#f39c12")   # naranja — placeholder
            else:
                colors.append(base_color)  # color del bloque

        ax.bar(group_data["team"], group_data[prob_col],
               color=colors, alpha=0.85)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, linewidth=1)

        bloque_label = "IZQ (A–F)" if bloque == "IZQ" else "DER (G–L)"
        ax.set_title(f"Grupo {group_name}  [{bloque_label}]",
                     fontsize=10, fontweight="bold",
                     color=base_color)
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
    for j in range(len(grupos_presentes), len(axes)):
        axes[j].set_visible(False)

    # Leyenda global
    izq_patch  = mpatches.Patch(color=COLOR_IZQ,  label="Bloque IZQ (grupos A–F)")
    der_patch  = mpatches.Patch(color=COLOR_DER,  label="Bloque DER (grupos G–L)")
    ph_patch   = mpatches.Patch(color="#f39c12",  label="Placeholder (por confirmar)")
    line_patch = mpatches.Patch(color="red", alpha=0.5, label="Línea 50%")
    
    # fig.legend(handles=[izq_patch, der_patch, ph_patch, line_patch],
    #            loc="lower center", ncol=4, fontsize=9, y=0.01)
    fig.legend(handles=[izq_patch, der_patch, ph_patch, line_patch],
           loc="lower center", ncol=4, fontsize=9)

    fig.suptitle(
        "Probabilidad de clasificar (pasar fase de grupos) — Mundial 2026\n"
        "Fila superior: Bloque IZQ (grupos A–F) · Fila inferior: Bloque DER (grupos G–L)",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig(paths["outputs"] / "plots" / "group_analysis.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✅ outputs/plots/group_analysis.png")


# ===========================================================
# PASO 5: GRÁFICA DE BRACKET IZQ vs DER  (NUEVA)
# ===========================================================

def plot_bracket_comparison(df: pd.DataFrame, paths: dict) -> None:
    """
    NUEVA gráfica — Comparación de los dos bloques del cuadro.

    Muestra en dos columnas (IZQ | DER) los equipos de cada
    bloque con sus probabilidades de llegar a semis, final y
    ser campeón. Permite ver de un vistazo qué bloque tiene
    los favoritos y qué tan equilibrado está cada cuadro.

    Diseño:
        ┌────────────────┬────────────────┐
        │  BLOQUE IZQ    │  BLOQUE DER    │
        │  (grupos A–F)  │  (grupos G–L)  │
        │   barras azul  │  barras naranj │
        └────────────────┴────────────────┘
    """
    print("[04_report] Generando gráfica de bracket IZQ vs DER...")

    df_izq = df[df["bloque"] == "IZQ"].head(12).copy()
    df_der = df[df["bloque"] == "DER"].head(12).copy()

    # Ordenar por prob_sf dentro de cada bloque
    df_izq = df_izq.sort_values("prob_sf", ascending=True)
    df_der = df_der.sort_values("prob_sf", ascending=True)

    fig, (ax_izq, ax_der) = plt.subplots(1, 2, figsize=(16, 7),
                                          sharey=False)

    def _draw_bracket_half(ax, df_half, bloque_name, base_color):
        p_sf     = df_half["prob_sf"]
        p_final  = df_half["prob_final"]
        p_winner = df_half["prob_winner"]

        seg_sf     = p_sf - p_final
        seg_final  = p_final - p_winner
        seg_winner = p_winner

        y = range(len(df_half))
        ax.barh(y, seg_sf,     left=p_final,  color=COLORS["sf"],
                label="Semis", alpha=0.85)
        ax.barh(y, seg_final,  left=p_winner, color=COLORS["final"],
                label="Final", alpha=0.85)
        ax.barh(y, seg_winner, left=0,        color=COLORS["winner"],
                label="Campeón", alpha=0.95)

        ax.set_yticks(list(y))
        ax.set_yticklabels(
            [f"{row['team']}  (Grp {row.get('grupo','?')})"
             for _, row in df_half.iterrows()],
            fontsize=9
        )

        # Anotar porcentaje de campeón
        for i, (_, row) in enumerate(df_half.iterrows()):
            if row["prob_winner"] >= 0.005:
                ax.text(row["prob_sf"] + 0.005, i,
                        f"{row['prob_winner']:.1%}",
                        va="center", fontsize=7.5,
                        color="#1a5276", fontweight="bold")

        ax.set_xlabel("Probabilidad", fontsize=10)
        ax.set_xlim(0, max(p_sf.max() * 1.25, 0.3))
        ax.set_title(
            f"Bloque {bloque_name}",
            fontsize=13, fontweight="bold", color=base_color, pad=10
        )
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.2, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _draw_bracket_half(ax_izq, df_izq, "IZQ (grupos A–F)", COLOR_IZQ)
    _draw_bracket_half(ax_der, df_der, "DER (grupos G–L)", COLOR_DER)

    fig.suptitle(
        "Cuadro del Mundial 2026 — Probabilidades de avance por bloque\n"
        "(Los dos bloques no se cruzan hasta la final)",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig(paths["outputs"] / "plots" / "bracket_comparison.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✅ outputs/plots/bracket_comparison.png")


# ===========================================================
# PASO 6: REPORTE EN TEXTO
# ===========================================================

def generate_text_report(df: pd.DataFrame, paths: dict) -> None:
    """
    Reporte en texto plano con los resultados completos.
    Incluye una sección separada por bloque (IZQ y DER).
    """
    print("[04_report] Generando reporte en texto...")

    lines = []
    lines.append("=" * 70)
    lines.append("  WORLDCUP2026-ML — PREDICCIONES MUNDIAL 2026")
    lines.append("  Simulación Monte Carlo (10,000 simulaciones)")
    lines.append("  Separación de bloques: IZQ (A–F) | DER (G–L)")
    lines.append("=" * 70)

    # ── Tabla completa ─────────────────────────────────────
    lines.append(f"\n{'Pos':>4}  {'Equipo':<22} {'Grp':>4} {'Bloque':>7} "
                 f"{'Clasifica':>10} {'Octavos':>8} {'Cuartos':>8} "
                 f"{'Semis':>7} {'Final':>7} {'Campeón':>9}")
    lines.append("  " + "-"*85)

    prob_r32_col = "prob_r32" if "prob_r32" in df.columns else "prob_r16"

    for i, row in df.iterrows():
        lines.append(
            f"{i+1:>4}  {row['team']:<22} "
            f"{str(row.get('grupo', '?')):>4} "
            f"{str(row.get('bloque', '?')):>7}  "
            f"{row.get(prob_r32_col, 0):>9.1%} "
            f"{row['prob_r16']:>7.1%} "
            f"{row['prob_qf']:>7.1%} "
            f"{row['prob_sf']:>6.1%} "
            f"{row['prob_final']:>6.1%} "
            f"{row['prob_winner']:>8.1%}"
        )

    # ── Por bloque ─────────────────────────────────────────
    for bloque_tag, bloque_label in [("IZQ", "BLOQUE IZQ — grupos A, B, C, D, E, F"),
                                      ("DER", "BLOQUE DER — grupos G, H, I, J, K, L")]:
        sub = df[df["bloque"] == bloque_tag]
        if sub.empty:
            continue

        lines.append(f"\n\n{'='*70}")
        lines.append(f"  {bloque_label}")
        lines.append("=" * 70)
        lines.append(f"  {'Equipo':<22} {'Grp':>4}  {'Semis':>7} "
                     f"{'Final':>7} {'Campeón':>9}")
        lines.append("  " + "-"*55)
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['team']:<22} {str(row.get('grupo','?')):>4}  "
                f"{row['prob_sf']:>6.1%} "
                f"{row['prob_final']:>6.1%} "
                f"{row['prob_winner']:>8.1%}"
            )

    # ── Candidatos al título ───────────────────────────────
    title_candidates = df[df["prob_winner"] >= 0.05]
    lines.append(f"\n\n{'='*70}")
    lines.append("  CANDIDATOS AL TÍTULO (prob >= 5%)")
    lines.append("=" * 70)
    for _, row in title_candidates.iterrows():
        bloque = row.get("bloque", "?")
        grupo  = row.get("grupo",  "?")
        lines.append(f"  {row['team']:<22} Grp {grupo} [{bloque}]  "
                     f"→ {row['prob_winner']:.1%}")

    # ── Favoritos por bloque ───────────────────────────────
    lines.append(f"\n\n{'='*70}")
    lines.append("  FAVORITO POR BLOQUE")
    lines.append("=" * 70)
    for bloque_tag in ["IZQ", "DER"]:
        sub = df[df["bloque"] == bloque_tag]
        if sub.empty:
            continue
        fav = sub.iloc[0]
        lines.append(
            f"  Bloque {bloque_tag}: {fav['team']:<18} "
            f"(campeón: {fav['prob_winner']:.1%} · "
            f"semis: {fav['prob_sf']:.1%})"
        )

    lines.append(f"\n\n{'='*70}")
    lines.append("  NOTAS")
    lines.append("=" * 70)
    lines.append("  - Los bloques IZQ y DER NO se cruzan hasta la final")
    lines.append("  - 6 equipos son placeholders (play-offs pendientes)")
    lines.append("  - Actualizar data/raw/groups_2026.csv cuando se")
    lines.append("    confirmen y correr: python 03_simulate.py")
    lines.append("  - El modelo predice partido a partido usando")
    lines.append("    ranking FIFA histórico + estadísticas mundialistas")
    lines.append("  - Walk-Forward AUC promedio: ~0.68 (ver metrics.json)")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print("\n" + report_text)

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

    # Paso 2: gráfica principal (barras apiladas por ronda)
    plot_predictions_by_round(df, paths)

    # Paso 3: top 16 candidatos
    plot_top_candidates(df, paths)

    # Paso 4: análisis por grupo (organizado por bloque)
    plot_group_analysis(df, paths)

    # Paso 5: bracket IZQ vs DER (nueva gráfica)
    plot_bracket_comparison(df, paths)

    # Paso 6: reporte en texto
    generate_text_report(df, paths)

    print("\n" + "="*55)
    print("  REPORTES GENERADOS")
    print("  outputs/plots/predictions_by_round.png")
    print("  outputs/plots/top_candidates.png")
    print("  outputs/plots/group_analysis.png")
    print("  outputs/plots/bracket_comparison.png  ← NUEVO")
    print("  outputs/report_2026.txt")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()