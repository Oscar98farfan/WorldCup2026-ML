"""Gráficas del reporte (matplotlib, estáticas para README/portafolio)."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
SERIES = {"version_anterior": "#e34948", "baseline_ranking": "#2a78d6", "modelo_nuevo": "#1baf7a"}
LABELS = {"version_anterior": "Versión anterior (con fuga)",
          "baseline_ranking": "Baseline: solo ranking FIFA",
          "modelo_nuevo": "Modelo nuevo (logreg)"}
ROUNDS = ["r32", "r16", "qf", "sf", "final", "champion"]
ROUND_ES = ["Pasa grupos", "Octavos", "Cuartos", "Semis", "Final", "Campeón"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_evaluation_2026() -> None:
    ev = json.loads((config.OUTPUTS_DIR / "evaluation_2026.json").read_text())["torneo"]
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    _style(ax)
    x, w = np.arange(len(ROUNDS)), 0.26
    for i, key in enumerate(SERIES):
        vals = [ev[key][r] for r in ROUNDS]
        ax.bar(x + (i - 1) * w, vals, w - 0.02, color=SERIES[key], label=LABELS[key])
    ax.set_xticks(x, ROUND_ES, color=INK)
    ax.set_ylabel("Brier score (menor = mejor)", color=MUTED)
    ax.set_title("Mundial 2026 real: error de las probabilidades por ronda",
                 loc="left", color=INK, fontsize=13)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(config.OUTPUTS_DIR / "plots" / "evaluation_2026.png", dpi=140)
    plt.close(fig)


def plot_top_candidates(top: int = 12) -> None:
    new = pd.read_csv(config.OUTPUTS_DIR / "simulation_2026.csv").set_index("team")
    base = pd.read_csv(config.OUTPUTS_DIR / "simulation_2026_baseline.csv").set_index("team")
    teams = (new["p_final"] + base["p_final"]).sort_values(ascending=False).head(top).index[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color="#ebeae6"); ax.grid(axis="y", visible=False)
    y, h = np.arange(len(teams)), 0.38
    ax.barh(y + h / 2, base.loc[teams, "p_final"], h - 0.03, color=SERIES["baseline_ranking"],
            label=LABELS["baseline_ranking"])
    ax.barh(y - h / 2, new.loc[teams, "p_final"], h - 0.03, color=SERIES["modelo_nuevo"],
            label=LABELS["modelo_nuevo"])
    ax.set_yticks(y, teams, color=INK)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Probabilidad de llegar a la final (antes del torneo)", loc="left",
                 color=INK, fontsize=13)
    ax.legend(frameon=False, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.OUTPUTS_DIR / "plots" / "prob_final_2026.png", dpi=140)
    plt.close(fig)
