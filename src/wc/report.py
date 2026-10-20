"""Reporte de texto legible (equivalente al antiguo report_2026.txt)."""
from __future__ import annotations

import json

import pandas as pd

from . import config, data

COLS = [("p_r32", "Pasa grupos"), ("p_r16", "Octavos"), ("p_qf", "Cuartos"),
        ("p_sf", "Semis"), ("p_final", "Final"), ("p_champion", "Campeón")]
REAL = {0: "Grupos", 1: "Dieciseisavos", 2: "Octavos", 3: "Cuartos",
        4: "Semis", 5: "Final", 6: "CAMPEÓN"}


def _table(df: pd.DataFrame, real: dict) -> list[str]:
    head = f"  {'#':>3}  {'Equipo':<24}" + "".join(f"{n:>13}" for _, n in COLS) + f"   {'Real 2026':<14}"
    lines = [head, "  " + "-" * (len(head) - 2)]
    for i, r in enumerate(df.itertuples(), 1):
        row = f"  {i:>3}  {r.team:<24}" + "".join(f"{getattr(r, c):>13.1%}" for c, _ in COLS)
        lines.append(row + f"   {REAL.get(real.get(r.team), '?'):<14}")
    return lines


def write_report() -> None:
    sim = pd.read_csv(config.OUTPUTS_DIR / "simulation_2026.csv")
    base = pd.read_csv(config.OUTPUTS_DIR / "simulation_2026_baseline.csv")
    ev = json.loads((config.OUTPUTS_DIR / "evaluation_2026.json").read_text())
    card = json.loads((config.MODELS_DIR / "model_card.json").read_text())
    real = data.load_actual_2026().set_index("team")["stage"].to_dict()

    L = ["=" * 110, "  WORLDCUP2026-ML — PREDICCIONES MUNDIAL 2026", "=" * 110, "",
         f"  Modelo: {card['model']}   |   Validación: {card['validation']}",
         f"  Simulaciones Monte Carlo: {config.N_SIMULATIONS:,}   |   Cuadro oficial FIFA (partidos 73–104)", "",
         "  RESULTADO REAL: España campeona (1-0 a Argentina en la prórroga). Semis: Francia e Inglaterra.", "",
         "-" * 110, "  1. MODELO NUEVO — los 48 equipos ordenados por probabilidad de ser campeón", "-" * 110]
    L += _table(sim, real)
    L += ["", "-" * 110, "  2. POR GRUPO (modelo nuevo)", "-" * 110]
    for g, d in sim.groupby("group"):
        L.append(f"\n  Grupo {g}")
        L += _table(d.sort_values("p_r32", ascending=False), real)
    L += ["", "-" * 110, "  3. BASELINE (solo ranking FIFA) — top 16", "-" * 110]
    L += _table(base.head(16), real)

    t = ev["torneo"]
    L += ["", "-" * 110, "  4. EVALUACIÓN CONTRA EL MUNDIAL REAL (Brier medio por ronda, menor = mejor)", "-" * 110]
    for k, name in [("version_anterior", "Versión anterior (con fuga)"),
                    ("baseline_ranking", "Baseline solo ranking FIFA"), ("modelo_nuevo", "Modelo nuevo")]:
        L.append(f"  {name:<30} Brier {t[k]['mean']:.4f}   puesto del campeón: {t[k]['rank_of_champion']:>2}"
                 f"   puesto del subcampeón: {t[k]['rank_of_runner_up']:>2}")
    for k, name in [("baseline_ranking", "Baseline solo ranking FIFA"), ("modelo_nuevo", "Modelo nuevo")]:
        e = ev["eliminatorias"][k]
        L.append(f"  {name:<30} eliminatorias: el favorito ganó {e['favourite_won']} de {e['n_matches']}")
    L.append("")
    (config.OUTPUTS_DIR / "report_2026.txt").write_text("\n".join(L), encoding="utf-8")
