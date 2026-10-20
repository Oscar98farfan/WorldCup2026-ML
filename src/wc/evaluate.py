"""
Evaluación contra el Mundial 2026 real (test final, nunca usado para
elegir modelo ni features).

Dos niveles:
- Partido: ¿asigna el modelo más probabilidad al equipo que pasó en las
  31 eliminatorias reales? (log loss binario y acierto)
- Torneo: Brier score de las probabilidades por ronda frente a lo que
  pasó (¿llegó a cuartos? ¿a la final?...). Permite comparar versiones,
  incluida la anterior con fuga de datos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .data import canonical
from .features import pair_features
from .simulate import ROUND_COLS

STAGE_LEVEL = {c: i + 1 for i, c in enumerate(ROUND_COLS)}   # r32=1 … champion=6


def knockout_match_eval(model, snapshots_2026: pd.DataFrame, ko: pd.DataFrame) -> dict:
    snap = snapshots_2026.set_index("team")
    X = pair_features(snap.loc[ko["team_a"]].reset_index(drop=True),
                      snap.loc[ko["team_b"]].reset_index(drop=True))
    p = model.predict_proba(X)
    p_adv = p[:, 2] + 0.5 * p[:, 1]
    y = (ko["winner"] == ko["team_a"]).astype(int).values
    return {
        "n_matches": int(len(ko)),
        "log_loss": float(log_loss(y, p_adv, labels=[0, 1])),
        "accuracy": float(((p_adv > 0.5).astype(int) == y).mean()),
        "favourite_won": int(((p_adv > 0.5).astype(int) == y).sum()),
    }


def tournament_brier(sim: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Brier medio por ronda. Menor = mejor. 0.25 sería decir 50% a todo."""
    df = sim.merge(actual[["team", "stage"]], on="team", how="inner")
    out = {}
    for col, lvl in STAGE_LEVEL.items():
        y = (df["stage"] >= lvl).astype(int)
        out[col] = float(np.mean((df[f"p_{col}"] - y) ** 2))
    out["mean"] = float(np.mean(list(out.values())))
    out["p_actual_champion"] = float(df.loc[df["stage"] == 6, "p_champion"].sum())
    out["p_actual_finalists_reach_final"] = float(df.loc[df["stage"] >= 5, "p_final"].mean())
    ranking = df.sort_values("p_champion", ascending=False).reset_index(drop=True)
    out["rank_of_champion"] = int(ranking.index[ranking["stage"] == 6][0] + 1)
    out["rank_of_runner_up"] = int(ranking.index[ranking["stage"] == 5][0] + 1)
    return out


def load_legacy_simulation(path) -> pd.DataFrame:
    """Adapta el CSV de la versión anterior al formato actual."""
    old = pd.read_csv(path)
    old["team"] = old["team"].map(canonical)
    return old.rename(columns={"prob_r32": "p_r32", "prob_r16": "p_r16", "prob_qf": "p_qf",
                               "prob_sf": "p_sf", "prob_final": "p_final",
                               "prob_winner": "p_champion"})
