"""
Exportación para la futura web/API.

Un único JSON autocontenido que un frontend (HTML/JS) o una vista de
Django pueden servir sin volver a ejecutar el modelo:
- metadata del modelo y de la validación
- probabilidades por equipo y ronda
- matriz de enfrentamientos (para un "simulador de partido" en la web)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from . import config


def export_predictions(sim: pd.DataFrame, probs: dict, model_card: dict,
                       evaluation: dict | None = None) -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sim.to_csv(config.OUTPUTS_DIR / "simulation_2026.csv", index=False)

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {k: model_card[k] for k in ("model", "features", "validation")},
        "walk_forward": model_card["metrics_walk_forward"],
        "evaluation_2026": evaluation,
        "teams": sim.round(4).to_dict(orient="records"),
        "matchups": [
            {"team_a": a, "team_b": b, "p_loss": round(p[0], 4),
             "p_draw": round(p[1], 4), "p_win": round(p[2], 4)}
            for (a, b), p in probs.items() if a < b
        ],
    }
    (config.OUTPUTS_DIR / "predictions_2026.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))
