"""
Orquestación del pipeline. Cada paso lee y escribe artefactos en disco,
así que se puede correr suelto (`python main.py simulate`) o completo.
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from . import config, data, evaluate, export, features, model
from .simulate import TournamentSimulator, probability_matrix

log = logging.getLogger("wc")


# ---------------------------------------------------------------------------
def step_build() -> None:
    """Integra las fuentes y construye snapshots + dataset de partidos."""
    matches = data.load_matches()
    rankings = data.load_rankings()
    groups = data.load_groups_2026()
    intl = data.load_international_results()
    log.info("Elo: %s", "partidos internacionales" if intl is not None else "solo Mundiales")

    teams_by_year = {y: sorted(set(d["home_team"]) | set(d["away_team"]))
                     for y, d in matches.groupby("year")}
    teams_by_year[config.TARGET_YEAR] = [t for ts in groups.values() for t in ts]

    snapshots = features.build_team_snapshots(matches, rankings, teams_by_year, intl)
    dataset = features.build_match_dataset(matches, snapshots)

    # Chequeo de integración: ningún equipo de 2026 sin ranking FIFA
    missing = snapshots.query("year == @config.TARGET_YEAR and rank_known == 0")["team"].tolist()
    if missing:
        raise ValueError(f"Equipos 2026 sin ranking (revisar CANONICAL_NAMES): {missing}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    snapshots.to_csv(config.PROCESSED_DIR / "team_snapshots.csv", index=False)
    dataset.to_csv(config.PROCESSED_DIR / "match_dataset.csv", index=False)
    log.info("build: %d partidos, %d snapshots", len(dataset), len(snapshots))


# ---------------------------------------------------------------------------
def step_train() -> None:
    """Walk-forward de todos los candidatos, selección y modelo final."""
    dataset = pd.read_csv(config.PROCESSED_DIR / "match_dataset.csv")
    metrics, preds = model.walk_forward(dataset)
    best = model.select_best(metrics)
    log.info("walk-forward:\n%s", metrics.round(4).to_string(index=False))
    log.info("modelo seleccionado: %s", best)

    cand = next(c for c in model.CANDIDATES if c.name == best)
    final = model.fit_candidate(cand, dataset)          # entrena con 1930–2022
    model.save_model(final, metrics)

    # El baseline también se guarda: la simulación se compara siempre con él.
    base = next(c for c in model.CANDIDATES if c.name == "baseline_rank")
    import joblib
    joblib.dump(model.fit_candidate(base, dataset), config.MODELS_DIR / "baseline_rank.joblib")
    preds.to_csv(config.OUTPUTS_DIR / "walk_forward_predictions.csv", index=False)
    metrics.to_csv(config.OUTPUTS_DIR / "walk_forward_metrics.csv", index=False)


# ---------------------------------------------------------------------------
def _simulate(match_model, snapshots_2026, groups, n):
    probs = probability_matrix(match_model, snapshots_2026)
    return TournamentSimulator(groups, probs).run(n), probs


def step_simulate(n: int = config.N_SIMULATIONS) -> None:
    import joblib
    snapshots = pd.read_csv(config.PROCESSED_DIR / "team_snapshots.csv")
    snap26 = snapshots[snapshots["year"] == config.TARGET_YEAR]
    groups = data.load_groups_2026()

    sim, probs = _simulate(model.load_model(), snap26, groups, n)
    base_sim, _ = _simulate(joblib.load(config.MODELS_DIR / "baseline_rank.joblib"),
                            snap26, groups, n)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    base_sim.to_csv(config.OUTPUTS_DIR / "simulation_2026_baseline.csv", index=False)

    card = json.loads((config.MODELS_DIR / "model_card.json").read_text())
    export.export_predictions(sim, probs, card)
    log.info("simulate: top 8\n%s", sim.head(8).round(3).to_string(index=False))


# ---------------------------------------------------------------------------
def step_evaluate() -> dict:
    """Compara contra el Mundial 2026 real. Se ejecuta después de elegir el
    modelo: el torneo real es el test final, no un conjunto de ajuste."""
    import joblib
    snapshots = pd.read_csv(config.PROCESSED_DIR / "team_snapshots.csv")
    snap26 = snapshots[snapshots["year"] == config.TARGET_YEAR]
    actual = data.load_actual_2026()
    ko = data.load_actual_2026_knockout()

    sims = {
        "modelo_nuevo": pd.read_csv(config.OUTPUTS_DIR / "simulation_2026.csv"),
        "baseline_ranking": pd.read_csv(config.OUTPUTS_DIR / "simulation_2026_baseline.csv"),
        "version_anterior": evaluate.load_legacy_simulation(
            config.EXTERNAL_DIR / "legacy_v4_simulation.csv"),
    }
    result = {
        "torneo": {name: evaluate.tournament_brier(s, actual) for name, s in sims.items()},
        "eliminatorias": {
            "modelo_nuevo": evaluate.knockout_match_eval(model.load_model(), snap26, ko),
            "baseline_ranking": evaluate.knockout_match_eval(
                joblib.load(config.MODELS_DIR / "baseline_rank.joblib"), snap26, ko),
        },
    }
    (config.OUTPUTS_DIR / "evaluation_2026.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))

    # Reinyectar la evaluación en el JSON de la web
    payload_path = config.OUTPUTS_DIR / "predictions_2026.json"
    payload = json.loads(payload_path.read_text())
    payload["evaluation_2026"] = result
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from . import plots
    (config.OUTPUTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    plots.plot_evaluation_2026()
    plots.plot_top_candidates()
    plots.plot_predictions_by_round()
    plots.plot_top_champions()
    from .report import write_report
    write_report()
    log.info("evaluate: %s", {k: round(v["mean"], 4) for k, v in result["torneo"].items()})
    return result
