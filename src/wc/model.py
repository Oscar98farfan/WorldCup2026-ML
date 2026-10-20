"""
Modelos de partido, validación walk-forward y persistencia.

Cambios de fondo respecto a la versión anterior:
- Objetivo de 3 clases (pierde / empata / gana) en vez de "gana vs. no
  gana". El simulador necesita la probabilidad de empate de forma
  explícita, no una "zona de empate" inventada a mano.
- Métricas probabilísticas (log loss y RPS) como métrica principal:
  el simulador consume probabilidades, así que importa que estén bien
  calibradas, no solo que el acierto sea alto.
- Toda versión se compara contra un baseline de "solo ranking FIFA".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import symmetrize


# ---------------------------------------------------------------------------
# Definición de candidatos
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    name: str
    features: list[str]
    train_from: int            # primer año de entrenamiento
    builder: callable          # fábrica del estimador sklearn
    description: str = ""


def _logreg(C: float = 1.0):
    # with_mean=False mantiene la simetría: una diferencia 0 sigue siendo 0.
    return make_pipeline(StandardScaler(with_mean=False),
                         LogisticRegression(C=C, max_iter=2000))


def _gboost():
    # Árboles poco profundos, muchas hojas mínimas y early stopping: con
    # ~1.000 partidos un boosting sin freno memoriza el pasado.
    return HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05,
                                          max_iter=300, min_samples_leaf=40,
                                          l2_regularization=1.0,
                                          early_stopping=True, random_state=0)


CANDIDATES = [
    Candidate("baseline_rank", config.BASELINE_FEATURES, 1994, _logreg,
              "Regresión logística con solo la diferencia de ranking FIFA"),
    Candidate("baseline_elo", ["d_elo"], 1930, _logreg,
              "Regresión logística con solo la diferencia de Elo"),
    Candidate("logreg_full", config.MATCH_FEATURES, 1930, lambda: _logreg(0.3),
              "Regresión logística regularizada con todas las features"),
    Candidate("gboost_full", config.MATCH_FEATURES, 1930, _gboost,
              "Gradient boosting regularizado con todas las features"),
]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def ranked_probability_score(y: np.ndarray, proba: np.ndarray) -> float:
    """RPS: métrica estándar para resultados ordenados (L < D < W).
    Penaliza menos predecir empate cuando ganó que predecir derrota."""
    onehot = np.eye(3)[y]
    cum_p, cum_y = np.cumsum(proba, axis=1), np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cum_p - cum_y)[:, :2] ** 2, axis=1) / 2))


def score(y: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "rps": ranked_probability_score(y, proba),
        "accuracy": float(accuracy_score(y, proba.argmax(axis=1))),
        "auc_win": float(roc_auc_score(y == 2, proba[:, 2])),
    }


# ---------------------------------------------------------------------------
# Predictor simétrico
# ---------------------------------------------------------------------------
@dataclass
class MatchModel:
    """Envuelve un estimador y garantiza P(A gana) = P(B pierde)."""
    estimator: object
    features: list[str]
    meta: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        p = self.estimator.predict_proba(X)
        p_rev = self.estimator.predict_proba(-X)[:, ::-1]
        return (p + p_rev) / 2


def fit_candidate(cand: Candidate, train: pd.DataFrame) -> MatchModel:
    data = symmetrize(train[train["year"] >= cand.train_from])
    est = cand.builder().fit(data[cand.features], data["outcome"])
    return MatchModel(est, cand.features, {"name": cand.name})


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def walk_forward(dataset: pd.DataFrame, candidates=CANDIDATES,
                 years=config.EVAL_YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Para cada año de test entrena solo con Mundiales anteriores.

    Devuelve (métricas agregadas por candidato, predicciones fuera de muestra).
    """
    preds = []
    for year in years:
        train = dataset[dataset["year"] < year]
        test = dataset[dataset["year"] == year]
        for cand in candidates:
            model = fit_candidate(cand, train)
            p = model.predict_proba(test)
            preds.append(test[["year", "home_team", "away_team", "outcome"]]
                         .assign(model=cand.name, p_loss=p[:, 0],
                                 p_draw=p[:, 1], p_win=p[:, 2]))
    preds = pd.concat(preds, ignore_index=True)

    rows = []
    for name, d in preds.groupby("model", sort=False):
        proba = d[["p_loss", "p_draw", "p_win"]].values
        overall = score(d["outcome"].values, proba)
        per_year = [score(g["outcome"].values, g[["p_loss", "p_draw", "p_win"]].values)["log_loss"]
                    for _, g in d.groupby("year")]
        rows.append({"model": name, **overall, "log_loss_std_by_year": float(np.std(per_year))})
    # Referencia trivial: frecuencias del entrenamiento (sin mirar equipos)
    freq = []
    for year in years:
        tr = symmetrize(dataset[dataset["year"] < year])["outcome"].value_counts(normalize=True)
        n = (dataset["year"] == year).sum()
        freq.append(np.tile([tr.get(0, 0), tr.get(1, 0), tr.get(2, 0)], (n, 1)))
    y_all = dataset[dataset["year"].isin(years)]["outcome"].values
    rows.append({"model": "climatology", **score(y_all, np.vstack(freq)),
                 "log_loss_std_by_year": np.nan})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True), preds


# Orden de complejidad: ante un empate técnico gana el más simple.
COMPLEXITY = ["baseline_rank", "baseline_elo", "logreg_full", "gboost_full"]


def select_best(metrics: pd.DataFrame, tolerance: float = 0.01) -> str:
    """Regla de selección (fijada antes de mirar 2026):
    1. Candidatos a menos de `tolerance` de log loss del mejor.
    2. Entre ellos, el más simple.
    Si ninguno mejora al baseline de ranking, gana el baseline."""
    m = metrics[metrics["model"] != "climatology"].set_index("model")["log_loss"]
    close = [name for name in COMPLEXITY if name in m and m[name] <= m.min() + tolerance]
    return close[0]


# ---------------------------------------------------------------------------
# Persistencia (pensada para una futura API)
# ---------------------------------------------------------------------------
def save_model(model: MatchModel, metrics: pd.DataFrame) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "match_model.joblib")
    card = {
        "model": model.meta["name"],
        "features": model.features,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation": "walk-forward " + ", ".join(map(str, config.EVAL_YEARS)),
        "metrics_walk_forward": metrics.to_dict(orient="records"),
    }
    (config.MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))


def load_model() -> MatchModel:
    return joblib.load(config.MODELS_DIR / "match_model.joblib")
