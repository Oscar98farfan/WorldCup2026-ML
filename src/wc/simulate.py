"""
Simulación Monte Carlo del Mundial 2026 con el cuadro oficial.

1. Se calcula UNA vez la matriz de probabilidades P(gana/empata/pierde)
   para todos los pares de equipos (48 × 48). Las features son fijas antes
   del torneo, así que no hace falta llamar al modelo dentro del bucle:
   la simulación pasa de minutos a segundos.
2. Cada simulación juega la fase de grupos (con empates), clasifica a los
   2 primeros de cada grupo + los 8 mejores terceros y recorre el cuadro
   oficial (bracket.py) hasta la final.
3. En eliminatoria un empate se decide por penales al 50/50.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import bracket, config
from .features import pair_features
from .model import MatchModel

ROUND_COLS = ["r32", "r16", "qf", "sf", "final", "champion"]


def probability_matrix(model: MatchModel, snapshots_2026: pd.DataFrame) -> dict:
    """{(A, B): (p_pierde, p_empata, p_gana)} para todos los pares."""
    snap = snapshots_2026.set_index("team")
    teams = list(snap.index)
    pairs = [(a, b) for a in teams for b in teams if a != b]
    a = snap.loc[[p[0] for p in pairs]].reset_index(drop=True)
    b = snap.loc[[p[1] for p in pairs]].reset_index(drop=True)
    proba = model.predict_proba(pair_features(a, b))
    return {pair: tuple(p) for pair, p in zip(pairs, proba)}


class TournamentSimulator:
    def __init__(self, groups: dict[str, list[str]], probs: dict, seed: int = config.RANDOM_SEED):
        self.groups = groups
        self.probs = probs
        self.rng = np.random.default_rng(seed)

    # -- partidos -----------------------------------------------------------
    def _group_match(self, a: str, b: str) -> int:
        """Devuelve 2 (gana A), 1 (empate) o 0 (gana B)."""
        p_loss, p_draw, _ = self.probs[(a, b)]
        u = self.rng.random()
        return 0 if u < p_loss else (1 if u < p_loss + p_draw else 2)

    def _knockout(self, a: str, b: str) -> str:
        p_loss, p_draw, p_win = self.probs[(a, b)]
        p_advance = p_win + 0.5 * p_draw          # penales 50/50
        return a if self.rng.random() < p_advance else b

    # -- fases --------------------------------------------------------------
    def _group_stage(self):
        """Tabla de cada grupo ordenada. Desempate: puntos y luego sorteo
        (no se simulan goles; ver limitaciones en el README)."""
        tables, thirds = {}, []
        for g, teams in self.groups.items():
            pts = dict.fromkeys(teams, 0)
            for a, b in combinations(teams, 2):
                r = self._group_match(a, b)
                pts[a] += {2: 3, 1: 1, 0: 0}[r]
                pts[b] += {2: 0, 1: 1, 0: 3}[r]
            tiebreak = self.rng.random(len(teams))
            order = sorted(teams, key=lambda t: (-pts[t], tiebreak[teams.index(t)]))
            tables[g] = order
            thirds.append((pts[order[2]], self.rng.random(), g))
        best_thirds = sorted(thirds, key=lambda x: (-x[0], x[1]))[:8]
        return tables, [g for _, _, g in best_thirds]

    def _resolve_slot(self, slot: str, tables, third_assign, match_id) -> str:
        if slot.startswith("3:"):
            return tables[third_assign[match_id]][2]
        pos, group = int(slot[0]), slot[1]
        return tables[group][pos - 1]

    def run_once(self) -> dict[str, str]:
        """Juega un torneo. Devuelve {equipo: ronda máxima alcanzada}."""
        tables, third_groups = self._group_stage()
        third_assign = bracket.assign_third_places(third_groups)

        reached = {}
        winners: dict[int, str] = {}
        for m, (sa, sb) in bracket.ROUND_OF_32.items():
            a = self._resolve_slot(sa, tables, third_assign, m)
            b = self._resolve_slot(sb, tables, third_assign, m)
            reached[a] = reached[b] = "r32"
            winners[m] = self._knockout(a, b)
        for rnd, fixtures in bracket.KNOCKOUT_ROUNDS:
            for m, (m1, m2) in fixtures.items():
                a, b = winners[m1], winners[m2]
                reached[a] = reached[b] = rnd
                winners[m] = self._knockout(a, b)
        reached[winners[104]] = "champion"
        return reached

    def run(self, n: int = config.N_SIMULATIONS) -> pd.DataFrame:
        teams = [t for ts in self.groups.values() for t in ts]
        idx = {r: i for i, r in enumerate(ROUND_COLS)}
        counts = {t: np.zeros(len(ROUND_COLS)) for t in teams}
        for _ in range(n):
            for team, rnd in self.run_once().items():
                counts[team][: idx[rnd] + 1] += 1      # llegar a QF implica R32 y R16
        group_of = {t: g for g, ts in self.groups.items() for t in ts}
        df = pd.DataFrame.from_dict(counts, orient="index", columns=[f"p_{c}" for c in ROUND_COLS]) / n
        df.insert(0, "group", df.index.map(group_of))
        return df.rename_axis("team").reset_index().sort_values("p_champion", ascending=False,
                                                                ignore_index=True)
