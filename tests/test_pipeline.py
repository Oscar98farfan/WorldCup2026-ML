"""
Tests que protegen las decisiones clave del proyecto.
Ejecutar con:  python -m pytest -q
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc import bracket, config, data, features  # noqa: E402
from wc.simulate import TournamentSimulator  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    matches = data.load_matches()
    rankings = data.load_rankings()
    teams_by_year = {y: sorted(set(d["home_team"]) | set(d["away_team"]))
                     for y, d in matches.groupby("year")}
    return matches, rankings, teams_by_year


def test_no_leakage_from_current_tournament(raw):
    """Las features de 2018 no deben cambiar si borramos los partidos de 2018
    en adelante: solo pueden depender del pasado."""
    matches, rankings, tby = raw
    full = features.build_team_snapshots(matches, rankings, tby)
    truncated = features.build_team_snapshots(matches[matches["year"] < 2018], rankings,
                                              {**tby})
    cols = ["elo", "hist_stage", "recent_stage", "experience"]
    a = full[full["year"] == 2018].set_index("team")[cols].sort_index()
    b = truncated[truncated["year"] == 2018].set_index("team")[cols].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_goals_are_not_features():
    assert not any("goal" in f or "score" in f or "penalty" in f
                   for f in config.ALL_PAIR_FEATURES)


def test_every_2026_team_has_fifa_ranking(raw):
    _, rankings, _ = raw
    teams = [t for ts in data.load_groups_2026().values() for t in ts]
    latest = rankings[rankings["rank_date"] == rankings["rank_date"].max()]
    assert set(teams) <= set(latest["team"])


def test_all_495_third_place_combinations_are_assignable():
    groups = "ABCDEFGHIJKL"
    for combo in combinations(groups, 8):
        assign = bracket.assign_third_places(list(combo))
        assert sorted(assign.values()) == sorted(combo)
        for match, g in assign.items():
            assert g in bracket.ROUND_OF_32[match][1]


def test_real_2026_combination_matches_fifa():
    assign = bracket.assign_third_places(list("BDEFIJKL"))
    assert assign[74] == "D" and assign[80] == "K" and assign[87] == "L"


def test_simulation_produces_32_qualifiers_and_one_champion():
    groups = data.load_groups_2026()
    teams = [t for ts in groups.values() for t in ts]
    even = {(a, b): (0.35, 0.30, 0.35) for a in teams for b in teams if a != b}
    reached = TournamentSimulator(groups, even, seed=1).run_once()
    assert len(reached) == 32
    assert list(reached.values()).count("champion") == 1
    assert list(reached.values()).count("final") == 1


def test_symmetrize_flips_features_and_label():
    df = pd.DataFrame({f: [1.0] for f in config.ALL_PAIR_FEATURES} | {"outcome": [2]})
    sym = features.symmetrize(df)
    assert sym.loc[1, "outcome"] == 0
    assert np.allclose(sym.loc[1, config.ALL_PAIR_FEATURES], -1.0)
