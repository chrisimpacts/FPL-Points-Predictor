import numpy as np
import pandas as pd
from scipy.stats import poisson


class FPLStatOverperformanceAnalyzer:
    """
    Tests whether a player's actual stat (goals, assists, etc.) overperforms
    its expected counterpart (xG, xA) beyond Poisson randomness, and returns
    a shrinkage-adjusted multiplier for future predictions.
    """

    def __init__(self, actual, expected, alpha=0.05):
        self.actual   = np.asarray(actual,   dtype=float)
        self.expected = np.asarray(expected, dtype=float)
        self.alpha    = alpha
        self.total_actual   = self.actual.sum()
        self.total_expected = self.expected.sum()

    def _is_significant(self):
        if self.total_expected <= 0:
            return False
        p = poisson.sf(self.total_actual - 1, mu=self.total_expected)
        return p < self.alpha

    def adjustment_factor(self, prior_weight=0.3, only_if_significant=True):
        """
        Empirical Bayes factor shrunk toward 1.0.
        Returns 1.0 when there is insufficient data or the test is not significant.
        """
        if self.total_expected <= 0:
            return 1.0
        if only_if_significant and not self._is_significant():
            return 1.0
        observed_rate = self.total_actual / self.total_expected
        return prior_weight * 1.0 + (1.0 - prior_weight) * observed_rate


def build_player_adjustments(
    df,
    stat_pairs,
    prior_weight=0.3,
    only_if_significant=True,
    min_xstat=2.0,
    min_games=10,
):
    """
    Returns {player_name_id: {key: factor}} for each stat pair.

    stat_pairs: list of (actual_col, expected_col, output_key), e.g.
        [('goals_scored', 'expected_goals', 'xg'),
         ('assists',      'expected_assists', 'xa')]
    """
    played = df[df["minutes"] > 0]
    result = {}

    for player, grp in played.groupby("player_name_id"):
        factors = {}
        for actual_col, expected_col, key in stat_pairs:
            factors[key] = 1.0
            if len(grp) < min_games:
                continue
            if grp[expected_col].sum() < min_xstat:
                continue
            analyzer = FPLStatOverperformanceAnalyzer(
                actual=grp[actual_col].fillna(0).values,
                expected=grp[expected_col].fillna(0).values,
            )
            factors[key] = analyzer.adjustment_factor(
                prior_weight=prior_weight,
                only_if_significant=only_if_significant,
            )
        result[player] = factors

    return result


def apply_adjustments(prediction_data, adjustments, col_map=None):
    """
    Multiplies prediction columns by per-player factors.
    Saves originals as *_raw and adds *_overperf_factor columns.

    col_map defaults to {'xg': 'xgp90_pred', 'xa': 'xap90_pred'}.
    """
    if col_map is None:
        col_map = {"xg": "xgp90_pred", "xa": "xap90_pred"}

    for key, col in col_map.items():
        if col not in prediction_data.columns:
            continue
        factors = prediction_data["player_name_id"].map(
            lambda p: adjustments.get(p, {}).get(key, 1.0)
        )
        prediction_data[f"{col}_raw"]             = prediction_data[col].copy()
        prediction_data[col]                      = prediction_data[col] * factors
        prediction_data[f"{key}_overperf_factor"] = factors

    return prediction_data