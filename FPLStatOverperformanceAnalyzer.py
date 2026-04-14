import numpy as np
from scipy.stats import poisson

class FPLStatOverperformanceAnalyzer:
    """
    Analyzes player performance using arrays of match data.
    Adjusts credibility based on the total sum of minutes played.
    """

    def __init__(self, actual_array, expected_array, minutes_array, alpha=0.05):
        # Convert to numpy arrays to ensure vector operations work
        self.actual = np.asarray(actual_array, dtype=float)
        self.expected = np.asarray(expected_array, dtype=float)
        self.minutes = np.asarray(minutes_array, dtype=float)
        
        self.alpha = alpha
        
        # Calculate totals for the statistical tests
        self.total_actual = np.nansum(self.actual)
        self.total_expected = np.nansum(self.expected)
        self.total_minutes = np.nansum(self.minutes)

    def _is_significant(self):
        if self.total_expected <= 0:
            return False
        # Poisson survival function: P(X >= total_actual)
        p = poisson.sf(self.total_actual - 1, mu=self.total_expected)
        return p < self.alpha

    def adjustment_factor(self, taper_k_90s=10, only_if_significant=True):
        """
        taper_k_90s : The number of 90s played to reach 50% credibility.
        """
        if self.total_expected <= 0:
            return 1.0
        
        if only_if_significant and not self._is_significant():
            return 1.0

        observed_rate = self.total_actual / self.total_expected
        
        # Calculate credibility based on '90s' played
        n_90s = self.total_minutes / 90.0
        credibility = n_90s / (n_90s + taper_k_90s)
        
        return (credibility * observed_rate) + ((1.0 - credibility) * 1.0)


def build_player_adjustments(
    df,
    stat_pairs,
    taper_k_90s=15.0, 
    only_if_significant=True,
    min_xstat=2.0,
):
    """
    df: DataFrame containing player match rows.
    stat_pairs: list of (actual_col, expected_col, key)
    taper_k_90s: number of 90s to reach 50% credibility (50% observed, 50% prior)
    only_if_significant: if True, only apply adjustments if overperformance is statistically significant
    min_xstat: minimum cumulative xStat to consider applying adjustments (prevents overfitting on small samples)
    """
    # Only consider games where the player actually stepped on the pitch
    played = df[df["minutes"] > 0].copy()
    result = {}

    for player_id, grp in played.groupby("player_name_id"):
        player_factors = {}
        
        for actual_col, expected_col, key in stat_pairs:
            # Check if cumulative xStat meets the floor
            if grp[expected_col].sum() < min_xstat:
                player_factors[key] = 1.0
                continue
            
            # Pass the arrays (Series) directly to the Analyzer
            analyzer = FPLStatOverperformanceAnalyzer(
                actual_array=grp[actual_col],
                expected_array=grp[expected_col],
                minutes_array=grp["minutes"]
            )
            
            player_factors[key] = analyzer.adjustment_factor(
                taper_k_90s=taper_k_90s,
                only_if_significant=only_if_significant
            )
            
        result[player_id] = player_factors

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