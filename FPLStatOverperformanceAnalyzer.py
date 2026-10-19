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
    alpha=0.05
):
    """
    df: DataFrame containing player match rows (must include 'team' column).
    """
    # Only consider games where the player actually stepped on the pitch
    played = df[df["minutes"] > 0].copy()
    result = {}

    # Update: Group by both player and team
    for (player_id, team), grp in played.groupby(["player_name_id", "team"]):
        player_factors = {}
        
        for actual_col, expected_col, key in stat_pairs:
            if grp[expected_col].sum() < min_xstat:
                player_factors[key] = 1.0
                continue
            
            analyzer = FPLStatOverperformanceAnalyzer(
                actual_array=grp[actual_col],
                expected_array=grp[expected_col],
                minutes_array=grp["minutes"],
                alpha=alpha
            )
            
            player_factors[key] = analyzer.adjustment_factor(
                taper_k_90s=taper_k_90s,
                only_if_significant=only_if_significant
            )
            
        # Store using a tuple key
        result[(player_id, team)] = player_factors

    return result


def apply_adjustments(prediction_data, adjustments, col_map=None):
    """
    prediction_data must include 'player_name_id' and 'team' columns.
    """
    if col_map is None:
        col_map = {"xg": "xgp90_pred", "xa": "xap90_pred"}

    for key, col in col_map.items():
        if col not in prediction_data.columns:
            continue
            
        # Update: Map factors using the (player_name_id, team) tuple
        factors = prediction_data.apply(
            lambda row: adjustments.get((row["player_name_id"], row["team"]), {}).get(key, 1.0),
            axis=1
        )
        
        prediction_data[f"{col}_raw"]             = prediction_data[col].copy()
        prediction_data[col]                      = prediction_data[col] * factors
        prediction_data[f"{key}_overperf_factor"] = factors

    return prediction_data