class FPLRunningAveragesCalculator:
    """
    A class to create running averages for Fantasy Premier League metrics with 
    optional normalization and advanced feature engineering.
    """
    
    def __init__(self, db_creator, current_season=20262027):
        """
        Initialize the calculator with a database connection.
        
        Parameters:
        -----------
        db_creator : object
            Database connection object with a run_sql method
        """
        self.db_creator = db_creator
        self.current_season = current_season
        self.base_fields = [
            'player_name_id', 'element', 'season', 'value', 'event', 'fixture','minutes', 'total_points',
            'team_elo', 'opp_team_elo', 'position', 'goals_scored', 'assists', 'bonus',
            'bps', 'clean_sheets', 'goals_conceded', 'was_home',
            'expected_goals', 'expected_assists', 'expected_goal_involvements',
            'expected_goals_conceded', 'team_name', 'opp_team_name', 
            'cbi', 'defensive_contribution', 'recoveries', 'tackles',
            'saves'
        ]
    
    def calculate(self, 
                  metrics_to_average,
                  window_sizes=[5, 10],
                  include_alltime=True,
                  include_per_90=True,
                  include_raw=True,
                  include_elo_adjusted=True,
                  include_squared_per_90=True,
                  interaction_pairs=None,
                  additional_fields=None):
        """
        Create running averages for specified metrics with optional normalization.
        Stops calculations at the most recent gameweek with total_points data.
        
        Parameters:
        -----------
        metrics_to_average : list
            List of metric names to calculate running averages for
        window_sizes : list, optional
            List of window sizes for rolling averages (default: [5, 10])
        include_alltime : bool, optional
            If True, include all-time averages (default: True)
        include_per_90 : bool, optional
            If True, include per-90-minute normalized metrics (default: True)
        include_raw : bool, optional
            If True, include raw (non-normalized) metrics (default: True)
        include_elo_adjusted : bool, optional
            If True, adjust metrics by opponent ELO difficulty.
            Uses ~30% adjustment per 100 ELO difference (default: True)
        include_squared_per_90 : bool, optional
            If True, include squared per-90 features for capturing 
            non-linear effects (default: True)
        interaction_pairs : list of tuples, optional
            List of (metric1, metric2) pairs to create interaction features.
            Example: [('expected_goals', 'expected_assists'), ('bps', 'minutes')]
            Creates features like metric1_x_metric2_per90_running_avg_prev_{window}
        additional_fields : list, optional
            Additional fields to include in the base selection
        
        Returns:
        --------
        DataFrame
            DataFrame with all requested running average features
        """
        select_parts = self._build_base_fields(additional_fields)
        select_parts.extend(self._build_season_totals())
        select_parts.extend(self._build_interaction_features(interaction_pairs, window_sizes, 
                                                             include_alltime, include_per_90, 
                                                             include_raw, include_elo_adjusted))
        select_parts.extend(self._build_metric_features(metrics_to_average, window_sizes,
                                                        include_alltime, include_per_90,
                                                        include_raw, include_elo_adjusted,
                                                        include_squared_per_90))
        
        query = self._build_query(select_parts)
        return self.db_creator.run_sql(query)
    
    def _build_base_fields(self, additional_fields=None):
        """Build the base field selection."""
        fields = self.base_fields.copy()
        if additional_fields:
            fields.extend(additional_fields)
        return fields
    
    def _build_season_totals(self):
        """Build season total and derived per-90 statistics."""
        return [
            """SUM(minutes) OVER (PARTITION BY player_name_id, season) AS player_season_minutes_total""",
            """(SUM(expected_goals) OVER (PARTITION BY player_name_id) / 
                NULLIF(SUM(goals_scored) OVER (PARTITION BY player_name_id), 0))
                    AS xg_over_g_ratio""",
            """(SUM(total_points) OVER (PARTITION BY player_name_id, season) / 
                NULLIF(SUM(minutes) OVER (PARTITION BY player_name_id, season), 0) * 90.0)
                    AS player_season_points_per90""",
            """CASE WHEN minutes > 0 THEN (total_points / minutes * 90.0) ELSE 0 END AS points_per90""",
            """CASE WHEN minutes > 0 THEN (expected_goals / minutes * 90.0) ELSE 0 END AS expected_goals_per90""",
            """CASE WHEN minutes > 0 THEN (expected_assists / minutes * 90.0) ELSE 0 END AS expected_assists_per90""",
            """CASE WHEN minutes > 0 THEN (expected_goals_conceded / minutes * 90.0) ELSE 0 END AS expected_goals_conceded_per90""",
            """CASE WHEN minutes > 0 THEN (saves / minutes * 90.0) ELSE 0 END AS saves_per90""",
            """CASE WHEN minutes > 0 THEN (bps / minutes * 90.0) ELSE 0 END AS bps_per90""",
            """CASE WHEN minutes > 0 THEN (bonus / minutes * 90.0) ELSE 0 END AS bonus_per90""",
            """CASE WHEN minutes > 0 THEN (cbi / minutes * 90.0) ELSE 0 END AS cbi_per90""",
            """CASE WHEN minutes > 0 THEN (defensive_contribution / minutes * 90.0) ELSE 0 END AS defensive_contribution_per90""",
            """CASE WHEN minutes > 0 THEN (recoveries / minutes * 90.0) ELSE 0 END AS recoveries_per90""",
            """CASE WHEN minutes > 0 THEN (tackles / minutes * 90.0) ELSE 0 END AS tackles_per90""",
            """team_elo - opp_team_elo AS elo_diff"""
        ]
    
    def _build_interaction_features(self, interaction_pairs, window_sizes, 
                                   include_alltime, include_per_90, 
                                   include_raw, include_elo_adjusted):
        """Build interaction feature SQL expressions."""
        features = []
        if not interaction_pairs:
            return features
        
        for metric1, metric2 in interaction_pairs:
            m1 = metric1.replace('expected_', 'x').replace('_contribution', '_cont')
            m2 = metric2.replace('expected_', 'x').replace('_contribution', '_cont')
            interaction_name = f"{m1}_x_{m2}"
            
            # All-time interaction averages
            if include_alltime:
                features.extend(self._build_alltime_interaction(
                    metric1, metric2, interaction_name, include_per_90, include_raw
                ))
            
            # Window-based interaction averages
            for window in window_sizes:
                features.extend(self._build_window_interaction(
                    metric1, metric2, interaction_name, window,
                    include_per_90, include_raw, include_elo_adjusted
                ))
        
        return features
    
    def _build_alltime_interaction(self, metric1, metric2, interaction_name, 
                                   include_per_90, include_raw):
        """Build all-time interaction features."""
        features = []
        
        if include_per_90:
            metric1_per90 = f"CASE WHEN minutes > 0 THEN ({metric1} / minutes * 90.0) ELSE 0 END"
            metric2_per90 = f"CASE WHEN minutes > 0 THEN ({metric2} / minutes * 90.0) ELSE 0 END"
            interaction_expr = f"({metric1_per90}) * ({metric2_per90})"
            avg_name = f"{interaction_name}_p90_at"
            features.append(f"""
                CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                    AVG(CASE WHEN total_points IS NOT NULL 
                        THEN {interaction_expr} ELSE NULL END) OVER (
                        PARTITION BY player_name_id
                        ORDER BY season ASC, event ASC, fixture ASC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    )
                END AS {avg_name}""")
        
        if include_raw:
            raw_interaction_expr = f"{metric1} * {metric2}"
            avg_name_raw = f"{interaction_name}_at"
            features.append(f"""
                CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                    AVG(CASE WHEN total_points IS NOT NULL 
                        THEN {raw_interaction_expr} ELSE NULL END) OVER (
                        PARTITION BY player_name_id
                        ORDER BY season ASC, event ASC, fixture ASC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    )
                END AS {avg_name_raw}""")
        
        return features
    
    def _build_window_interaction(self, metric1, metric2, interaction_name, window,
                                 include_per_90, include_raw, include_elo_adjusted):
        """Build window-based interaction features."""
        features = []
        
        if include_per_90:
            metric1_per90 = f"CASE WHEN minutes > 0 THEN ({metric1} / minutes * 90.0) ELSE 0 END"
            metric2_per90 = f"CASE WHEN minutes > 0 THEN ({metric2} / minutes * 90.0) ELSE 0 END"
            interaction_expr = f"({metric1_per90}) * ({metric2_per90})"
            avg_name = f"{interaction_name}_p90_{window}"
            features.append(f"""
                CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                    AVG(CASE WHEN total_points IS NOT NULL
                        THEN {interaction_expr} ELSE NULL END) OVER (
                        PARTITION BY player_name_id
                        ORDER BY season ASC, event ASC, fixture ASC
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    )
                END AS {avg_name}""")
        
        if include_raw:
            raw_interaction_expr = f"{metric1} * {metric2}"
            avg_name_raw = f"{interaction_name}_{window}"
            features.append(f"""
                CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                    AVG(CASE WHEN total_points IS NOT NULL
                        THEN {raw_interaction_expr} ELSE NULL END) OVER (
                        PARTITION BY player_name_id
                        ORDER BY season ASC, event ASC, fixture ASC
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    )
                END AS {avg_name_raw}""")
        
        if include_elo_adjusted and include_per_90:
            elo_difficulty_multiplier = f"POWER(1.3, (opp_team_elo - team_elo) / 100.0)"
            metric1_elo_per90 = f"CASE WHEN minutes > 0 THEN (({metric1} / minutes * 90.0) / {elo_difficulty_multiplier}) ELSE 0 END"
            metric2_elo_per90 = f"CASE WHEN minutes > 0 THEN (({metric2} / minutes * 90.0) / {elo_difficulty_multiplier}) ELSE 0 END"
            elo_interaction_expr = f"({metric1_elo_per90}) * ({metric2_elo_per90})"
            avg_name_elo = f"{interaction_name}_elo_p90_{window}"
            features.append(f"""
                CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                    AVG(CASE WHEN total_points IS NOT NULL
                        THEN {elo_interaction_expr} ELSE NULL END) OVER (
                        PARTITION BY player_name_id
                        ORDER BY season ASC, event ASC, fixture ASC
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    )
                END AS {avg_name_elo}""")
        
        return features
    
    def _build_metric_features(self, metrics_to_average, window_sizes,
                              include_alltime, include_per_90,
                              include_raw, include_elo_adjusted,
                              include_squared_per_90):
        """Build metric running average features."""
        features = []
        
        for metric in metrics_to_average:
            short_metric = metric.replace('expected_', 'x').replace('_contribution', '_cont')
            
            # All-time averages
            if include_alltime:
                features.extend(self._build_alltime_metric(
                    metric, short_metric, include_per_90, include_squared_per_90
                ))
            
            # Window-based averages
            for window in window_sizes:
                features.extend(self._build_window_metric(
                    metric, short_metric, window, include_raw,
                    include_per_90, include_elo_adjusted, include_squared_per_90
                ))
        
        return features
    
    def _build_alltime_metric(self, metric, short_metric, include_per_90, include_squared_per_90):
        """Build all-time metric features."""
        features = []
        
        avg_name = f"{short_metric}_at"
        features.append(f"""
            CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                AVG(CASE WHEN total_points IS NOT NULL 
                    THEN {metric} ELSE NULL END) OVER (
                    PARTITION BY player_name_id
                    ORDER BY season ASC, event ASC, fixture ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                )
            END AS {avg_name}""")
        
        if include_per_90:
            metric_expr = f"CASE WHEN minutes > 0 THEN ({metric} / minutes * 90.0) ELSE 0 END"
            avg_name_per90 = f"{short_metric}_p90_at"
            features.append(f"""
                CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                    AVG(CASE WHEN total_points IS NOT NULL 
                        THEN {metric_expr} ELSE NULL END) OVER (
                        PARTITION BY player_name_id
                        ORDER BY season ASC, event ASC, fixture ASC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    )
                END AS {avg_name_per90}""")
            
            if include_squared_per_90:
                squared_expr = f"CASE WHEN minutes > 0 THEN POWER(({metric} / minutes * 90.0), 2) ELSE NULL END"
                avg_name_squared = f"{short_metric}_p90sq_at"
                features.append(f"""
                    CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                        AVG(CASE WHEN total_points IS NOT NULL 
                            THEN {squared_expr} ELSE NULL END) OVER (
                            PARTITION BY player_name_id
                            ORDER BY season ASC, event ASC, fixture ASC
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                        )
                    END AS {avg_name_squared}""")
        
        return features
    
    def _build_window_metric(self, metric, short_metric, window, include_raw,
                            include_per_90, include_elo_adjusted, include_squared_per_90):
        """Build window-based metric features."""
        features = []
        elo_difficulty_multiplier = f"POWER(1.3, (opp_team_elo - team_elo) / 100.0)"
        
        # Raw running average
        if include_raw:
            avg_name = f"{short_metric}_{window}"
            features.append(f"""
                CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                    AVG(CASE WHEN total_points IS NOT NULL
                        THEN {metric} ELSE NULL END) OVER (
                        PARTITION BY player_name_id
                        ORDER BY season ASC, event ASC, fixture ASC
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    )
                END AS {avg_name}""")
        
        # Per-90 running average
        if include_per_90:
            metric_expr = f"CASE WHEN minutes > 0 THEN ({metric} / minutes * 90.0) ELSE 0 END"
            avg_name_per90 = f"{short_metric}_p90_{window}"
            features.append(f"""
                CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                    AVG(CASE WHEN total_points IS NOT NULL
                        THEN {metric_expr} ELSE NULL END) OVER (
                        PARTITION BY player_name_id
                        ORDER BY season ASC, event ASC, fixture ASC
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    )
                END AS {avg_name_per90}""")
            
            if include_squared_per_90:
                squared_expr = f"CASE WHEN minutes > 0 THEN POWER(({metric} / minutes * 90.0), 2) ELSE NULL END"
                avg_name_squared = f"{short_metric}_p90sq_{window}"
                features.append(f"""
                    CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                        AVG(CASE WHEN total_points IS NOT NULL
                            THEN {squared_expr} ELSE NULL END) OVER (
                            PARTITION BY player_name_id
                            ORDER BY season ASC, event ASC, fixture ASC
                            ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                        )
                    END AS {avg_name_squared}""")
        
        # ELO-adjusted running average
        if include_elo_adjusted:
            elo_adjusted_expr = f"{metric} / {elo_difficulty_multiplier}"
            avg_name_elo = f"{short_metric}_elo_{window}"
            features.append(f"""
                CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                    AVG(CASE WHEN total_points IS NOT NULL
                        THEN {elo_adjusted_expr} ELSE NULL END) OVER (
                        PARTITION BY player_name_id
                        ORDER BY season ASC, event ASC, fixture ASC
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    )
                END AS {avg_name_elo}""")
        
        # ELO-adjusted per-90
        if include_per_90 and include_elo_adjusted:
            elo_per90_expr = f"CASE WHEN minutes > 0 THEN (({metric} / minutes * 90.0) / {elo_difficulty_multiplier}) ELSE 0 END"
            avg_name_elo_per90 = f"{short_metric}_elo_p90_{window}"
            features.append(f"""
                CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                    AVG(CASE WHEN total_points IS NOT NULL
                        THEN {elo_per90_expr} ELSE NULL END) OVER (
                        PARTITION BY player_name_id
                        ORDER BY season ASC, event ASC
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    )
                END AS {avg_name_elo_per90}""")
            
            if include_squared_per_90:
                squared_elo_expr = f"CASE WHEN minutes > 0 THEN POWER((({metric} / minutes * 90.0) / {elo_difficulty_multiplier}), 2) ELSE NULL END"
                avg_name_elo_squared = f"{short_metric}_elo_p90sq_{window}"
                features.append(f"""
                    CASE WHEN (season < {self.current_season} OR (season = {self.current_season} AND event <= latest_completed_gw+1)) THEN
                        AVG(CASE WHEN total_points IS NOT NULL
                            THEN {squared_elo_expr} ELSE NULL END) OVER (
                            PARTITION BY player_name_id
                            ORDER BY season ASC, event ASC, fixture ASC
                            ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                        )
                    END AS {avg_name_elo_squared}""")
        
        return features
    
    def _build_query(self, select_parts):
        """Build the final SQL query."""
        select_clause = ',\n                        '.join(select_parts)
        
        return f"""
        WITH latest_gw AS (
            SELECT 
                MAX(event) AS latest_completed_gw
            FROM playergw
            WHERE season = {self.current_season} AND total_points IS NOT NULL
        ),
        base AS (
            SELECT 
                playergw.*, 
                latest_gw.latest_completed_gw
            FROM playergw
            CROSS JOIN latest_gw
        )

        SELECT *
        FROM (
            SELECT
                {select_clause}
            FROM base
        ) subquery
        
        ORDER BY player_name_id ASC, season ASC, event ASC, fixture ASC
        """
        # WHERE player_season_minutes_total > 0 # remove this until mid-season to avoid losing players with no minutes yet