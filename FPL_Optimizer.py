import pandas as pd
from pulp import *
from typing import Dict, Optional

class FPLOptimizer:
    """
    Fantasy Premier League team optimizer using Linear Programming.
    Configurable to work with different datasets and column naming conventions.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        budget: float = 100.0,
        squad_size: int = 15,
        starting_xi_size: int = 11,
        max_per_team: int = 3,
        column_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the FPL optimizer.
        
        Args:
            df: DataFrame containing player data
            budget: Total budget for the squad
            squad_size: Number of players in the full squad
            starting_xi_size: Number of players in the starting XI
            max_per_team: Maximum players from the same team
            column_mapping: Dictionary mapping standard column names to dataset column names
                Expected keys: 'player_name', 'team_name', 'position', 'value', 'xpoints'
        """
        self.df = df.copy()
        self.budget = budget
        self.squad_size = squad_size
        self.starting_xi_size = starting_xi_size
        self.max_per_team = max_per_team
        
        # Default column mapping
        default_mapping = {
            'player_name': 'player_name_id',
            'team_name': 'team_name',
            'position': 'position',
            'value': 'value',
            'xpoints': 'total_xpoints'
        }
        
        # Use provided mapping or defaults
        self.cols = column_mapping if column_mapping else default_mapping
        
        # Validate that all required columns exist
        self._validate_columns()
        
        # Squad composition (position: count)
        self.squad_composition = {
            'GK': 2,
            'DEF': 5,
            'MID': 5,
            'FWD': 3
        }
        
        # Starting XI constraints (position: (min, max))
        self.starting_xi_constraints = {
            'GK': (1, 1),
            'DEF': (3, 5),
            'MID': (2, 5),
            'FWD': (1, 3)
        }
        
        # Weights for objective function
        self.starter_weight = 0.9
        self.bench_weight = 0.1
        
        # Optimization variables
        self.prob = None
        self.squad_vars = {}
        self.start_vars = {}
        self.selected_squad = None
        self.starting_xi = None
        
    def _validate_columns(self):
        """Validate that all required columns exist in the DataFrame."""
        missing_cols = []
        for standard_name, actual_name in self.cols.items():
            if actual_name not in self.df.columns:
                missing_cols.append(f"{standard_name} ('{actual_name}')")
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    def _get_position_indices(self, position: str):
        """Get DataFrame indices for players of a given position."""
        return self.df[self.df[self.cols['position']] == position].index
    
    def _create_variables(self):
        """Create binary decision variables for squad and starting XI selection."""
        self.squad_vars = {}
        self.start_vars = {}
        
        for idx in self.df.index:
            self.squad_vars[idx] = LpVariable(f"squad_{idx}", cat='Binary')
            self.start_vars[idx] = LpVariable(f"start_{idx}", cat='Binary')
    
    def _add_objective_function(self):
        """
        Add the objective function to maximize expected points.
        Points = (Starter xPts * starter_weight) + (Squad xPts * bench_weight)
        """
        self.prob += lpSum([
            (self.starter_weight * self.df.loc[idx, self.cols['xpoints']] * self.start_vars[idx]) + 
            (self.bench_weight * self.df.loc[idx, self.cols['xpoints']] * self.squad_vars[idx]) 
            for idx in self.df.index
        ]), "Maximize_Expected_Points"
    
    def _add_core_constraints(self):
        """Add core constraints for squad selection."""
        # Budget constraint
        self.prob += lpSum([
            self.df.loc[idx, self.cols['value']] * self.squad_vars[idx] 
            for idx in self.df.index
        ]) <= self.budget, "Budget_Constraint"
        
        # Squad size constraint
        self.prob += lpSum([
            self.squad_vars[idx] for idx in self.df.index
        ]) == self.squad_size, "Total_Squad_Size"
        
        # Starting XI size constraint
        self.prob += lpSum([
            self.start_vars[idx] for idx in self.df.index
        ]) == self.starting_xi_size, "Starting_XI_Size"
        
        # Can only start if in squad
        for idx in self.df.index:
            self.prob += self.start_vars[idx] <= self.squad_vars[idx], f"Start_Requires_Squad_{idx}"
    
    def _add_squad_position_constraints(self):
        """Add positional constraints for the full squad."""
        for position, count in self.squad_composition.items():
            position_indices = self._get_position_indices(position)
            self.prob += lpSum([
                self.squad_vars[idx] for idx in position_indices
            ]) == count, f"Squad_{position}s"
    
    def _add_starting_xi_position_constraints(self):
        """Add positional constraints for the starting XI."""
        for position, (min_count, max_count) in self.starting_xi_constraints.items():
            position_indices = self._get_position_indices(position)
            
            self.prob += lpSum([
                self.start_vars[idx] for idx in position_indices
            ]) >= min_count, f"Min_Start_{position}s"
            
            self.prob += lpSum([
                self.start_vars[idx] for idx in position_indices
            ]) <= max_count, f"Max_Start_{position}s"
    
    def _add_team_constraints(self):
        """Add constraint for maximum players from same team."""
        for team in self.df[self.cols['team_name']].unique():
            team_indices = self.df[self.df[self.cols['team_name']] == team].index
            self.prob += lpSum([
                self.squad_vars[idx] for idx in team_indices
            ]) <= self.max_per_team, f"Max_{self.max_per_team}_Per_Team_{team}"
    
    def optimize(self, verbose: bool = True):
        """
        Run the optimization to find the best squad.
        
        Args:
            verbose: Whether to print progress and results
            
        Returns:
            bool: True if optimization was successful, False otherwise
        """
        if verbose:
            print("=" * 80)
            print("FPL TEAM OPTIMIZER (Linear Programming)")
            print("=" * 80)
            print(f"\nBudget for {self.squad_size}-Player Squad: £{self.budget}m")
        
        # Create optimization problem
        self.prob = LpProblem("FPL_Squad_Selection", LpMaximize)
        
        # Build the model
        self._create_variables()
        self._add_objective_function()
        self._add_core_constraints()
        self._add_squad_position_constraints()
        self._add_starting_xi_position_constraints()
        self._add_team_constraints()
        
        # Solve
        if verbose:
            print("\nSolving optimization problem...")
        self.prob.solve(PULP_CBC_CMD(msg=0))
        
        # Extract results
        if LpStatus[self.prob.status] == 'Optimal':
            self.selected_squad = self.df[[
                self.squad_vars[idx].varValue == 1 for idx in self.df.index
            ]].copy()
            
            self.starting_xi = self.df[[
                self.start_vars[idx].varValue == 1 for idx in self.df.index
            ]].copy()
            
            if verbose:
                self._display_results()
            
            return True
        else:
            if verbose:
                print(f"\nOptimization Status: {LpStatus[self.prob.status]}")
                print(f"No valid {self.squad_size}-player squad found within constraints.")
            return False
    
    def _calculate_total_xpoints(self):
        """Calculate the total weighted expected points."""
        return lpSum([
            (self.starter_weight * self.df.loc[idx, self.cols['xpoints']] * self.start_vars[idx].varValue) + 
            (self.bench_weight * self.df.loc[idx, self.cols['xpoints']] * self.squad_vars[idx].varValue) 
            for idx in self.df.index
        ]).value()
    
    def _get_formation(self):
        """Get the starting XI formation string."""
        position_counts = self.starting_xi[self.cols['position']].value_counts()
        def_count = position_counts.get('DEF', 0)
        mid_count = position_counts.get('MID', 0)
        fwd_count = position_counts.get('FWD', 0)
        return f"{def_count}-{mid_count}-{fwd_count}"
    
    def _display_results(self):
        """Display the optimization results."""
        total_cost = self.selected_squad[self.cols['value']].sum()
        total_xpoints = self._calculate_total_xpoints()
        formation = self._get_formation()
        
        # Add starter indicator
        self.selected_squad['Starter'] = self.selected_squad.index.map(
            lambda idx: '✅' if self.start_vars[idx].varValue == 1 else '🪑'
        )
        
        print(f"\n{'=' * 80}")
        print(f"OPTIMAL {self.squad_size}-PLAYER SQUAD")
        print(f"{'=' * 80}")
        print(f"Starting XI Formation: {formation}")
        print(f"Total Squad Cost: £{total_cost:.1f}m / £{self.budget}m")
        print(f"Total Weighted Expected Points: {total_xpoints:.2f}")
        print(f"Remaining Budget: £{self.budget - total_cost:.1f}m")
        
        # Display by position
        position_labels = {
            'GK': f'GOALKEEPERS ({self.squad_composition["GK"]})',
            'DEF': f'DEFENDERS ({self.squad_composition["DEF"]})',
            'MID': f'MIDFIELDERS ({self.squad_composition["MID"]})',
            'FWD': f'FORWARDS ({self.squad_composition["FWD"]})'
        }
        
        for pos, label in position_labels.items():
            pos_players = self.selected_squad[
                self.selected_squad[self.cols['position']] == pos
            ].sort_values(by='Starter', ascending=False)
            
            if len(pos_players) > 0:
                print(f"\n{label:-^80}")
                print(f" {'Starter':<8} {'Player Name':<30} {'Team':<12} {'Cost':<6} {'xPts':>6} ")
                print("-" * 80)
                for _, player in pos_players.iterrows():
                    print(
                        f" {player['Starter']:<8} "
                        f"{player[self.cols['player_name']]:<30} "
                        f"{player[self.cols['team_name']]:<12} "
                        f"£{player[self.cols['value']]:<5.1f} "
                        f"{player[self.cols['xpoints']]:>6.2f} "
                    )
        
        print(f"\n{'=' * 80}")
        print(f"Optimization Status: {LpStatus[self.prob.status]}")
    
    def get_squad_dataframe(self, include_starter_flag: bool = True):
        """
        Get the selected squad as a DataFrame.
        
        Args:
            include_starter_flag: Whether to include a column indicating starters
            
        Returns:
            DataFrame with selected squad
        """
        if self.selected_squad is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        result = self.selected_squad.copy()
        
        if include_starter_flag:
            result['Starter'] = result.index.map(
                lambda idx: 1 if self.start_vars[idx].varValue == 1 else 0
            )
        
        return result
    
    def get_starting_xi_dataframe(self):
        """Get the starting XI as a DataFrame."""
        if self.starting_xi is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        return self.starting_xi.copy()

    def calculate_bench_boost_xpoints(self):
        """
        Calculate the expected points gained by playing the Bench Boost chip."""
        if self.selected_squad is None or self.starting_xi is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
 
        # Identify bench players as squad members not in the starting XI
        starting_indices = set(self.starting_xi.index)
        bench_players = self.selected_squad[
            ~self.selected_squad.index.isin(starting_indices)
        ].copy()
 
        bench_xpoints    = bench_players[self.cols['xpoints']].sum()
        already_counted  = self.bench_weight * bench_xpoints
        bench_boost_gain = (1 - self.bench_weight) * bench_xpoints

        return {
            'bench_players'   : bench_players,
            'bench_xpoints'   : bench_xpoints,
            'already_counted' : already_counted,
            'bench_boost_gain': bench_boost_gain,
        }