import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Optional, Dict, Any

class FPLPredictor:
    """
    Fantasy Premier League points predictor using Random Forest.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe containing FPL player statistics
    target : str
        Name of the target column to predict
    positions : Set[str] or List[str]
        Positions to filter (e.g., {'MID', 'FWD'})
    columns_to_drop : List[str], optional
        Additional columns to exclude from features
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    rf_params : dict, optional
        Parameters for RandomForestRegressor
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        positions: Set[str] | List[str],
        columns_to_drop: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        rf_params: Optional[Dict[str, Any]] = None
    ):
        self.data = data.copy()
        self.target = target
        self.positions = set(positions) if isinstance(positions, list) else positions
        self.test_size = test_size
        self.random_state = random_state
        
        # Default columns to drop
        self.default_columns_to_drop = [
            'event', 'season','element','value', 'player_name_id', 'position', 'minutes',
            'total_points', 'points_per90','expected_goals_per90','expected_assists_per90','expected_goals_conceded_per90', 
            'saves_per90', 'bps_per90', 'bonus_per90', 'cbi_per90', 'defensive_contribution_per90',
            'player_season_points_per90',
            'goals_scored', 'bonus', 'bps', 'clean_sheets', 'goals_conceded', 'cbi', 'defensive_contribution', 'recoveries','tackles',
            'saves',
            'was_home', 'expected_goals', 'expected_assists',
            'expected_goal_involvements', 'expected_goals_conceded',
            'team_name', 'opp_team_name',
            'saves_per90', 'bps_per90', 'bonus_per90', 'cbi_per90', 'defensive_contribution_per90',
            'recoveries_per90','tackles_per90',
            'player_season_minutes_total','expected_goal_involvements_per90_alltime_avg',
            'total_points_alltime_avg','minutes_alltime_avg','goals_scored_alltime_avg','goals_conceded_alltime_avg',
            'expected_goals_alltime_avg','expected_goal_involvements_alltime_avg','expected_assists_alltime_avg',
            'team_elo_alltime_avg','opp_team_elo_alltime_avg','expected_goals_conceded_alltime_avg','bps_alltime_avg','bonus_alltime_avg',
            'clean_sheets_alltime_avg',
            'minutes_per90_alltime_avg','opp_team_elo_per90_alltime_avg',
            'expected_goals_per90_alltime_avg','expected_assists_per90_alltime_avg','expected_goals_conceded_per90_alltime_avg','goals_scored_per90_alltime_avg','total_points_per90_alltime_avg','bps_per90_alltime_avg','team_elo_per90_alltime_avg',
            'goals_conceded_per90_alltime_avg','clean_sheets_per90_alltime_avg','bonus_per90_alltime_avg',
            # 'opp_team_elo_per90_running_avg_prev_1', 'team_elo_per90_running_avg_prev_1',
            # 'opp_team_elo_per90_running_avg_prev_2', 'team_elo_per90_running_avg_prev_2',
            'opp_team_elo_per90_running_avg_prev_3', 'team_elo_per90_running_avg_prev_3',
            'opp_team_elo_per90_running_avg_prev_5', 'team_elo_per90_running_avg_prev_5',
            'opp_team_elo_per90_running_avg_prev_10', 'team_elo_per90_running_avg_prev_10',
            # 'minutes_per90_running_avg_prev_1', 'minutes_per90_running_avg_prev_2',
            'minutes_per90_running_avg_prev_3', 'minutes_per90_running_avg_prev_5',
            'minutes_per90_running_avg_prev_10'
        ]
        
        # Add custom columns to drop
        if columns_to_drop:
            self.columns_to_drop = list(set(self.default_columns_to_drop + columns_to_drop))
        else:
            self.columns_to_drop = self.default_columns_to_drop
        
        # Ensure target is in columns to drop
        if self.target not in self.columns_to_drop:
            self.columns_to_drop.append(self.target)
        
        # Default RF parameters
        default_rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': random_state,
            'n_jobs': -1
        }
        
        # Update with custom parameters
        if rf_params:
            default_rf_params.update(rf_params)
        
        self.rf_params = default_rf_params
        
        # Initialize model and data containers
        self.model = None
        self.features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.feature_importance_df = None
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Filter positions and prepare features and target.
        
        Returns:
        --------
        X : pd.DataFrame
            Features dataframe
        y : pd.Series
            Target series
        """
        # Filter by positions
        df_filtered = self.data[self.data['position'].isin(self.positions)]
        
        # Define features
        self.features = df_filtered.columns.drop(self.columns_to_drop).tolist()
        
        # # Check NAs
        # nas = [(col, df_filtered[col].isna().sum()) for col in self.features]
        # nas_sorted = sorted(nas, key=lambda x: x[1], reverse=True)
        # for col, count in nas_sorted:
        #     print(f"{col}: {count}")

        # Remove rows with missing values
        df_clean = df_filtered[self.features + [self.target]].dropna()
        
        print(f"Cleaned df size: {len(df_clean)}/{len(df_filtered)}")
        
        X = df_clean[self.features]
        y = df_clean[self.target]
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split data into train and test sets.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
    def train(self) -> None:
        """Train the Random Forest model."""
        self.model = RandomForestRegressor(**self.rf_params)
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on train and test sets.
        
        Returns:
        --------
        y_pred_train : np.ndarray
            Training set predictions
        y_pred_test : np.ndarray
            Test set predictions
        """
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)
        
        return self.y_pred_train, self.y_pred_test
    
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on train and test sets.
        
        Returns:
        --------
        metrics : dict
            Dictionary containing train and test metrics
        """
        train_metrics = {
            'r2': r2_score(self.y_train, self.y_pred_train),
            'rmse': np.sqrt(mean_squared_error(self.y_train, self.y_pred_train)),
            'mae': mean_absolute_error(self.y_train, self.y_pred_train)
        }
        
        test_metrics = {
            'r2': r2_score(self.y_test, self.y_pred_test),
            'rmse': np.sqrt(mean_squared_error(self.y_test, self.y_pred_test)),
            'mae': mean_absolute_error(self.y_test, self.y_pred_test)
        }
        
        print("\nTraining Set Performance:")
        print(f"R² Score: {train_metrics['r2']:.4f}")
        print(f"RMSE: {train_metrics['rmse']:.4f}")
        print(f"MAE: {train_metrics['mae']:.4f}")
        
        print("\nTest Set Performance:")
        print(f"R² Score: {test_metrics['r2']:.4f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"MAE: {test_metrics['mae']:.4f}")
        
        return {'train': train_metrics, 'test': test_metrics}
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
        --------
        feature_importance_df : pd.DataFrame
            DataFrame with features and their importance scores
        """
        self.feature_importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(self.feature_importance_df.to_string(index=False))
        
        return self.feature_importance_df
    
    def plot_results(self, figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Visualize feature importance and actual vs predicted values.
        
        Parameters:
        -----------
        figsize : tuple, default=(14, 10)
            Figure size for plots
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Feature Importance
        axes[0].barh(self.feature_importance_df['feature'], 
                     self.feature_importance_df['importance'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Feature Importance')
        axes[0].invert_yaxis()
        
        # Plot 2: Actual vs Predicted
        axes[1].scatter(self.y_test, self.y_pred_test, alpha=0.5)
        axes[1].plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 
                     'r--', lw=2)
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Predicted')
        axes[1].set_title('Actual vs Predicted (Test Set)')
        
        plt.tight_layout()
        plt.show()
    
    def fit(self) -> 'FPLPredictor':
        """
        Complete training pipeline: prepare data, split, train, predict, evaluate.
        
        Returns:
        --------
        self : FPLPredictor
            Fitted predictor instance
        """
        X, y = self.prepare_data()
        self.split_data(X, y)
        self.train()
        self.predict()
        self.evaluate()
        self.get_feature_importance()
        
        return self
    
    def predict_new(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X_new : pd.DataFrame
            New data with same features as training data
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Ensure features match
        missing_features = set(self.features) - set(X_new.columns)
        if missing_features:
            raise ValueError(f"Missing features in new data: {missing_features}")
        
        return self.model.predict(X_new[self.features])