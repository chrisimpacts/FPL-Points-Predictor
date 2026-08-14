# FPLPredictor.py 
    
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Optional, Dict, Any

class FPLPredictor:
    """
    Fantasy Premier League points predictor using Random Forest with GridSearch optimization.
    
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
    optimize : bool, default=False
        Whether to use GridSearchCV for hyperparameter optimization
    param_grid : dict, optional
        Parameter grid for GridSearchCV
    cv_folds : int, default=5
        Number of cross-validation folds for GridSearchCV
    max_nan_pct : float, optional
        Maximum percentage of NaN values allowed in features (0-100).
        Features exceeding this threshold will be excluded.
        If None, no filtering is applied.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        positions: Set[str] | List[str],
        columns_to_drop: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        rf_params: Optional[Dict[str, Any]] = None,
        optimize: bool = False,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 5,
        max_nan_pct: Optional[float] = None
    ):
        self.data = data.copy()
        self.target = target
        self.positions = set(positions) if isinstance(positions, list) else positions
        self.test_size = test_size
        self.random_state = random_state
        self.optimize = optimize
        self.cv_folds = cv_folds
        self.max_nan_pct = max_nan_pct
        
        # Default columns to drop
        self.default_columns_to_drop = ['event','fixture', 'season', 'element', 'value', 'player_name_id', 'position', 'minutes',
            'total_points', 'points_per90', 'expected_goals_per90', 'expected_assists_per90', 'expected_goals_conceded_per90',
            'saves_per90', 'bps_per90', 'bonus_per90', 'cbi_per90', 'defensive_contribution_per90',
            'player_season_points_per90',
            'goals_scored', 'bonus', 'bps', 'clean_sheets', 'goals_conceded', 'cbi', 'defensive_contribution', 'recoveries', 'tackles',
            'saves',
            'was_home', 'expected_goals', 'expected_assists',
            'expected_goal_involvements', 'expected_goals_conceded',
            'team_name', 'opp_team_name',
            'recoveries_per90', 'tackles_per90',
            'player_season_minutes_total',
            'xgoal_involvements_p90_at',
            'total_points_at', 'minutes_at', 'goals_scored_at', 'goals_conceded_at',
            'xgoals_at', 'xgoal_involvements_at', 'xassists_at',
            'team_elo_at', 'opp_team_elo_at', 'xgoals_conceded_at', 'bps_at', 'bonus_at',
            'clean_sheets_at',
            'minutes_p90_at', 'opp_team_elo_p90_at',
            'xgoals_p90_at', 'xassists_p90_at', 'xgoals_conceded_p90_at', 'goals_scored_p90_at', 'total_points_p90_at', 'bps_p90_at', 'team_elo_p90_at',
            'goals_conceded_p90_at', 'clean_sheets_p90_at', 'bonus_p90_at',
            'opp_team_elo_p90_3', 'team_elo_p90_3',
            'opp_team_elo_p90_5', 'team_elo_p90_5',
            'opp_team_elo_p90_10', 'team_elo_p90_10',
            'minutes_p90_3', 'minutes_p90_5',
            'minutes_p90_10']
        
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
        
        # Default parameter grid for GridSearchCV
        self.default_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Use custom param grid if provided
        self.param_grid = param_grid if param_grid else self.default_param_grid
        
        # Initialize model and data containers
        self.model = None
        self.grid_search = None
        self.best_params = None
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
        self.features = [col for col in df_filtered.columns if col not in set(self.columns_to_drop)]

        if self.max_nan_pct is not None:
            # Calculate NaN percentage for each feature
            nan_pct = (df_filtered[self.features].isna().sum() / len(df_filtered) * 100)
            # Keep only features below threshold
            self.features = nan_pct[nan_pct <= self.max_nan_pct].index.tolist()
            excluded_features = nan_pct[nan_pct > self.max_nan_pct].index.tolist()
            print(f"Excluded features (>{self.max_nan_pct}% NaN): {excluded_features}")
        
        # Remove rows with missing values
        df_clean = df_filtered[self.features + [self.target]].dropna()
        
        print(f"Cleaned df size: {len(df_clean)}/{len(df_filtered)}")
        
        X = df_clean[self.features]
        y = df_clean[self.target]
        
        return X, y
    
    def get_nan_summary(self) -> pd.DataFrame:
        """
        Get summary of NaN values in all potential features.
        Useful for deciding what max_nan_pct threshold to use.
        
        Returns:
        --------
        nan_summary : pd.DataFrame
            DataFrame with feature names, NaN counts, and percentages
        """
        # Filter by positions
        df_filtered = self.data[self.data['position'].isin(self.positions)]
        
        # Get potential features (before NaN filtering)
        potential_features = [col for col in df_filtered.columns if col not in self.columns_to_drop]
        
        # Calculate NaN statistics
        nan_counts = df_filtered[potential_features].isna().sum()
        nan_pct = (nan_counts / len(df_filtered) * 100).round(2)
        
        nan_summary = pd.DataFrame({
            'feature': potential_features,
            'nan_count': nan_counts.values,
            'nan_pct': nan_pct.values,
            'non_null_count': len(df_filtered) - nan_counts.values
        }).sort_values('nan_pct', ascending=False)
        
        print(f"\nNaN Summary for {len(potential_features)} potential features:")
        print("=" * 80)
        print(f"Total rows: {len(df_filtered)}")
        print(f"\nFeatures with NaN values:")
        print(nan_summary[nan_summary['nan_count'] > 0].to_string(index=False))
        
        print(f"\nFeatures with no NaN values: {(nan_summary['nan_count'] == 0).sum()}")
        
        return nan_summary
    
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
        """Train the Random Forest model with optional GridSearchCV optimization."""
        if self.optimize:
            print("\nPerforming GridSearchCV hyperparameter optimization...")
            print(f"Parameter grid: {self.param_grid}")
            print(f"Cross-validation folds: {self.cv_folds}")
            
            # Create base model
            base_model = RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Perform grid search
            self.grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=self.param_grid,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2
            )
            
            self.grid_search.fit(self.X_train, self.y_train)
            
            # Get best model and parameters
            self.model = self.grid_search.best_estimator_
            self.best_params = self.grid_search.best_params_
            
            print("\nBest parameters found:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
            print(f"\nBest CV RMSE: {np.sqrt(-self.grid_search.best_score_):.4f}")
            
        else:
            # Train with specified parameters
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
    
    def get_best_params_as_grid(self) -> Dict[str, List]:
        """
        Return the best parameters from GridSearchCV in grid format.
        Useful for copying to use as a new param_grid or for final model training.
        
        Returns:
        --------
        param_grid : dict
            Best parameters formatted as a grid (each value in a list)
        """
        if not self.optimize or self.best_params is None:
            print("No optimization was performed. Use optimize=True to run GridSearchCV.")
            return {}
        
        # Convert best params to grid format (wrap each value in a list)
        param_grid = {k: [v] for k, v in self.best_params.items()}
        
        print("\nBest Parameters as Grid:")
        print("param_grid = {")
        for key, value in param_grid.items():
            print(f"    '{key}': {value},")
        print("}")
        
        return param_grid
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to display
        
        Returns:
        --------
        feature_importance_df : pd.DataFrame
            DataFrame with features and their importance scores
        """
        self.feature_importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Feature Importances:")
        print(self.feature_importance_df.head(top_n).to_string(index=False))
        
        return self.feature_importance_df
    
    def plot_results(self, figsize: Tuple[int, int] = (16, 6), top_n: int = 15) -> None:
        """
        Visualize feature importance and actual vs predicted values.
        
        Parameters:
        -----------
        figsize : tuple, default=(16, 6)
            Figure size for plots
        top_n : int, default=15
            Number of top features to display in importance plot
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Top N Feature Importance
        top_features = self.feature_importance_df.head(top_n)
        axes[0].barh(range(len(top_features)), top_features['importance'])
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['feature'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title(f'Top {top_n} Feature Importance')
        axes[0].invert_yaxis()
        
        # Plot 2: Actual vs Predicted
        axes[1].scatter(self.y_test, self.y_pred_test, alpha=0.5)
        axes[1].plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 
                     'r--', lw=2)
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Predicted')
        axes[1].set_title('Actual vs Predicted (Test Set)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cv_results(self) -> None:
        """Plot GridSearchCV results if optimization was performed."""
        if not self.optimize or self.grid_search is None:
            print("No GridSearchCV results to plot. Set optimize=True.")
            return
        
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        
        # Plot mean test scores
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scores = -cv_results['mean_test_score']  # Convert negative MSE to positive
        scores_rmse = np.sqrt(scores)
        
        ax.plot(range(len(scores_rmse)), scores_rmse, 'o-')
        ax.axhline(y=np.sqrt(-self.grid_search.best_score_), 
                   color='r', linestyle='--', 
                   label=f'Best RMSE: {np.sqrt(-self.grid_search.best_score_):.4f}')
        ax.set_xlabel('Parameter Combination Index')
        ax.set_ylabel('RMSE (Cross-Validation)')
        ax.set_title('GridSearchCV Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
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
    
    def get_largest_errors(self, n: int = 20, dataset: str = 'test', 
                          return_full_data: bool = False) -> pd.DataFrame:
        """
        Get rows with largest prediction errors.
        
        Parameters:
        -----------
        n : int, default=20
            Number of rows to return
        dataset : str, default='test'
            Which dataset to analyze ('test' or 'train')
        return_full_data : bool, default=False
            If True, return all original columns; if False, return only key columns
            
        Returns:
        --------
        error_df : pd.DataFrame
            DataFrame with largest errors, sorted by absolute error
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Select the appropriate dataset
        if dataset == 'test':
            X = self.X_test
            y_actual = self.y_test
            y_pred = self.y_pred_test
        elif dataset == 'train':
            X = self.X_train
            y_actual = self.y_train
            y_pred = self.y_pred_train
        else:
            raise ValueError("dataset must be 'test' or 'train'")
        
        # Calculate errors
        errors = y_actual - y_pred
        abs_errors = np.abs(errors)
        
        # Get original data for these indices (always, to access dropped columns)
        original_indices = X.index
        error_df = self.data.loc[original_indices].copy()
        
        # Add prediction columns
        error_df['actual'] = y_actual.values
        error_df['predicted'] = y_pred
        error_df['error'] = errors.values
        error_df['abs_error'] = abs_errors.values
        error_df['pct_error'] = (errors.values / y_actual.values * 100).round(2)
        
        # Sort by absolute error and get top n
        error_df = error_df.sort_values('abs_error', ascending=False).head(n)
        
        # Display summary
        print(f"\nTop {n} Largest Prediction Errors ({dataset.upper()} set):")
        print("=" * 80)
        
        # Build display columns - prioritize key info columns
        display_cols = []
        key_info_cols = ['player_name_id', 'position', 'team_name', 'opp_team_name', 
                        'event', 'season', 'minutes']
        
        for col in key_info_cols:
            if col in error_df.columns:
                display_cols.append(col)
        
        # Add prediction columns
        display_cols.extend(['actual', 'predicted', 'error', 'abs_error', 'pct_error'])
        
        available_display_cols = [col for col in display_cols if col in error_df.columns]
        print(error_df[available_display_cols].to_string(index=False))
        
        # Return appropriate columns based on return_full_data flag
        if not return_full_data:
            # Return just key columns + predictions
            key_cols = [col for col in key_info_cols if col in error_df.columns]
            return_cols = key_cols + ['actual', 'predicted', 'error', 'abs_error', 'pct_error']
            return error_df[return_cols]
        
        return error_df
    
    def analyze_errors(self, dataset: str = 'test') -> Dict[str, Any]:
        """
        Comprehensive error analysis including statistics and visualizations.
        
        Parameters:
        -----------
        dataset : str, default='test'
            Which dataset to analyze ('test' or 'train')
            
        Returns:
        --------
        analysis : dict
            Dictionary containing error statistics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Select the appropriate dataset
        if dataset == 'test':
            y_actual = self.y_test
            y_pred = self.y_pred_test
        elif dataset == 'train':
            y_actual = self.y_train
            y_pred = self.y_pred_train
        else:
            raise ValueError("dataset must be 'test' or 'train'")
        
        # Calculate errors
        errors = y_actual - y_pred
        abs_errors = np.abs(errors)
        
        # Calculate statistics
        analysis = {
            'mean_error': errors.mean(),
            'median_error': np.median(errors),
            'std_error': errors.std(),
            'mean_abs_error': abs_errors.mean(),
            'median_abs_error': np.median(abs_errors),
            'max_overpredict': errors.min(),  # Most negative error
            'max_underpredict': errors.max(),  # Most positive error
            'max_abs_error': abs_errors.max()
        }
        
        print(f"\nError Analysis ({dataset.upper()} set):")
        print("=" * 80)
        print(f"Mean Error: {analysis['mean_error']:.4f}")
        print(f"Median Error: {analysis['median_error']:.4f}")
        print(f"Std Dev of Errors: {analysis['std_error']:.4f}")
        print(f"Mean Absolute Error: {analysis['mean_abs_error']:.4f}")
        print(f"Median Absolute Error: {analysis['median_abs_error']:.4f}")
        print(f"Max Over-prediction: {analysis['max_overpredict']:.4f}")
        print(f"Max Under-prediction: {analysis['max_underpredict']:.4f}")
        print(f"Max Absolute Error: {analysis['max_abs_error']:.4f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Error distribution
        axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Prediction Error (Actual - Predicted)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residual plot
        axes[0, 1].scatter(y_pred, errors, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Value')
        axes[0, 1].set_ylabel('Residual (Actual - Predicted)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Absolute error distribution
        axes[1, 0].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Absolute Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Absolute Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Q-Q plot for error normality
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Error Normality Check)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return analysis
    
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