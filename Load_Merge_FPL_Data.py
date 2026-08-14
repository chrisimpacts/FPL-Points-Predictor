import pandas as pd
import requests
from typing import Optional, List, Tuple
from FPLDataLoader import FPLDataLoader
from FPLAPIDataFetcher import FPLAPIDataFetcher

import pandas as pd
import requests
from typing import Optional, List, Tuple
from FPLDataLoader import FPLDataLoader
from FPLAPIDataFetcher import FPLAPIDataFetcher


class FPLDataManager:
    """
    Manages loading and merging FPL data across multiple seasons.
    Handles historical seasons via CSV files, 24/25 via manual merge, and 25/26 via API.
    """
    
    def __init__(self, current_season: int = 20262027):
        self.seasons_data = {}
        self.fixtures_data = {}
        self.merged_data = {}
        self.current_season = current_season
        
    def load_historical_season(self, season: int, gw_url: str, fixtures_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load a historical season's data from CSV files.
        
        Args:
            season: Season identifier (e.g., 20182019)
            gw_url: URL or path to gameweek data
            fixtures_path: Path to fixtures CSV file
            
        Returns:
            Tuple of (fpl_data, fixtures, merged_data)
        """
        loader = FPLDataLoader(
            season=season,
            gw_url=gw_url,
            fixtures_path=fixtures_path
        )
        
        fpl_data = loader.load_fpl_data()
        fixtures = loader.load_fixtures()
        fixtures_with_elo = loader.add_elo_to_fixtures()
        merged = loader.merge_data()
        
        self.seasons_data[season] = fpl_data
        self.fixtures_data[season] = fixtures_with_elo
        self.merged_data[season] = merged
        
        return fpl_data, fixtures_with_elo, merged
    
    def load_2024_25_season(self, gw1_21_path: str, gw22_38_path: str, 
                            fixtures_path: str, output_path: str = "data/vaastav2425playergwsreworked.csv") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load 2024-25 season with manual merge of split gameweek files.
        
        Args:
            gw1_21_path: Path to gameweeks 1-21 CSV
            gw22_38_path: Path to gameweeks 22-38 CSV
            fixtures_path: Path to fixtures CSV
            output_path: Path to save merged gameweeks
            
        Returns:
            Tuple of (fpl_data, fixtures, merged_data)
        """
        # Manually merge the split gameweek files
        gws1to21 = pd.read_csv(gw1_21_path, encoding='latin1')
        gws22to38 = pd.read_csv(gw22_38_path, encoding='latin1')
        fpl24_reworked = pd.concat([gws1to21, gws22to38], axis=0)
        fpl24_reworked.to_csv(output_path, index=False)
        
        # Load using standard loader
        loader = FPLDataLoader(
            season=20242025,
            gw_url=output_path,
            fixtures_path=fixtures_path
        )
        
        fpl_data = loader.load_fpl_data()
        fixtures = loader.load_fixtures()
        fixtures_with_elo = loader.add_elo_to_fixtures()
        merged = loader.merge_data()
        
        self.seasons_data[20242025] = fpl_data
        self.fixtures_data[20242025] = fixtures_with_elo
        self.merged_data[20242025] = merged
        
        return fpl_data, fixtures_with_elo, merged
    
    def load_current_season_from_api(self, prediction_gw: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load current season data from FPL API.
        
        Args:
            prediction_gw: The gameweek we are predicting for. If prediction_gw == 1, 
                           no gameweeks have been played, so we use bootstrap data.
        
        Returns:
            Tuple of (fpl_data, fixtures, merged_data, all_fixtures_for_current_players)
        """
        # Fetch current season data from API
        fpl_fetcher = FPLAPIDataFetcher(season=self.current_season)
        current_seasondf_api = fpl_fetcher.load_season_data(verbose=True)
        
        bootstrap_players = None
        # If prediction_gw is 1, 0 gameweeks have been played, use bootstrap data
        if prediction_gw == 1:
            print("Prediction GW is 1 (0 gameweeks played): fetching bootstrap player list...")
            bootstrap_players = fpl_fetcher.get_player_data()
        
        # Fetch fixtures from API
        fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
        response = requests.get(fixtures_url)
        response.raise_for_status()
        current_fix_api = pd.DataFrame(response.json())
        
        # Load using FPLDataLoader with API data
        loader = FPLDataLoader(
            season=self.current_season,
            gw_df=current_seasondf_api,
            fixtures_df=current_fix_api
        )
        
        fpl_data = loader.load_fpl_data()
        fixtures = loader.load_fixtures()
        fixtures_with_elo = loader.add_elo_to_fixtures()
        merged = loader.merge_data()
        
        # Create home and away fixtures for current season projections
        all_fixtures = self._create_home_away_fixtures(fixtures_with_elo)
        
        # Create player-fixture combinations for rest of season
        current_year_players_allfixtures = self._create_player_fixture_projections(
            loader, fpl_data, all_fixtures, fixtures_with_elo, bootstrap_players
        )
        
        self.seasons_data[self.current_season] = fpl_data
        self.fixtures_data[self.current_season] = fixtures_with_elo
        self.merged_data[self.current_season] = merged
        
        return fpl_data, fixtures_with_elo, merged, current_year_players_allfixtures
    
    def _create_home_away_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """
        Create separate home and away fixture records from combined fixtures.
        
        Args:
            fixtures: Combined fixtures dataframe
            
        Returns:
            Dataframe with separate records for home and away teams
        """
        home_fixtures = fixtures.copy().rename(columns={
            'team_a': 'opp_team',
            'fixture_team_a_score': 'fixture_opp_team_score',
            'team_h': 'team',
            'fixture_team_h_score': 'fixture_team_score',
            'team_h_difficulty': 'team_difficulty',
            'team_a_difficulty': 'opp_team_difficulty',
            'away_team_name': 'opp_team_name',
            'home_team_name': 'team_name',
            'home_team_elo': 'team_elo',
            'away_team_elo': 'opp_team_elo',
            'team': 'team_code'
        })
        
        away_fixtures = fixtures.copy().rename(columns={
            'team_a': 'team',
            'fixture_team_a_score': 'fixture_team_score',
            'team_h': 'opp_team',
            'fixture_team_h_score': 'fixture_opp_team_score',
            'team_h_difficulty': 'opp_team_difficulty',
            'team_a_difficulty': 'team_difficulty',
            'away_team_name': 'team_name',
            'home_team_name': 'opp_team_name',
            'home_team_elo': 'opp_team_elo',
            'away_team_elo': 'team_elo',
            'team': 'team_code'
        })
        
        all_fixtures = pd.concat([home_fixtures, away_fixtures], join='inner', axis=0)
        return all_fixtures
    
    def _create_player_fixture_projections(self, loader: FPLDataLoader, 
                                          fpl_data: pd.DataFrame, 
                                          all_fixtures: pd.DataFrame,
                                          fixtures_with_elo: pd.DataFrame,
                                          bootstrap_players: pd.DataFrame = None
                                          ) -> pd.DataFrame:
        """
        Create player-fixture combinations for rest of current season projections.
        
        Args:
            loader: FPLDataLoader instance
            fpl_data: Player gameweek data
            all_fixtures: Combined home/away fixtures
            fixtures_with_elo: Fixtures dataframe with ELO ratings
            bootstrap_players: Base player data from the FPL API bootstrap-static endpoint
            
        Returns:
            Dataframe with all player-fixture combinations
        """
        # Use bootstrap data if explicitly provided (e.g. pre-season / GW1 cold start)
        if bootstrap_players is not None and not bootstrap_players.empty:
            playerfix = bootstrap_players.copy()
            playerfix['player_name_id'] = playerfix['first_name'] + ' ' + playerfix['second_name']
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            playerfix['position'] = playerfix['element_type'].map(position_map)
            playerfix['season'] = loader.season
            playerfix['element'] = playerfix['id']
            
            playerfix = playerfix[['player_name_id', 'element', 'team', 'position', 'season']].drop_duplicates()
        else:
            # Otherwise, use the original rich historical method
            cols = ['player_name_id', 'team', 'position', 'season']
            if 'element' in fpl_data.columns:
                cols.append('element')
            playerfix = fpl_data[cols].drop_duplicates()

        # fixture ID is 'fixture_id' in all_fixtures (renamed by load_fixtures)
        # and 'fixture' in fpl_data (direct from FPL API history)
        team_fixtures = all_fixtures[['event', 'team', 'fixture_id']].drop_duplicates()
        playerfix = playerfix.merge(team_fixtures, on='team', how='inner')

        all_playerfix = playerfix.merge(
            fpl_data,
            left_on=['player_name_id', 'event', 'fixture_id'],
            right_on=['player_name_id', 'gw', 'fixture'],
            suffixes=('', '_gw'),
            how='left'
        )

        current_year_players_allfixtures = all_playerfix.merge(
            all_fixtures,
            on=['event', 'team', 'fixture_id'],
            suffixes=('', '_fixtures'),
        )

        # make sure fixture column is filled, so that deduplication doesn't remove double gameweeks
        current_year_players_allfixtures['fixture'] = (
            current_year_players_allfixtures['fixture']
            .fillna(current_year_players_allfixtures['fixture_id'])
            .astype(int)
        )

        return current_year_players_allfixtures
    
    def load_all_seasons(self) -> pd.DataFrame:
        """
        Load all seasons from 2018-19 through to current season.
        
        Returns:
            Combined dataframe with all seasons including rest-of-season projections
        """
        # Load historical seasons (2018-19 through 2023-24)
        historical_seasons = [
            (20182019, "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2018-19/gws/merged_gw.csv", 
             r"data\fixtures1819.csv"),
            (20192020, "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2019-20/gws/merged_gw.csv",
             r"data\fixtures1920.csv"),
            (20202021, "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2020-21/gws/merged_gw.csv",
             r"data\fixtures2021.csv"),
            (20212022, "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/gws/merged_gw.csv",
             r"data\fixtures2122.csv"),
            (20222023, "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/gws/merged_gw.csv",
             r"data\fixtures2223.csv"),
            (20232024, "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv",
             r"data\fixtures2324.csv"),
             # 20242025: Load 2024-25 season with manual merge below
             (20252026, "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv",
             r"data\fixtures2526.csv"),
        ]
        
        for season, gw_url, fixtures_path in historical_seasons:
            print(f"Loading season {season}...")
            self.load_historical_season(season, gw_url, fixtures_path)
        
        # Load 2024-25 season with manual merge
        print("Loading season 2024-25 (manual merge)...")
        self.load_2024_25_season(
            r"data\vaastavplayergwsmerged_gw1-21.csv",
            r"data\vaastavplayergwsmerged_gw22-38.csv",
            r"data\fixtures2425.csv"
        )
        
        # Load current season from API
        print(f"Loading season {self.current_season} (API)...")
        _, _, _, current_year_players_allfixtures = self.load_current_season_from_api()
        
        # Combine all merged data
        all_merged = pd.concat(list(self.merged_data.values()), axis=0)

        # Add rest-of-season projections
        df_rest_of_current_season = pd.concat([all_merged, current_year_players_allfixtures], axis=0)
        df_rest_of_current_season = df_rest_of_current_season.drop_duplicates(
            subset=['player_name_id', 'season','event', 'fixture_id']
        )
        #Turns Mohamed_Salah_254 into Mohamed Salah. Players with the same name will be combined, but this is a rare occurrence and can be handled later if needed.
        df_rest_of_current_season['player_name_id'] = df_rest_of_current_season['player_name_id'].str.replace('_', ' ').str.replace(r'\d+', '', regex=True).str.strip()
        df_rest_of_current_season['event'] = df_rest_of_current_season['event'].astype(int)

        return df_rest_of_current_season
    
    def get_season_data(self, season: int) -> Optional[pd.DataFrame]:
        """Get raw FPL data for a specific season."""
        return self.seasons_data.get(season)
    
    def get_fixtures_data(self, season: int) -> Optional[pd.DataFrame]:
        """Get fixtures data for a specific season."""
        return self.fixtures_data.get(season)
    
    def get_merged_data(self, season: int) -> Optional[pd.DataFrame]:
        """Get merged data for a specific season."""
        return self.merged_data.get(season)