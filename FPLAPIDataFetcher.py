import json
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List

class FPLAPIDataFetcher:
    """
    A class to fetch and process Fantasy Premier League data from the official API.
    
    Attributes:
        season (int): The season year (e.g., 20252026)
        base_url (str): Base URL for the FPL API
        headers (Dict[str, str]): HTTP headers for API requests
        player_raw (pd.DataFrame): Raw player data
        player (pd.DataFrame): Filtered player data
        season_data (pd.DataFrame): Historical gameweek data for all players
    """
    
    BASE_URL = "https://fantasy.premierleague.com/api"
    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
    }
    
    def __init__(self, season: int = 20252026, timeout: int = 5):
        """
        Initialize the FPL API Data Fetcher.
        
        Args:
            season (int): The season year in format YYYYYYYY
            timeout (int): Request timeout in seconds
        """
        self.season = season
        self.timeout = timeout
        self.headers = self.DEFAULT_HEADERS.copy()
        self.player_raw: Optional[pd.DataFrame] = None
        self.player: Optional[pd.DataFrame] = None
        self.season_data: Optional[pd.DataFrame] = None
        
    def fetch_bootstrap_data(self) -> Dict:
        """
        Fetch the bootstrap-static data from FPL API.
        
        Returns:
            Dict: JSON response containing all bootstrap data
            
        Raises:
            requests.RequestException: If the API request fails
        """
        url = f"{self.BASE_URL}/bootstrap-static/"
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        return json.loads(response.text)
    
    def load_player_data(self) -> pd.DataFrame:
        """
        Load and process player data from the bootstrap endpoint.
        
        Returns:
            pd.DataFrame: Processed player dataframe with selected columns
        """
        data = self.fetch_bootstrap_data()
        
        # Create raw player dataframe
        cols = list(data['elements'][0].keys())
        self.player_raw = pd.DataFrame(data['elements'], columns=cols)
        
        # Filter to relevant columns
        self.player = self.player_raw[['web_name', 'id', 'team', 'element_type', 
                                       'first_name', 'second_name']]
        
        return self.player
    
    def fetch_player_history(self, element_id: int) -> List[Dict]:
        """
        Fetch historical gameweek data for a specific player.
        
        Args:
            element_id (int): The player's element ID
            
        Returns:
            List[Dict]: List of gameweek history records
            
        Raises:
            requests.RequestException: If the API request fails
        """
        url = f"{self.BASE_URL}/element-summary/{element_id}/"
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        element_data = json.loads(response.text)
        return element_data['history']
    
    def load_season_data(self, verbose: bool = True) -> pd.DataFrame:
        """
        Load historical gameweek data for all players in the season.
        
        Args:
            verbose (bool): Whether to print progress messages
            
        Returns:
            pd.DataFrame: Season data merged with player information
        """
        if self.player is None:
            self.load_player_data()
        
        season_data_list = []
        player_ids = set(self.player['id'])
        max_player = len(player_ids)
        
        for element_id in list(player_ids): #[:5]: # Limit to first 5 players for testing
            try:
                history = self.fetch_player_history(element_id)
                if history:
                    season_data_list.extend(history)
                
                if verbose:
                    print(f"Loaded data for element_id: {element_id}/{max_player}")
                    
            except requests.RequestException as e:
                print(f"Error loading data for element_id {element_id}: {e}")
                continue
        
        # Create dataframe: Check if we have data to avoid KeyError during merge before season has started
        if not season_data_list:
            # Initialize with all expected FPL API history columns
            expected_columns = [
                'element', 'fixture', 'opponent_team', 'total_points', 'was_home', 
                'kickoff_time', 'team_h_score', 'team_a_score', 'round', 'minutes', 
                'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 
                'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 
                'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 
                'threat', 'ict_index', 'starts', 'expected_goals', 'expected_assists', 
                'expected_goal_involvements', 'expected_goals_conceded', 'value', 
                'transfers_balance', 'selected', 'transfers_in', 'transfers_out'
            ]
            self.season_data = pd.DataFrame(columns=expected_columns)
        else:
            self.season_data = pd.DataFrame.from_dict(season_data_list)
            
        self.season_data = self.season_data.merge(
            self.player, 
            left_on='element', 
            right_on='id', 
            how='left', 
            suffixes=('', '_player')
        )
        
        return self.season_data
       
    def get_player_data(self) -> Optional[pd.DataFrame]:
        """Get the loaded player data."""
        return self.player
    
    def get_season_data(self) -> Optional[pd.DataFrame]:
        """Get the loaded season data."""
        return self.season_data
    
    def get_player_raw(self) -> Optional[pd.DataFrame]:
        """Get the raw player data with all columns."""
        return self.player_raw