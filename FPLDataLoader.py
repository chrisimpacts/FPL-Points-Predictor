import pandas as pd
import numpy as np
import re

class FPLDataLoader:
    # Class-level ELO ratings dictionary shared across all seasons
    _global_elo_ratings = {}

    teams_by_season = {
        20162017: {
        1: "Arsenal",
        2: "Bournemouth",
        3: "Burnley",
        4: "Chelsea",
        5: "Crystal Palace",
        6: "Everton",
        7: "Hull",
        8: "Leicester",
        9: "Liverpool",
        10: "Man City",
        11: "Man Utd",
        12: "Middlesbrough",
        13: "Southampton",
        14: "Stoke",
        15: "Sunderland",
        16: "Swansea",
        17: "Spurs",
        18: "Watford",
        19: "West Brom",
        20: "West Ham",
    },
    20172018: {
        1: "Arsenal",
        2: "Bournemouth",
        3: "Brighton",
        4: "Burnley",
        5: "Chelsea",
        6: "Crystal Palace",
        7: "Everton",
        8: "Huddersfield",
        9: "Leicester",
        10: "Liverpool",
        11: "Man City",
        12: "Man Utd",
        13: "Newcastle",
        14: "Southampton",
        15: "Stoke",
        16: "Swansea",
        17: "Spurs",
        18: "Watford",
        19: "West Brom",
        20: "West Ham",
    },
    20182019: {
        1: "Arsenal",
        2: "Bournemouth",
        3: "Brighton",
        4: "Burnley",
        5: "Cardiff",
        6: "Chelsea",
        7: "Crystal Palace",
        8: "Everton",
        9: "Fulham",
        10: "Huddersfield",
        11: "Leicester",
        12: "Liverpool",
        13: "Man City",
        14: "Man Utd",
        15: "Newcastle",
        16: "Southampton",
        17: "Spurs",
        18: "Watford",
        19: "West Ham",
        20: "Wolves",
    },
    20192020: {
        1: "Arsenal",
        2: "Aston Villa",
        3: "Bournemouth",
        4: "Brighton",
        5: "Burnley",
        6: "Chelsea",
        7: "Crystal Palace",
        8: "Everton",
        9: "Leicester",
        10: "Liverpool",
        11: "Man City",
        12: "Man Utd",
        13: "Newcastle",
        14: "Norwich",
        15: "Sheffield Utd",
        16: "Southampton",
        17: "Spurs",
        18: "Watford",
        19: "West Ham",
        20: "Wolves",
    },
    20202021: {
        1: "Arsenal",
        2: "Aston Villa",
        3: "Brighton",
        4: "Burnley",
        5: "Chelsea",
        6: "Crystal Palace",
        7: "Everton",
        8: "Fulham",
        9: "Leicester",
        10: "Leeds",
        11: "Liverpool",
        12: "Man City",
        13: "Man Utd",
        14: "Newcastle",
        15: "Sheffield Utd",
        16: "Southampton",
        17: "Spurs",
        18: "West Brom",
        19: "West Ham",
        20: "Wolves",
    },
    20212022: {
        1: "Arsenal",
        2: "Aston Villa",
        3: "Brentford",
        4: "Brighton",
        5: "Burnley",
        6: "Chelsea",
        7: "Crystal Palace",
        8: "Everton",
        9: "Leicester",
        10: "Leeds",
        11: "Liverpool",
        12: "Man City",
        13: "Man Utd",
        14: "Newcastle",
        15: "Norwich",
        16: "Southampton",
        17: "Spurs",
        18: "Watford",
        19: "West Ham",
        20: "Wolves",
    },
    20222023: {
        1: "Arsenal",
        2: "Aston Villa",
        3: "Bournemouth",
        4: "Brentford",
        5: "Brighton",
        6: "Chelsea",
        7: "Crystal Palace",
        8: "Everton",
        9: "Fulham",
        10: "Leicester",
        11: "Leeds",
        12: "Liverpool",
        13: "Man City",
        14: "Man Utd",
        15: "Newcastle",
        16: "Nott'm Forest",
        17: "Southampton",
        18: "Spurs",
        19: "West Ham",
        20: "Wolves",
    },
    20232024: {
        1: "Arsenal",
        2: "Aston Villa",
        3: "Bournemouth",
        4: "Brentford",
        5: "Brighton",
        6: "Burnley",
        7: "Chelsea",
        8: "Crystal Palace",
        9: "Everton",
        10: "Fulham",
        11: "Liverpool",
        12: "Luton",
        13: "Man City",
        14: "Man Utd",
        15: "Newcastle",
        16: "Nott'm Forest",
        17: "Sheffield Utd",
        18: "Spurs",
        19: "West Ham",
        20: "Wolves",
    },
    20242025: {
        1: "Arsenal",
        2: "Aston Villa",
        3: "Bournemouth",
        4: "Brentford",
        5: "Brighton",
        6: "Chelsea",
        7: "Crystal Palace",
        8: "Everton",
        9: "Fulham",
        10: "Ipswich",
        11: "Leicester",
        12: "Liverpool",
        13: "Man City",
        14: "Man Utd",
        15: "Newcastle",
        16: "Nott'm Forest",
        17: "Southampton",
        18: "Spurs",
        19: "West Ham",
        20: "Wolves",
    },
    20252026: {
        1: "Arsenal",
        2: "Aston Villa",
        3: "Burnley",
        4: "Bournemouth",
        5: "Brentford",
        6: "Brighton",
        7: "Chelsea",
        8: "Crystal Palace",
        9: "Everton",
        10: "Fulham",
        11: "Leeds",
        12: "Liverpool",
        13: "Man City",
        14: "Man Utd",
        15: "Newcastle",
        16: "Nott'm Forest",
        17: "Sunderland",
        18: "Spurs",
        19: "West Ham",
        20: "Wolves",
    }
}
    rename_gw_data_dict = {
            "name":"player_name_id",
            # "assists":"",
            # "attempted_passes":"",
            # "big_chances_created":"",
            # "big_chances_missed":"",
            # "bonus":"",
            # "bps":"",
            # "clean_sheets":"",
            "clearances_blocks_interceptions":"cbi",
            # "completed_passes":"",
            # "creativity":"",
            # "dribbles":"",
            # "ea_index":"",
            # "element":"",
            # "errors_leading_to_goal":"",
            # "errors_leading_to_goal_attempt":"",
            # "fixture":"",
            # "fouls":"",
            # "goals_conceded":"",
            # "goals_scored":"",
            # "ict_index":"",
            # "id":"",
            # "influence":"",
            # "key_passes":"",
            # "kickoff_time":"",
            # "kickoff_time_formatted":"",
            # "loaned_in":"",
            # "loaned_out":"",
            # "minutes":"",
            # "offside":"",
            # "open_play_crosses":"",
            # "opponent_team":"",
            # "own_goals":"",
            # "penalties_conceded":"",
            # "penalties_missed":"",
            # "penalties_saved":"",
            # "recoveries":"",
            # "red_cards":"",
            # "round":"",
            # "saves":"",
            # "selected":"",
            # "tackled":"",
            # "tackles":"",
            # "target_missed":"",
            # "team_a_score":"",
            # "team_h_score":"",
            # "threat":"",
            # "total_points":"",
            # "transfers_balance":"",
            # "transfers_in":"",
            # "transfers_out":"",
            # "value":"",
            # "was_home":"",
            # "winning_goals":"",
            # "yellow_cards":"",
            "GW":"gw",
        }
    rename_fixture_data_dict = {
            "code":"fixture_code",
            # "deadline_time":"",
            # "deadline_time_formatted":"",
            # "event":"",
            # "event_day":"",
            # "finished":"",
            # "finished_provisional":"",
            "id":"fixture_id",
            "kickoff_time":"fixture_kickoff_time",
            "kickoff_time_formatted":"fixture_kickoff_time_formatted",
            "minutes":"fixture_minutes",
            # "provisional_start_time":"",
            "started":"fixture_started",
            "stats":"fixture_stats",
            # "team_a":"",
            # "team_a_difficulty":"",
            "team_a_score":"fixture_team_a_score",
            # "team_h":"",
            # "team_h_difficulty":"",
            "team_h_score":"fixture_team_h_score",
            "season":"fixture_season",
        }
    
    def __init__(self, season: int, gw_url: str = None, gw_df = None, fixtures_path: str = None, fixtures_df = None, 
                 rename_gw_data_dict: dict = rename_gw_data_dict, rename_fixture_data_dict: dict = rename_fixture_data_dict, 
                 teams_by_season: dict = teams_by_season, reset_elo: bool = False):
        """
        Initialize the FPL data loader

        Args:
            season (int): The season (e.g. 20192020).
            gw_url (str): URL or path to the merged gameweek data CSV.
            gw_df (DataFrame): Alternative to gw_url - pass DataFrame directly.
            fixtures_path (str): Path to the fixtures CSV.
            fixtures_df (DataFrame): Alternative to fixtures_path.
            rename_gw_data_dict (dict): Dict that renames columns for clarity.
            rename_fixture_data_dict (dict): Dict that renames columns for clarity.
            teams_by_season (dict): Mapping of team IDs to names by season.
            reset_elo (bool): If True, reset all ELO ratings to start_rating.
        """
        self.season = season
        self.gw_url = gw_url
        self.gw_df = gw_df
        self.fixtures_path = fixtures_path
        self.fixtures_df = fixtures_df
        self.fpl_data = None
        self.fixtures = None
        self.merged_data = None
        self.teams_by_season = teams_by_season or {}
        self.rename_gw_data_dict = rename_gw_data_dict or {}
        self.rename_fixture_data_dict = rename_fixture_data_dict or {}
        self.K = 20  # ELO K-factor
        self.START_RATING = 1000
        
        if reset_elo:
            FPLDataLoader._global_elo_ratings = {}

    @classmethod
    #fixes the elo ratings to the class, so that the elo ratings persist for following seasons (as long as analysis is done sequentially)
    def get_global_elo_ratings(cls):
        """Get the current global ELO ratings dictionary."""
        return cls._global_elo_ratings.copy()

    def team_name(self, row, teams_by_season):
        if row['was_home']:
            return teams_by_season[row['fixture_season']].get(row['team_h'])
        else:
            return teams_by_season[row['fixture_season']].get(row['team_a'])
    
    def opp_team_name(self, row, teams_by_season):
        if row['was_home']:
            return teams_by_season[row['fixture_season']].get(row['team_a'])
        else:
            return teams_by_season[row['fixture_season']].get(row['team_h'])
    
    def away_team_name(self, row, teams_by_season):
        return teams_by_season[row['fixture_season']].get(row['team_a'])
    
    def home_team_name(self, row, teams_by_season):
        return teams_by_season[row['fixture_season']].get(row['team_h'])
    
    def format_player_name_id(text):
        result = re.sub(r'[^\w\s]', '', text).replace(' ', '_')
        return result

    def load_fpl_data(self):
        """Load the merged gameweek data from CSV or DataFrame and add a season column."""
        if self.gw_df is not None:
            self.fpl_data = self.gw_df.copy()
            self.fpl_data['gw'] = self.fpl_data['round']  
            #recreate player_name_id
            self.fpl_data['player_name_id'] = self.fpl_data.apply(lambda x: f"{x['first_name']} {x['second_name']}", axis=1)     
            #map position 1,2,3,4 to GK, DEF, MID, FWD
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            self.fpl_data['position'] = self.fpl_data['element_type'].map(position_map)
            #get team name
            columns_to_convert = ['expected_goals','expected_assists','expected_goal_involvements','expected_goals_conceded']
            self.fpl_data[columns_to_convert] = self.fpl_data[columns_to_convert].apply(pd.to_numeric, errors='coerce')


        elif self.gw_url:
            self.fpl_data = pd.read_csv(self.gw_url, encoding="latin1", engine="python")
        else:
            raise ValueError("Either gw_url or gw_df must be provided")
        
        self.fpl_data = self.fpl_data.rename(columns=self.rename_gw_data_dict)
        self.fpl_data["season"] = self.season
        return self.fpl_data

    def load_fixtures(self):
        """Load the fixtures CSV."""
        if self.fixtures_path:
            self.fixtures = pd.read_csv(self.fixtures_path, encoding="latin1", engine="python")
        elif self.fixtures_df is not None:
            self.fixtures = self.fixtures_df.copy()
        else:
            raise ValueError("Either fixtures_path or fixtures_df must be provided")

        self.fixtures = self.fixtures.rename(columns=self.rename_fixture_data_dict)
        self.fixtures['fixture_season'] = self.season
        self.fixtures['away_team_name'] = self.fixtures.apply(
            lambda row: self.away_team_name(row, self.teams_by_season), axis=1
        )
        self.fixtures['home_team_name'] = self.fixtures.apply(
            lambda row: self.home_team_name(row, self.teams_by_season), axis=1
        )

    def add_elo_to_fixtures(self):
        """Add ELO ratings to fixtures, carrying over from previous seasons."""
        # Sort by kickoff_time
        self.fixtures = self.fixtures.sort_values("fixture_kickoff_time").reset_index(drop=True)
        
        home_elo_list = []
        away_elo_list = []
        
        for _, row in self.fixtures.iterrows():
            home = row['home_team_name']
            away = row['away_team_name']
            
            # Get current ratings from global dictionary or default
            R_home = FPLDataLoader._global_elo_ratings.get(home, self.START_RATING)
            R_away = FPLDataLoader._global_elo_ratings.get(away, self.START_RATING)
            
            # Save pre-match ratings
            home_elo_list.append(R_home)
            away_elo_list.append(R_away)
            
            # Only update if scores are available
            if not np.isnan(row['fixture_team_h_score']) and not np.isnan(row['fixture_team_a_score']):
                home_score = row['fixture_team_h_score']
                away_score = row['fixture_team_a_score']
                
                # Actual results
                if home_score > away_score:
                    S_home, S_away = 1, 0
                elif home_score < away_score:
                    S_home, S_away = 0, 1
                else:
                    S_home, S_away = 0.5, 0.5
                
                # Expected results
                E_home = 1 / (1 + 10 ** ((R_away - R_home) / 400))
                E_away = 1 - E_home
                
                # Update ratings in global dictionary
                FPLDataLoader._global_elo_ratings[home] = R_home + self.K * (S_home - E_home)
                FPLDataLoader._global_elo_ratings[away] = R_away + self.K * (S_away - E_away)
        
        # Add to dataframe
        self.fixtures['home_team_elo'] = home_elo_list
        self.fixtures['away_team_elo'] = away_elo_list
        
        return self.fixtures

    def merge_data(self):
        """Merge FPL data with fixtures on fixture ID."""
        if self.fpl_data is None:
            self.load_fpl_data()
        if self.fixtures is None:
            self.load_fixtures()
            self.add_elo_to_fixtures()

        self.merged_data = self.fpl_data.merge(
            self.fixtures,
            left_on="fixture",
            right_on="fixture_id",
            suffixes=("", "_fixture"),
            how='left',
            validate='many_to_one'
        )
        
        self.merged_data['team_name'] = self.merged_data.apply(
            lambda row: self.team_name(row, self.teams_by_season), axis=1
        )
        self.merged_data['team_elo'] = self.merged_data.apply(
            lambda row: row["home_team_elo"] if row["was_home"] else row["away_team_elo"], 
            axis=1
        )
        self.merged_data['opp_team_name'] = self.merged_data.apply(
            lambda row: self.opp_team_name(row, self.teams_by_season), axis=1
        )
        self.merged_data['opp_team_elo'] = self.merged_data.apply(
            lambda row: row["away_team_elo"] if row["was_home"] else row["home_team_elo"], 
            axis=1
        )
        
        return self.merged_data