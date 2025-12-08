import pandas as pd
import requests
import json
from FPLDataLoader import FPLDataLoader
from FPLAPIDataFetcher import FPLAPIDataFetcher
from DatabaseCreator import DatabaseCreator
from Load_Merge_FPL_Data import FPLDataManager 
import CJDH_local_settings

#Run Database Creator


# Usage example
if __name__ == "__main__":
    manager = FPLDataManager()
    df_all_seasons = manager.load_all_seasons()
    print(f"\nTotal records: {len(df_all_seasons)}")

    db_creator = DatabaseCreator(db_settings=CJDH_local_settings.local_settings['FPL_Points_Predictor'])
    fpl_engine = db_creator.get_engine_for("fpl_data_analysis")

    #Add a new staging table & data into the database
    playergw = df_all_seasons[['player_name_id','element','value','season','event','fixture','total_points',
                            'minutes','goals_scored','assists','team_elo','opp_team_elo','position','bonus','bps',
                            'clean_sheets','goals_conceded','was_home','expected_assists','expected_goal_involvements',
                            'expected_goals','expected_goals_conceded','starts',
                            'cbi','defensive_contribution','recoveries','tackles',
                            'saves','team_name','opp_team_name']]

    table_name = "playergw"
    db_creator.create_staging_table_then_insert_data(table_name, data=playergw)
    playergwdf = db_creator.table_to_df(table_name=table_name)
    print("Final playergw dataframe loaded into psql!")
    print(playergwdf.info())
