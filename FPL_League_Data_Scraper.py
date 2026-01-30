import requests
import pandas as pd
import json

class FPL_League_Data_Scraper:
    def __init__(self, league_code: str, season: int):
        """Initialize FPL Scraper"""
        self.league_code = league_code
        self.season = season

    #Request first page of league standings and return all league data 
    def get_json_from_url(self):
        league_url=f"https://fantasy.premierleague.com/api/leagues-classic/{self.league_code}/standings/?page_new_entries=1&page_standings=1&phase=1"
        get=requests.get(league_url)
        self.league_data=json.loads(get.text)
        self.league_name = self.league_data['league'].get('name')
        print("Data fetched, league name: ", self.league_name)
        return self.league_data, self.league_name

    #Use league data to return a dict of entry ids to names (handles duplicates)
    def make_ids(self):
        self.member_ids={}
        self.id_to_name = {}  # New: maps entry ID to name
        for member in self.league_data['standings']['results']:
            name=member.get('player_name')
            mid=member.get('entry')
            self.id_to_name[mid] = name
            self.member_ids[name]=mid  # Keep for backward compatibility (will have last occurrence)
        return self.member_ids

    #Use first value in member_ids to find the max gameweek that data is available for
    #Get data1
    def get_first_member_data(self):
        self.first_xid=list(self.id_to_name.keys())[0]
        gw=1
        get_first=requests.get("https://fantasy.premierleague.com/api/entry/"+str(self.first_xid)+"/event/"+str(gw)+"/picks/")
        data_first=json.loads(get_first.text)
        return data_first
    
    def find_max_gw(self): #xid=member entry id
        gw=1
        while gw < 100:
            try:
                get_mgw=requests.get("https://fantasy.premierleague.com/api/entry/"+str(self.first_xid)+"/event/"+str(gw)+"/picks/")
                data_mgw=json.loads(get_mgw.text)
                list(data_mgw['entry_history'].keys())
                gw+=1
            except KeyError as err:
                print("Max gameweek data available for is: ",gw)
                return (gw)

    #Member picks
    def dataframe_from_ids(self):
        data_for_columns=self.get_first_member_data()
        mgw = self.find_max_gw()
        cols=list(data_for_columns["picks"][0].keys())
        p=[]

        for xid in self.id_to_name.keys():  # Changed: iterate over all entry IDs
            for gw in range(1,mgw):    
                get_gw=requests.get("https://fantasy.premierleague.com/api/entry/"+str(xid)+"/event/"+str(gw)+"/picks/")
                data_gw=json.loads(get_gw.text)
                try:
                    for x in range(0,15):
                        #picks
                        row=data_gw["picks"][x]

                        name=self.id_to_name[xid]  # Changed: direct lookup
                        selected_row = [name,xid,gw,x+1]

                        for i in cols:
                            selected_row.append(row.get(i))

                        p.append(selected_row)

                        print(xid,name,"GW "+str(gw),"Pick "+str(x+1))
                except KeyError as e:
                    name = self.id_to_name[xid]  # Changed: direct lookup
                    print("GW Could not be found!",xid,name,"GW "+str(gw))
                    print(e)
                    continue

        print("Finished")
        pcols=['member','memberid','gw','pick']
        pcols.extend(cols)
        picksdf = pd.DataFrame(p, columns=pcols)
        picksdf['season']=self.season
        return picksdf

    def gw_summary_from_ids(self):
        """Fetch gameweek summary data for all members and merge with chips"""
        summary_data = []
        
        for xid, name in self.id_to_name.items():  # Changed: iterate over all entry IDs
            try:
                # Fetch history data for this member
                history_url = f"https://fantasy.premierleague.com/api/entry/{xid}/history/"
                response = requests.get(history_url)
                history_data = json.loads(response.text)
                
                # Create a dict mapping event to chip name for quick lookup
                chip_dict = {}
                for chip in history_data.get('chips', []):
                    chip_dict[chip['event']] = chip['name']
                
                # Extract current season gameweek data
                for gw_data in history_data['current']:
                    row = gw_data.copy()
                    row['member'] = name
                    row['memberid'] = xid
                    row['chip'] = chip_dict.get(gw_data['event'], None)
                    summary_data.append(row)
                
                print(f"Fetched history for {name} (ID: {xid})")
                
            except Exception as e:
                print(f"Error fetching history for {name} (ID: {xid}): {e}")
                continue
        
        # Create DataFrame
        summary = pd.DataFrame(summary_data)
        summary['season'] = self.season
        
        return summary
    
    def scrape_picks(self):
        self.get_json_from_url()
        self.make_ids()
        membergw_summary = self.gw_summary_from_ids()
        picksdf = self.dataframe_from_ids()
        return picksdf, membergw_summary