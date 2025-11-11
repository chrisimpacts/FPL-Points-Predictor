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

    #Use league data to return a dict of names & entry ids
    def make_ids(self):
        self.member_ids={}
        for member in self.league_data['standings']['results']:
            name=member.get('player_name')
            mid=member.get('entry')
            self.member_ids[name]=mid
        return self.member_ids

    #Use first value in member_ids to find the max gameweek that data is available for
    #Get data1
    def get_first_member_data(self):
        self.first_xid=list(self.member_ids.values())[0]
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

        for xid in self.member_ids.values():
            for gw in range(1,mgw):    
                get_gw=requests.get("https://fantasy.premierleague.com/api/entry/"+str(xid)+"/event/"+str(gw)+"/picks/")
                data_gw=json.loads(get_gw.text)
                try:
                    for x in range(0,15):
                        #picks
                        # print("Requesting...... ",xid,"GW "+str(gw),"Pick "+str(x+1))
                        row=data_gw["picks"][x]

                        name=list(self.member_ids.keys())[list(self.member_ids.values()).index(xid)]
                        selected_row = [name,xid,gw,x+1]

                        for i in cols:
                            selected_row.append(row.get(i))

                        p.append(selected_row)

                        print(xid,name,"GW "+str(gw),"Pick "+str(x+1))
                except KeyError as e:
                    name = list(self.member_ids.keys())[list(self.member_ids.values()).index(xid)]
                    print("GW Could not be found!",xid,name,"GW "+str(gw))
                    print(e)
                    continue

        print("Finished")
        pcols=['member','memberid','gw','pick']
        pcols.extend(cols)
        picksdf = pd.DataFrame(p, columns=pcols)
        picksdf['season']=self.season
        return picksdf
    
    def scrape_picks(self):
        self.get_json_from_url()
        self.make_ids()
        picksdf = self.dataframe_from_ids()
        return picksdf
