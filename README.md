# FPL-Data-Analysis

Code to scrape, merge and analyse Fantasy Premier League (FPL) data

## Description

Fantasy Premier League (FPL) is a game played by millions of football fans globally. Players of the game choose real-life premier league goalkeepers, defenders, midfielders and forwards and are awarded points according to the performances, for example a goal scored by a defender is 6 points and an assist is worth 3 points. Each player has a cost which can change over time.

There are 4 steps to this process:
1. Scrape and join data from Vaastav Github & the FPL website
2. Transform data and add new features: running averages, metrics per 90 etc.
3. Use Random Forest regression to predict components of FPL points. Combine these to create a xPoints estimate for each player and gameweek.
4. Analyse my teams in TRDL Div1 & DivX to inform decision making.

##### Set up
1. Clone the repository to your lcoal machine
2. Install the reuired python packages with pip
3. Set up your PostgreSQL database and update the connection details in the script (replace placeholders in the CJDH_local_settings.py file).
4. Run the scripts/notebooks in order: Load_FPL_data_vaastav, FPL-League-Scrape, TransformData, FPL-Components-Prediction, TRDL-Team-Analysis (or DIvX-Team-Analysis for a more generalised solution)


#### Load_FPL_data_vaastav:
This script performs Extract, Transform, and Load (ETL) operations on Fantasy Premier League (FPL) data.

#### FPL-League-Scrape:
This script gets data from teams in my friends' FPL league

#### TransformData:
This script cretes running averages and add per90 metrics

#### FPL-Components-Prediction:
This script predicts future gameweek scores using past data

#### DIVX-Team-Analysis:
This script compares players available for transfer with players in my current FPL team in terms of xPoints over the next  gameweeks.


## Contributing
Contributions to this project are welcome. Feel free to submit pull requests for improvements, bug fixes, or new features.

## Acknowledgment

Links
* https://medium.com/@frenzelts/fantasy-premier-league-api-endpoints-a-detailed-guide-acbd5598eb19
