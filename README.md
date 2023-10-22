# NFL Game Analysis and Prediction
 This project provides a comprehensive toolkit for NFL game analysis, including data fetching, preprocessing, visualization, and game outcome prediction using machine learning.

## Features
* Fetching game statistics from the Sportradar API.
* Preprocessing and aggregating game statistics.
* Visualizing key game statistics with various plots.
* Predicting game outcomes using a logistic regression model.

## Prerequisites
 Before you begin, ensure you have met the following requirements:

* You have a Python 3.x installation.
* You have installed the required Python packages: pandas, matplotlib, seaborn, numpy, sklearn, os and sportradar
* You have a Sportradar API key for accessing NFL data.

## Usage
 The project consists of two main scripts: 'main.py' for data initialization, preprocessing, and visualization, and 'model.py' for creating and using a logistic regression model to  predict game outcomes. 'elo.py' is also included, but it's currently not being used in the model. 

### Data Initialization, Preprocessing, and Visualization
 In main.py, we fetch NFL game statistics, convert relevant data to a usable format, save data for each game week, and create visualizations.


```python
year = 2023
end_week = 3
save_weekly_data(year, end_week)
generate_nfl_visualizations(3)
```

### Game Outcome Prediction
In 'model.py', we utilize a logistic regression model to predict the outcomes of NFL games based on aggregated team statistics.


```python
current_week = 3
features, outcomes = prepare_features(weekly_data, aggregated_data, current_week)
model, scaler = train_logistic_regression(features, outcomes)
team_name = "Bills"
week = 4
win_probability, opponent = predict_game_outcome(team_name, week, model, scaler, weekly_data, aggregated_data)
```

## Agknowledgements 
 Here are the sources I utilized to build this project:

 * ActiveState: https://www.activestate.com/blog/how-to-predict-nfl-winners-with-python/
 * Sportsradar: https://developer.sportradar.com/API_Packaging
 * Sportradar API's: https://github.com/johnwmillr/SportradarAPIs
 * FiveThirtyEight: https://projects.fivethirtyeight.com/2022-nfl-predictions/





 

