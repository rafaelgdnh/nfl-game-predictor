# Logistic Regression Model for Predicting NFL Game Outcomes ##


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os


# Define the mappings of abbreviations to full team names
team_mapping = {
    'KC': 'Chiefs', 'JAX': 'Jaguars', 'CAR': 'Panthers', 'BAL': 'Ravens', 'BUF': 'Bills',
    'MIN': 'Vikings', 'DET': 'Lions', 'ATL': 'Falcons', 'NE': 'Patriots', 'WAS': 'Commanders',
    'CIN': 'Bengals', 'NO': 'Saints', 'SF': '49ers', 'LAR': 'Rams', 'NYG': 'Giants',
    'DEN': 'Broncos', 'CLE': 'Browns', 'IND': 'Colts', 'TEN': 'Titans', 'NYJ': 'Jets',
    'TB': 'Buccaneers', 'MIA': 'Dolphins', 'PIT': 'Steelers', 'PHI': 'Eagles', 'GB': 'Packers',
    'CHI': 'Bears', 'DAL': 'Cowboys', 'ARI': 'Cardinals', 'LAC': 'Chargers', 'HOU': 'Texans',
    'SEA': 'Seahawks', 'LV': 'Raiders'
}

# Load schedule CSV (2022 schedule also included in project directory)
file_path = '2023_nfl_schedule.csv'
df = pd.read_csv(file_path, header=None)

# Sets first row as the header
new_header = df.iloc[0]  # Grab the first row for the header
df = df[1:]  # Take the data less the header row
df.columns = new_header  # Set the header row as the df header


# Function to replace team abbreviations with shortened names
def replace_team_names(cell):
    if isinstance(cell, str):
        # Remove '@' character
        cell = cell.replace('@', '')

        # Replace team abbreviation with full name
        for abbreviation, full_name in team_mapping.items():
            cell = cell.replace(abbreviation, full_name)

    return cell


# Apply the function to each cell in DataFrame
df = df.applymap(replace_team_names)

# Save the cleaned data back to a CSV if needed
df.to_csv('Output_DFs/2023_nfl_schedule_cleaned.csv', index=True)  # replace with desired clean file path


# Function to get opponents of a specific team
def get_opponents(team, schedule_df):
    # Check if the team is in the schedule
    if team in schedule_df['Team'].values:
        # Get the row for the specified team
        team_schedule = schedule_df[schedule_df['Team'] == team]

        # Drops 'Team' column and any other non-opponent columns, if necessary
        opponents = team_schedule.drop(['Team'], axis=1)

        # Convert the row of opponents into a list
        opponents_list = opponents.values.flatten().tolist()

        return opponents_list

    else:
        return f"Could not find schedule for {team}."


# Example usage of the function:
team_name = 'Cardinals'  # replace with the team you're searching for
opponents = get_opponents(team_name, df)
print(f"{team_name} will face: {opponents}")


# Loads aggregated data for model evaluation
aggregated_data = pd.read_csv('Output_DFs/aggregated.csv')

# Reads schedule from the CSV file into a DataFrame
schedule_df = pd.read_csv('Output_DFs/2023_nfl_schedule_cleaned.csv', index_col=0)

# Empty list of DataFrames that will load week_{week}_stats.csv files
weekly_data = []

# Define the features to be used in the model
feature_columns = [
    'points', 'used_timeouts', 'possession_time', 'avg_gain', 'safeties', 'turnovers',
    'play_count', 'rush_plays', 'total_yards', 'fumbles', 'lost_fumbles',
    'penalty_yards', 'return_yards', 'avg_yards', 'attempts', 'touchdowns',
    'tlost_yards', 'yards', 'longest', 'redzone_attempts', 'broken_tackles', 'kneel_downs',
    'scrambles', 'yards_after_contact'
]

# Directory where your CSV files are located
data_directory = 'Output_DFs'  # Change this to the path of your directory if it's not in the current directory

# Maximum week number - change this as necessary
max_week_number = 3  # for example, if you have data up to week 3

# Load the data from the CSV files into the list
for week in range(1, max_week_number + 1):
    file_path = os.path.join(data_directory, f'week_{week}_stats.csv')

    if os.path.exists(file_path):
        week_data = pd.read_csv(file_path)
        weekly_data.append(week_data)

    else:
        print(f"File {file_path} does not exist. Skipping this week.")


# Function that prepares features and outcomes for model
def prepare_features(weekly_data, aggregated_data, current_week):
    # Filter games up to the current week
    weekly_data_current = pd.concat(weekly_data[:current_week], ignore_index=True)

    # Merge the weekly data with the aggregated data to get historical team performance up to the current week
    data = weekly_data_current.merge(aggregated_data, on='team', suffixes=('', '_agg'), how='left')

    # For simplicity, we'll use only the current week's stats and aggregated stats as features
    features = data[feature_columns + [f'{col}_agg' for col in feature_columns]]
    outcomes = data['won']

    return features, outcomes


# Function that conducts training and testing of Logistic Regression model
def train_logistic_regression(features, outcomes):
    # Splits dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2, random_state=0)

    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the logistic regression model
    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    return model, scaler


# Function that calculates win probability based on model and the opponent's features
def predict_win_probability(model, scaler, team_features, opponent_features):
    # Standardize each team's features independently
    team_features_scaled = scaler.transform(team_features.reshape(1, -1))
    opponent_features_scaled = scaler.transform(opponent_features.reshape(1, -1))

    # Predict win probability for both teams
    team_win_probability = model.predict_proba(team_features_scaled)[:, 1]  # Probability of class 1 (win)
    opponent_win_probability = model.predict_proba(opponent_features_scaled)[:, 1]

    # If you want to return the relative probability of the team winning compared to the opponent,
    # you might calculate it as follows. Note: This isn't a true probability, but rather a relative score.
    relative_win_probability = team_win_probability / (team_win_probability + opponent_win_probability)

    return relative_win_probability


# Function that predicts win probability of a team on a given week based on their features and their opponents
def predict_game_outcome(team_name, week, model, scaler, weekly_data, aggregated_data):
    # Get the opponent
    opponents = get_opponents(team_name, schedule_df)

    if week <= len(opponents):
        opponent_name = opponents[week - 1]  # weeks start at 1, list indexing starts at 0

    else:
        raise ValueError(f"No game scheduled for {team_name} in week {week}")

    # Get the team and opponent features
    team_features = prepare_team_features(team_name, week, weekly_data, aggregated_data)
    opponent_features = prepare_team_features(opponent_name, week, weekly_data, aggregated_data)

    # Predict the win probability
    win_probability = predict_win_probability(model, scaler, team_features, opponent_features)

    return win_probability, opponent_name


# Function that prepares the necessary features for the home and away team
def prepare_team_features(team_name, week, weekly_data, aggregated_data):
    # Concatenate all weekly data into a single DataFrame
    all_weekly_data = pd.concat(weekly_data, ignore_index=True)

    # Now, filter for the specific team
    team_weekly_data = all_weekly_data[all_weekly_data['team'] == team_name]

    # Get the data up to the specified week
    team_data_current_week = team_weekly_data[team_weekly_data['week'] <= week]

    if team_data_current_week.empty:
        raise ValueError(f"No data available for {team_name} in week {week}.")

    else:
        team_latest_game = team_data_current_week.iloc[-1]

    # Get the aggregated data for the team up to the specified week
    team_aggregated = aggregated_data[(aggregated_data['team'] == team_name) & (aggregated_data['week'] <= week)]

    # If there's no data for the team, doesn't create features
    if team_aggregated.empty or team_latest_game.empty:
        print(f"No data available for team {team_name} up to week {week}.")
        return None

    # Select only the feature columns from both current game and aggregated data
    team_features_current = team_latest_game[feature_columns].values
    team_features_aggregated = team_aggregated[feature_columns].mean().values  # Using mean stats up to the current week

    # Combine current game and aggregated stats to form the feature set
    team_features = np.concatenate([team_features_current, team_features_aggregated])

    return team_features


# Usage
current_week = 3  # specify the current week
features, outcomes = prepare_features(weekly_data, aggregated_data, current_week)
model, scaler = train_logistic_regression(features, outcomes)

# Predict the outcome of a game for a specific team in a specific week
team_name = "Bills"  # specify the team name
week = 4  # specify the week for which you want to predict the outcome
win_probability, opponent = predict_game_outcome(team_name, week, model, scaler, weekly_data, aggregated_data)
print(f"Predicted Win Probability for {team_name} against {opponent} in Week {week}: {win_probability[0]}")

