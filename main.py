# Data Initialization, Preprocessing and Visualization ##


import pandas as pd
from sportradar import NFL
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np


# Initializes NFL API using sportradar
api_key = 'dznnaaf7zzmpj8bgqtnhctpu'
nfl_api = NFL.NFL(api_key)


# Function to convert MM:SS to decimal minutes
def convert_to_decimal_minutes(time_str):
    try:
        # Splits the string into minutes and seconds and converts them to integers
        minutes, seconds = map(int, time_str.split(':'))

        # Calculate total minutes in decimal form
        total_minutes = minutes + seconds / 60.0

        return total_minutes

    except ValueError:
        # Handles strings that cannot be converted
        print(f"Warning: Could not convert {time_str} to decimal minutes")

        return None  # or some default value


# Function to extract all game statistics from specified year and week
def fetch_game_stats(year, week):
    # Specify season details
    nfl_season = 'REG'  # Assuming you're interested in the regular season

    # Get the schedule for the specified week
    response = nfl_api.get_weekly_schedule(year, nfl_season, week)

    if response.status_code == 200:
        schedule = response.json()

    else:
        print(f"Failed to retrieve data: {response.text}")

        return None

    # Extract game IDs
    try:
        game_ids = [game['id'] for game in schedule['week']['games']]

    except KeyError:
        print("Unexpected structure received. 'games' key not found.")

        return None

    # Fetch game statistics for each game
    all_game_stats = []

    for game_id in game_ids:
        stats_response = nfl_api.get_game_statistics(game_id)

        if stats_response.status_code == 200:
            stats = stats_response.json()
            all_game_stats.append(stats)

        else:
            print(f"Failed to retrieve stats for game {game_id}: {stats_response.text}")

    # Convert the list of game statistics to a DataFrame
    games_stats_df = pd.DataFrame(all_game_stats)
    games_stats_df['week'] = week  # Adds a week column

    return games_stats_df


# Function that creates CSV with key game statistics for specified year and weeks up to end week
def save_weekly_data(year, end_week):
    for week in range(1, end_week + 1):
        weekly_stats = fetch_game_stats(year, week)

        if weekly_stats is not None:
            # Prepare an empty list to hold the extracted data
            new_data = []

            # Iterate over the rows of the DataFrame
            for index, row in weekly_stats.iterrows():
                # Extract points and used timeouts for both teams
                home_points = row['summary']['home'].get('points', 0)
                away_points = row['summary']['away'].get('points', 0)
                home_timeouts = row['summary']['home'].get('used_timeouts', 0)
                away_timeouts = row['summary']['away'].get('used_timeouts', 0)

                # Process both 'home' and 'away' data
                for team_type in ['home', 'away']:
                    # Extract the 'name' and 'summary' data for the team
                    team_data = row['statistics'][team_type]
                    name = team_data['name']
                    summary = team_data['summary']
                    rushing = team_data['rushing']['totals']

                    # Converts 'possession_time' to decimal format
                    if 'possession_time' in summary:
                        summary['possession_time'] = convert_to_decimal_minutes(summary['possession_time'])

                    # Extracts data from 'summary' column
                    game_summary = row['summary'][team_type]
                    points = game_summary.get('points', 0)

                    # Uses previously extracted used_timeouts for home and away teams
                    used_timeouts = home_timeouts if team_type == 'home' else away_timeouts

                    # Determines if team won
                    team_won = 1 if (team_type == 'home' and home_points > away_points) or (team_type == 'away' and away_points > home_points) else 0

                    # Creates a new record with the data and add it to our list
                    new_record = {
                        'team': name,
                        'type': team_type,  # adding home or away to game
                        'week': row['week'],  # adding week number to record
                        'won': team_won,  # adding win status to the record
                        'points': points,  # adding points to the record
                        'used_timeouts': used_timeouts  # adding used timeouts to the record
                    }
                    new_record.update(summary)  # add the summary data to the new record
                    new_record.update(rushing)  # add the rushing totals to the new record

                    new_data.append(new_record)

            # Create a new DataFrame from the list of new data
            new_df = pd.DataFrame(new_data)

            # Remove 'longest_touchdown' column if it exists
            new_df = new_df.drop(columns=['longest_touchdown'], errors='ignore')

            # Convert 'type' column to binary: 1 if 'home', 0 if 'away'
            new_df['type'] = new_df['type'].apply(lambda x: 1 if x == 'home' else 0)

            # Save the new structured data
            new_df.to_csv(f'Output_DFs/week_{week}_stats.csv', index=False)

        else:
            print(f"Failed to retrieve data for week {week}")


# Function that aggregates the mean of the statistics up to a specified week
def agg_weekly_data(all_data_df, up_to_week):
    # Filter the data for the weeks we're interested in
    filtered_data = all_data_df[all_data_df['week'] <= up_to_week]

    # Before aggregating, set 'team' as the index. This way, it's not included in the aggregation computations.
    filtered_data.set_index('team', inplace=True)

    # Define the aggregation functions. 'first' keeps the first occurrence within each group.
    agg_functions = {col: 'mean' for col in filtered_data.columns if col not in ['type', 'week']}
    agg_functions['week'] = 'first'

    # Performs aggregation with specified functions.
    agg_stats = filtered_data.groupby('team').agg(agg_functions)

    # Reset the index to turn 'team' back into a column, if preferred for further processing or saving to CSV.
    agg_stats.reset_index(inplace=True)

    return agg_stats


# Usage
year = 2023
end_week = 3  # for example, fetch and save data for weeks 1 through 3
save_weekly_data(year, end_week)

# For aggregation, you'll first need to concatenate your weekly data into one DataFrame.
# This example assumes you'll have saved your weekly data CSVs with the naming convention 'week_{week}_stats.csv'.

all_weeks_data = []
for week in range(1, end_week + 1):
    week_data = pd.read_csv(f'Output_DFs/week_{week}_stats.csv')
    all_weeks_data.append(week_data)

all_data_df = pd.concat(all_weeks_data, ignore_index=True)

aggregated_data = agg_weekly_data(all_data_df, end_week)
aggregated_data.to_csv('Output_DFs/aggregated.csv', index=False)


# Creates 2x3 png including various visualizations of statistics up to a specified week
def generate_nfl_visualizations(week):
    all_weeks_data = []

    # Checks if directory exists
    if not os.path.exists('Output_DFs'):
        print("Directory 'Output_DFs' does not exist. Please make sure the data files are available.")

        return

    # Loads data from CSV files
    for i in range(1, week + 1):
        file_path = f'Output_DFs/week_{i}_stats.csv'

        if os.path.exists(file_path):
            week_data = pd.read_csv(file_path)
            all_weeks_data.append(week_data)

        else:
            print(f"File {file_path} does not exist. Skipping this week.")

            continue

    # Concatenate all weekly data
    all_data = pd.concat(all_weeks_data, ignore_index=True)

    # Generate a palette with a distinct color for each team
    number_of_teams = all_data['team'].nunique()
    distinct_colors = generate_distinct_colors(number_of_teams)

    # Create a dictionary that maps each team to a color
    teams = all_data['team'].unique()
    color_map = dict(zip(teams, distinct_colors))

    # Set up the matplotlib figure
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    sns.despine(left=True)

    # Increases font size of titles, labels, and ticks
    plt.rcParams.update({'font.size': 14})

    # Scatter Plot: Turnovers vs Total Yards
    sns.scatterplot(ax=axes[0, 0], data=all_data, x='turnovers', y='total_yards', hue='team', style='type', palette=color_map, legend=False)
    axes[0, 0].set_title('Turnovers vs Total Yards')
    axes[0, 0].set_xlabel('Turnovers')
    axes[0, 0].set_ylabel('Total Yards')

    # Scatter Plot: Penalties vs Total Yards
    sns.scatterplot(ax=axes[0, 1], data=all_data, x='penalties', y='total_yards', hue='team', style='type', palette=color_map, legend=False)
    axes[0, 1].set_title('Penalties vs Total Yards')
    axes[0, 1].set_xlabel('Penalties')
    axes[0, 1].set_ylabel('Total Yards')

    # Bar Plot: Average Possession Time by Team
    sns.barplot(ax=axes[1, 0], data=all_data, x='team', y='possession_time', palette=color_map, ci=None)
    axes[1, 0].set_title('Average Possession Time by Team')
    axes[1, 0].set_xlabel('Team')
    axes[1, 0].set_ylabel('Average Possession Time (in decimal minutes)')

    # Box Plot: Touchdowns per Game by Team
    sns.boxplot(ax=axes[1, 1], data=all_data, x='team', y='touchdowns', palette=color_map)
    axes[1, 1].set_title('Touchdowns per Game by Team')
    axes[1, 1].set_xlabel('Team')
    axes[1, 1].set_ylabel('Touchdowns')

    # Line Plot: Redzone Attempts Over Weeks
    max_week = all_data['week'].max()
    sns.lineplot(ax=axes[2, 0], data=all_data, x='week', y='redzone_attempts', hue='team', palette=color_map, legend=False)
    axes[2, 0].set_title('Redzone Attempts Over Weeks')
    axes[2, 0].set_xlabel('Week')
    axes[2, 0].set_ylabel('Redzone Attempts')
    axes[2, 0].set_xticks(np.arange(1, max_week + 1, step=1))  # Set x-ticks to show each week

    # Bar Plot: Average Rush Plays by Team
    sns.barplot(ax=axes[2, 1], data=all_data, x='team', y='rush_plays', palette=color_map, ci=None)
    axes[2, 1].set_title('Average Rush Plays by Team')
    axes[2, 1].set_xlabel('Team')
    axes[2, 1].set_ylabel('Average Rush Plays')

    # For x-axis labels on bar plots with team names, rotate for better readability
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, horizontalalignment='right')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, horizontalalignment='right')
    axes[2, 1].set_xticklabels(axes[2, 1].get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.tight_layout()
    plt.savefig(f'Output_DFs/nfl_visualizations_up_to_week_{week}.png')  # Save the figure as a single image file
    plt.close()  # Close the plot to save resources

    print(f"Visualizations for Week 1 to {week} have been saved to 'Output_DFs/nfl_visualizations_up_to_week_{week}.png'.")

def generate_distinct_colors(number_of_colors):
    # Chooses a range of hue values in the HSV space
    hues = np.linspace(0, 1, number_of_colors+1)[:-1]  # +1 and [:-1] to avoid the upper limit (same as lower limig)

    # Create a list of colors in the RGB space
    colors = [mcolors.hsv_to_rgb((hue, 1, 1)) for hue in hues]

    return colors


# Usage
generate_nfl_visualizations(3)  # Replace 3 with the desired week number







