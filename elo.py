# CURRENTLY UNUSED ##
# FiveThirtyEight NFL Elo Merger ##


import pandas as pd
import os


def get_elo():
    elo_df = pd.read_csv('nfl_elo_latest.csv')
    elo_df = elo_df.drop(columns = ['season','neutral' ,'playoff', 'elo_prob1', 'elo_prob2', 'elo1_post', 'elo2_post',
                                    'qbelo1_pre', 'qbelo2_pre', 'qb1', 'qb2', 'qb1_adj', 'qb2_adj', 'qbelo_prob1', 'qbelo_prob2',
                                    'qb1_game_value', 'qb2_game_value', 'qb1_value_post', 'qb2_value_post',
                                    'qbelo1_post', 'qbelo2_post', 'score1', 'score2'])
    elo_df.date = pd.to_datetime(elo_df.date)
    elo_df = elo_df[elo_df.date < '01-09-2023']

    elo_df['team1'] = elo_df['team1'].replace(['KC', 'JAX', 'CAR', 'BAL', 'BUF', 'MIN', 'DET', 'ATL', 'NE', 'WSH',
                                               'CIN', 'NO', 'SF', 'LAR', 'NYG', 'DEN', 'CLE', 'IND', 'TEN', 'NYJ',
                                               'TB', 'MIA', 'PIT', 'PHI', 'GB', 'CHI', 'DAL', 'ARI', 'LAC', 'HOU',
                                               'SEA', 'OAK'],
                                              ["Chiefs", "Jaguars", "Panthers", "Ravens", "Bills", "Vikings", "Lions",
                                               "Falcons", "Patriots", "Football Team", "Bengals", "Saints", "49ers",
                                               "Rams", "Giants", "Broncos", "Browns", "Colts", "Titans", "Jets",
                                               "Buccaneers", "Dolphins", "Steelers", "Eagles", "Packers", "Bears",
                                               "Cowboys", "Cardinals", "Chargers", "Texans", "Seahawks", "Raiders"])
    elo_df['team2'] = elo_df['team2'].replace(['KC', 'JAX', 'CAR', 'BAL', 'BUF', 'MIN', 'DET', 'ATL', 'NE', 'WSH',
                                               'CIN', 'NO', 'SF', 'LAR', 'NYG', 'DEN', 'CLE', 'IND', 'TEN', 'NYJ',
                                               'TB', 'MIA', 'PIT', 'PHI', 'GB', 'CHI', 'DAL', 'ARI', 'LAC', 'HOU',
                                               'SEA', 'OAK'],
                                              ["Chiefs", "Jaguars", "Panthers", "Ravens", "Bills", "Vikings", "Lions",
                                               "Falcons", "Patriots", "Football Team", "Bengals", "Saints", "49ers",
                                               "Rams", "Giants", "Broncos", "Browns", "Colts", "Titans", "Jets",
                                               "Buccaneers", "Dolphins", "Steelers", "Eagles", "Packers", "Bears",
                                               "Cowboys", "Cardinals", "Chargers", "Texans", "Seahawks", "Raiders"])
    return elo_df


def merge_rankings(agg_games_df,elo_df):

    agg_games_df = pd.merge(agg_games_df, elo_df, how = 'inner', left_on = ['home_abbr', 'away_abbr'], right_on = ['team1', 'team2']).drop(columns = ['date','team1', 'team2'])
    agg_games_df['elo_dif'] = agg_games_df['elo2_pre'] - agg_games_df['elo1_pre']
    agg_games_df['qb_dif'] = agg_games_df['qb2_value_pre'] - agg_games_df['qb1_value_pre']
    agg_games_df = agg_games_df.drop(columns = ['elo1_pre', 'elo2_pre', 'qb1_value_pre', 'qb2_value_pre'])

    return agg_games_df


def merge_elo_with_weekly_stats(elo_df, week, output_dir='Output_DFs'):
    # Construct the file path
    file_path = f'{output_dir}/week_{week}_stats.csv'

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return None

    # If the file exists, proceed to load the weekly game stats
    try:
        weekly_stats_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Failed to read the file {file_path}.")
        return None

    # Rest of the processing remains the same...
    weekly_stats_df['week'] = weekly_stats_df['week'].astype(int)
    elo_df['week'] = elo_df['week'].astype(int)

    elo_df_long = pd.melt(elo_df, id_vars=['date', 'season', 'week', 'elo1_pre', 'elo2_pre'],
                          value_vars=['team1', 'team2'], value_name='team')

    elo_df_long['elo_pre'] = elo_df_long.apply(
        lambda row: row['elo1_pre'] if row['variable'] == 'team1' else row['elo2_pre'], axis=1
    )
    elo_df_long = elo_df_long.drop(['elo1_pre', 'elo2_pre', 'variable'], axis=1)

    merged_df = pd.merge(weekly_stats_df, elo_df_long, on=['team', 'week'], how='left')

    merged_df.to_csv(f'{output_dir}/week_{week}_stats_with_elo.csv', index=False)

    return merged_df


