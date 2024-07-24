import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

def calculate_fantasy_points(row):
    # Ensure fumbles column exists, default to 0 if missing
    fumbles = row.get('fumbles', 0)

    passing_points = (row['passing_yards'] * 0.04) + (row['passing_touchdowns'] * 6) - (row['interceptions'] * -1) - (row['fumbles'] * 1)
    rushing_points = (row['rushing_yards'] * 0.1) + (row['rushing_touchdowns'] * 6) - (row['fumbles'] * 1)
    receiving_points = row['receptions'] + (row['receiving_yards'] * 0.1) + (row['receiving_touchdowns'] * 6) - (row['fumbles'] * 1)
    kicking_points = row['extra_points_made']
    
    # Add points for field goals
    for distance in ['under_19', '20_29', '30_39', '40_49', '50']:
        attempts_col = f'field_goals_attempted_{distance}'
        made_col = f'field_goals_made_{distance}'
        
        if distance == 'under_19' or distance == '20_29':
            kicking_points += row[made_col] * 3
        elif distance == '30_39':
            kicking_points += row[made_col] * 3 + row[made_col] * 0.1 * (row[attempts_col] - 30)
        elif distance == '40_49':
            kicking_points += row[made_col] * 4 + row[made_col] * 0.1 * (row[attempts_col] - 40)
        elif distance == '50':
            kicking_points += row[made_col] * 5 + row[made_col] * 0.1 * (row[attempts_col] - 50)

    return passing_points + rushing_points + receiving_points + kicking_points


def train_and_evaluate(X, y, test_size=0.2, random_state=None):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Prediction and evaluation
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, scaler, model # Return the scaler and model for later use

# Load data
# passing_data = pd.read_csv('./passing.csv')
# rushing_data = pd.read_csv('./rushing.csv')
# receiving_data = pd.read_csv('./receiving.csv')
# kicking_data = pd.read_csv('./kicking.csv')

passing_data = pd.read_csv('/Users/georgesmall/Documents/FantasyFootballML/FantasyFootballML/fantasy_football/passing.csv')
rushing_data = pd.read_csv('/Users/georgesmall/Documents/FantasyFootballML/FantasyFootballML/fantasy_football/rushing.csv')
receiving_data = pd.read_csv('/Users/georgesmall/Documents/FantasyFootballML/FantasyFootballML/fantasy_football/receiving.csv')
kicking_data = pd.read_csv('/Users/georgesmall/Documents/FantasyFootballML/FantasyFootballML/fantasy_football/kicking.csv')


# Merge dataframes on a common identifier, e.g., player_id
merged_data = passing_data.merge(rushing_data, on='player_id', how='outer')
merged_data = merged_data.merge(receiving_data, on='player_id', how='outer')
merged_data = merged_data.merge(kicking_data, on='player_id', how='outer')

merged_data.fillna(0, inplace=True)  # You can choose a different strategy for missing values

merged_data['fantasy_points'] = merged_data.apply(calculate_fantasy_points, axis=1)

print(merged_data.head())

# Preprocessing and feature selection
#no features such as player_name, player_team, and player_position. only numerical features used to train the model
feature_columns = ['player_age', 'games_played', 'games_started', 'passes_completed', 'passes_attempted', 'passing_yards',
                   'passing_touchdowns', 'interceptions', 'passing_first_downs', 'longest_pass', 'passer_rating',
                   'times_sacked', 'yards_lost_from_sacks', 'game_winning_drives', 'rushing_attempts', 'rushing_yards', 
                   'rushing_touchdowns', 'rushing_first_downs', 'longest_rush', 'fumbles', 'targets', 'receptions', 
                   'receiving_yards', 'receiving_touchdowns', 'receiving_first_downs', 'longest_reception',
                   'field_goals_attempted_under_19', 'field_goals_made_under_19', 'field_goals_attempted_20_29',
                   'field_goals_made_20_29', 'field_goals_attempted_30_39', 'field_goals_made_30_39',
                   'field_goals_attempted_40_49', 'field_goals_made_40_49', 'field_goals_attempted_50',
                   'field_goals_made_50', 'total_field_goals_attempted', 'total_field_goals_made', 
                   'longest_field_goal_made', 'extra_points_attempted', 'extra_points_made', 'kickoffs', 
                   'kickoff_yards', 'touchbacks', 'kickoff_average_yardage']

# It's possible that not all players will have data in all categories, so fill missing columns with 0
for col in feature_columns:
    if col not in merged_data.columns:
        merged_data[col] = 0

X = merged_data[feature_columns]
y = merged_data['fantasy_points']

# Number of iterations
num_runs = 10
mse_scores = []
r2_scores = []

for i in range(num_runs):
    mse, r2, scaler, model = train_and_evaluate(X, y, test_size=0.2, random_state=i)
    mse_scores.append(mse)
    r2_scores.append(r2)
    print(f'Run {i+1}: MSE = {mse}, R2 = {r2}')

# Average results
print(f'Average MSE: {np.mean(mse_scores)}')
print(f'Average R2: {np.mean(r2_scores)}')

# Optionally save the final model
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
final_model = LinearRegression()
final_model.fit(X_scaled, y)  # Train on the full dataset

# Save the scaler and model
joblib.dump(final_scaler, 'scaler.pkl')
joblib.dump(final_model, 'model.pkl')
