import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load data
data = pd.read_csv('players_data.csv')

# Preprocessing and feature selection

# Train-test split
X = data[['player_age', 'games_played', 'passing_yards', 'passing_tds', 'passing_attempts', 
          'passing_completions', 'interceptions', 'times_sacked', 'rushing_attempts', 
          'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 
          'receiving_tds', 'YAC']]
y = data['fantasy_points']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
