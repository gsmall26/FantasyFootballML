import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

def train_and_evaluate(X, y, test_size=0.2, random_state=None):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediction and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2 # Mean squared error, R squared

# Load data
data = pd.read_csv('players_data.csv')

# Preprocessing and feature selection
X = data[['player_age', 'games_played', 'passing_yards', 'passing_tds', 'passing_attempts', 
          'passing_completions', 'interceptions', 'times_sacked', 'rushing_attempts', 
          'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 
          'receiving_tds', 'YAC']]
y = data['fantasy_points']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Number of iterations
num_runs = 10
mse_scores = []
r2_scores = []

for i in range(num_runs):
    # Initialize model within the loop to ensure fresh model for each iteration
    model = LinearRegression()
    mse, r2 = train_and_evaluate(X_scaled, y, test_size=0.2, random_state=i)
    mse_scores.append(mse)
    r2_scores.append(r2)
    print(f'Run {i+1}: MSE = {mse}, R2 = {r2}')

# Average results
print(f'Average MSE: {np.mean(mse_scores)}')
print(f'Average R2: {np.mean(r2_scores)}')

# Optionally save the final model
final_model = LinearRegression()
final_model.fit(X_scaled, y)  # Train on the full dataset
joblib.dump(final_model, 'model.pkl')
