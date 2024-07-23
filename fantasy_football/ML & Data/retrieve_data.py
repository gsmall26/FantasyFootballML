import pyodbc
import pandas as pd

# Database connection parameters
server = 'your_server_name'
database = 'your_database_name'
username = 'your_username'
password = 'your_password'

# Create a connection to the SQL Server database
conn = pyodbc.connect(
    f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
)

# Query to retrieve data
query = 'SELECT * FROM players'

# Retrieve data and save to CSV
data = pd.read_sql(query, conn)
data.to_csv('players_data.csv', index=False)

conn.close()
