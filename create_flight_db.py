import pandas as pd
import sqlite3
import os

def process_and_save_data():
    """
    Loads flight data, cleans it, and saves it to a SQLite database.
    """
    db_name = 'final_flights.db'
    
    print("Starting data processing...")

    try:
        # 1. Load airlines and airports data
        # We check if files exist to provide helpful error messages
        if not os.path.exists('airlines.csv') or not os.path.exists('airports.csv'):
            print("Error: airlines.csv or airports.csv not found.")
            return

        airlines = pd.read_csv('airlines.csv')
        airports = pd.read_csv('airports.csv')
        print("Loaded airlines and airports data.")

        # 2. Load the first 100,000 rows of flights.csv
        if not os.path.exists('flights.csv'):
            print("Error: flights.csv not found.")
            return

        print("Loading first 300000 rows of flights.csv...")
        flights = pd.read_csv('flights.csv', nrows=300000, low_memory=False)

        # 3. Clean the data
        # Drop rows where 'DEPARTURE_DELAY' is missing
        initial_rows = len(flights)
        flights = flights.dropna(subset=['DEPARTURE_DELAY'])
        cleaned_rows = len(flights)
        print(f"Dropped {initial_rows - cleaned_rows} rows with empty DEPARTURE_DELAY.")

        # 4. Connect to SQLite database
        conn = sqlite3.connect(db_name)

        # 5. Save DataFrames to the database
        # if_exists='replace' ensures that if you run this script twice, it overwrites the old tables
        print("Saving tables to database...")
        flights.to_sql('flights', conn, if_exists='replace', index=False)
        airlines.to_sql('airlines', conn, if_exists='replace', index=False)
        airports.to_sql('airports', conn, if_exists='replace', index=False)

        # Close the connection
        conn.close()

        print("Database created successfully")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_and_save_data()