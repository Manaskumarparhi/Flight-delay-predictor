
Flight Delay Predictor :

A Python desktop application that predicts flight delays using a Random Forest Machine Learning model. Built with Tkinter, Pandas, and SQLite.

Features:

Data Processing: Merges and cleans real-world flight data (Flights, Airlines, Airports) into a SQLite database.

Machine Learning: Trains a Random Forest Classifier on flight features (Month, Day, Airline, Origin, Destination, Time).

Interactive GUI: User-friendly desktop interface built with Tkinter.

Visual Analytics: Generates feature importance charts to visualize what causes delays.

Tech Stack:

Language: Python 3.x

GUI: Tkinter

Data Manipulation: Pandas, SQLite3

Machine Learning: Scikit-Learn (Random Forest)

Visualization: Matplotlib

Setup & Installation:

Clone the repository (or download the files):

git clone [https://github.com/manas/flight-delay-predictor.git](https://github.com/Manaskumarparhi/flight-delay-predictor.git)
cd flight-delay-predictor



Install required dependencies:

pip install pandas scikit-learn matplotlib



Add Data Files:
Note: Due to file size limits, the raw CSV files are not included in this repo.

Download the 2015 Flight Delays dataset.

Place the following files in the project root folder:

flights.csv

airlines.csv

airports.csv

How to Run:

Step 1: Build the Database
Process the raw CSV files into a local SQLite database. This prepares the training data.

python create_flight_db.py



Step 2: Launch the App
Start the GUI. The application will train the model fresh upon launch.

python app.py



Model Performance:

The Random Forest model is configured with class_weight='balanced' to better detect delays.

Recall (Delay Detection): ~48% (on 100k row dataset)

Overall Accuracy: ~74-82% (depending on dataset size)
