import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class FlightDelayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flight Delay Predictor")
        self.root.geometry("500x500")
        
        # UI Styles
        self.padding = {'padx': 10, 'pady': 5}
        self.label_font = ('Helvetica', 10, 'bold')
        
        # Data and Model Placeholders
        self.df = None
        self.model = None
        self.encoders = {}
        self.unique_values = {}
        
        # Status Label
        self.status_label = tk.Label(root, text="Loading data, training model, and generating charts...\nPlease wait.", fg="red")
        self.status_label.pack(pady=20)
        self.root.update() # Force update to show the loading text
        
        # Run training immediately
        self.load_and_train()
        
        # Clear loading label and build UI
        self.status_label.destroy()
        self.build_ui()

    def load_and_train(self):
        try:
            # 1. Connect and Fetch Data
            db_name = 'final_flights.db'
            conn = sqlite3.connect(db_name)
            
            # The app will automatically load all rows available in the database (e.g., 100,000)
            sql_query = """
            SELECT
                f.MONTH,
                f.DAY,
                f.SCHEDULED_DEPARTURE,
                f.DEPARTURE_DELAY,
                al.AIRLINE as AIRLINE,
                ap1.CITY as ORIGIN_CITY,
                ap2.CITY as DEST_CITY,
                CASE
                    WHEN f.DEPARTURE_DELAY > 15 THEN 1
                    ELSE 0
                END AS IS_DELAYED
            FROM flights f
            JOIN airlines al ON f.AIRLINE = al.IATA_CODE
            JOIN airports ap1 ON f.ORIGIN_AIRPORT = ap1.IATA_CODE
            JOIN airports ap2 ON f.DESTINATION_AIRPORT = ap2.IATA_CODE
            """
            
            self.df = pd.read_sql_query(sql_query, conn)
            conn.close()

            # Store unique values for the GUI Dropdowns BEFORE encoding
            self.unique_values['AIRLINE'] = sorted(self.df['AIRLINE'].unique())
            self.unique_values['ORIGIN_CITY'] = sorted(self.df['ORIGIN_CITY'].unique())
            self.unique_values['DEST_CITY'] = sorted(self.df['DEST_CITY'].unique())

            # 2. Encode Categorical Columns
            self.encoders['AIRLINE'] = LabelEncoder()
            self.df['AIRLINE'] = self.encoders['AIRLINE'].fit_transform(self.df['AIRLINE'])
            
            self.encoders['ORIGIN_CITY'] = LabelEncoder()
            self.df['ORIGIN_CITY'] = self.encoders['ORIGIN_CITY'].fit_transform(self.df['ORIGIN_CITY'])
            
            self.encoders['DEST_CITY'] = LabelEncoder()
            self.df['DEST_CITY'] = self.encoders['DEST_CITY'].fit_transform(self.df['DEST_CITY'])

            # 3. Define Features and Target
            feature_cols = ['MONTH', 'DAY', 'SCHEDULED_DEPARTURE', 'AIRLINE', 'ORIGIN_CITY', 'DEST_CITY']
            X = self.df[feature_cols]
            y = self.df['IS_DELAYED']

            # 4. Split Data (80% Train, 20% Test)
            print("Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 5. Train Model
            # Added class_weight='balanced' to handle imbalance between on-time and delayed flights
            # Added n_jobs=-1 to use all CPU cores for faster training on larger datasets
            print("Training Random Forest model (n_estimators=100)...")
            self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
            self.model.fit(X_train, y_train)
            
            # 6. Evaluate Model
            y_pred = self.model.predict(X_test)
            
            print("\n--- Model Evaluation ---")
            print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # 7. Generate Feature Importance Chart
            print("Generating feature importance chart...")
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
            plt.title('Feature Importance for Flight Delays')
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('feature_importance.png')
            print("Chart saved as 'feature_importance.png'")
            
        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("Error", f"Failed to load data or train model.\n{e}")
            self.root.destroy()

    def build_ui(self):
        # Title
        title = tk.Label(self.root, text="Flight Delay Predictor", font=("Helvetica", 16, "bold"))
        title.pack(pady=10)

        # Form Container
        form_frame = tk.Frame(self.root)
        form_frame.pack(pady=10)

        # --- Inputs ---

        # Month
        tk.Label(form_frame, text="Month:", font=self.label_font).grid(row=0, column=0, sticky="e", **self.padding)
        self.combo_month = ttk.Combobox(form_frame, values=[str(i) for i in range(1, 13)], state="readonly")
        self.combo_month.grid(row=0, column=1, **self.padding)
        self.combo_month.current(0)

        # Day
        tk.Label(form_frame, text="Day:", font=self.label_font).grid(row=1, column=0, sticky="e", **self.padding)
        self.combo_day = ttk.Combobox(form_frame, values=[str(i) for i in range(1, 32)], state="readonly")
        self.combo_day.grid(row=1, column=1, **self.padding)
        self.combo_day.current(0)

        # Airline
        tk.Label(form_frame, text="Airline:", font=self.label_font).grid(row=2, column=0, sticky="e", **self.padding)
        self.combo_airline = ttk.Combobox(form_frame, values=self.unique_values['AIRLINE'], state="readonly", width=30)
        self.combo_airline.grid(row=2, column=1, **self.padding)
        if self.unique_values['AIRLINE']: self.combo_airline.current(0)

        # Origin City
        tk.Label(form_frame, text="Origin City:", font=self.label_font).grid(row=3, column=0, sticky="e", **self.padding)
        self.combo_origin = ttk.Combobox(form_frame, values=self.unique_values['ORIGIN_CITY'], state="readonly", width=30)
        self.combo_origin.grid(row=3, column=1, **self.padding)
        if self.unique_values['ORIGIN_CITY']: self.combo_origin.current(0)

        # Destination City
        tk.Label(form_frame, text="Destination City:", font=self.label_font).grid(row=4, column=0, sticky="e", **self.padding)
        self.combo_dest = ttk.Combobox(form_frame, values=self.unique_values['DEST_CITY'], state="readonly", width=30)
        self.combo_dest.grid(row=4, column=1, **self.padding)
        if self.unique_values['DEST_CITY']: self.combo_dest.current(0)

        # Scheduled Departure Hour
        tk.Label(form_frame, text="Departure Hour (0-23):", font=self.label_font).grid(row=5, column=0, sticky="e", **self.padding)
        self.combo_hour = ttk.Combobox(form_frame, values=[str(i) for i in range(0, 24)], state="readonly")
        self.combo_hour.grid(row=5, column=1, **self.padding)
        self.combo_hour.current(12)

        # --- Button ---
        predict_btn = tk.Button(self.root, text="Predict Delay", command=self.make_prediction, bg="#4CAF50", fg="black", font=("Helvetica", 12, "bold"), padx=20, pady=5)
        predict_btn.pack(pady=20)
        
        # Info Label
        info_label = tk.Label(self.root, text="Check console for Accuracy Score & feature_importance.png", font=("Helvetica", 8, "italic"), fg="gray")
        info_label.pack(pady=5)

    def make_prediction(self):
        try:
            # 1. Get Inputs
            month = int(self.combo_month.get())
            day = int(self.combo_day.get())
            hour = int(self.combo_hour.get())
            airline = self.combo_airline.get()
            origin = self.combo_origin.get()
            dest = self.combo_dest.get()

            # 2. Preprocess Input
            # Convert Hour to 'HH00' format (e.g., 14 -> 1400) to match dataset's SCHEDULED_DEPARTURE
            scheduled_departure = hour * 100 

            # Encode Categorical inputs using the fitted encoders
            airline_encoded = self.encoders['AIRLINE'].transform([airline])[0]
            origin_encoded = self.encoders['ORIGIN_CITY'].transform([origin])[0]
            dest_encoded = self.encoders['DEST_CITY'].transform([dest])[0]

            # 3. Create Feature Array as DataFrame
            # Using a DataFrame with column names eliminates the "valid feature names" warning
            features_df = pd.DataFrame(
                [[month, day, scheduled_departure, airline_encoded, origin_encoded, dest_encoded]], 
                columns=['MONTH', 'DAY', 'SCHEDULED_DEPARTURE', 'AIRLINE', 'ORIGIN_CITY', 'DEST_CITY']
            )

            # 4. Predict
            prediction = self.model.predict(features_df)[0]
            
            # 5. Show Result
            if prediction == 1:
                messagebox.showwarning("Prediction Result", "Likely Delayed! ⚠️")
            else:
                messagebox.showinfo("Prediction Result", "On Time ✅")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FlightDelayApp(root)
    root.mainloop()