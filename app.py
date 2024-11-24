import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import re
import joblib
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model.pkl")  # Ensure this is in the correct directory
scaler = joblib.load("scaler.pkl")  # Ensure this is in the correct directory

# Route for the homepage with the input form
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle form submission and prediction
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        # Get data from the form input
        UNIXTime = int(request.form["UNIXTime"])
        Data = request.form["Data"]
        Time = request.form["Time"]
        Temperature = float(request.form["Temperature"])
        Pressure = float(request.form["Pressure"])
        Humidity = float(request.form["Humidity"])
        WindDirection = float(request.form["WindDirection"])
        Speed = float(request.form["Speed"])
        TimeSunRise = request.form["TimeSunRise"]
        TimeSunSet = request.form["TimeSunSet"]

        # Convert the input data to a DataFrame
        input_data = {
            "UNIXTime": UNIXTime,
            "Data": Data,
            "Time": Time,
            "Temperature": Temperature,
            "Pressure": Pressure,
            "Humidity": Humidity,
            "WindDirection(Degrees)": WindDirection,
            "Speed": Speed,
            "TimeSunRise": TimeSunRise,
            "TimeSunSet": TimeSunSet
        }

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Feature extraction from Data and Time
        input_df['Month'] = pd.to_datetime(input_df['Data']).dt.month
        input_df['Day'] = pd.to_datetime(input_df['Data']).dt.day
        input_df['Hour'] = pd.to_datetime(input_df['Time']).dt.hour
        input_df['Minute'] = pd.to_datetime(input_df['Time']).dt.minute
        input_df['Second'] = pd.to_datetime(input_df['Time']).dt.second  # Ensure 'Second' is included

        # Extract sunrise and sunset hours and minutes
        input_df['risehour'] = input_df['TimeSunRise'].apply(lambda x: int(re.search(r'^\d+', x).group(0)))
        input_df['riseminuter'] = input_df['TimeSunRise'].apply(lambda x: int(re.search(r'(?<=\:)\d+', x).group(0)))
        input_df['sethour'] = input_df['TimeSunSet'].apply(lambda x: int(re.search(r'^\d+', x).group(0)))
        input_df['setminute'] = input_df['TimeSunSet'].apply(lambda x: int(re.search(r'(?<=\:)\d+', x).group(0)))

        # Drop the columns that are not needed for prediction
        input_df.drop(['Data', 'Time', 'TimeSunRise', 'TimeSunSet', 'UNIXTime'], axis=1, inplace=True)

        # Ensure the same order of columns as in the training phase
        expected_columns = ['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 
                            'Month', 'Day', 'Hour', 'Minute', 'Second', 'risehour', 'riseminuter', 'sethour', 'setminute']

        input_df = input_df[expected_columns]

        # Scaling the input data using the same scaler used during training
        scaled_input = scaler.transform(input_df)

        # Get the prediction from the model
        prediction = model.predict(scaled_input)

# Ensure non-negative prediction
        prediction = np.maximum(prediction, 0)

# Return the prediction result
#return render_template("result.html", prediction=prediction[0])
        # Return the prediction result
        return render_template("result.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
