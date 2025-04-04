import pandas as pd
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from dask_ml.linear_model import LinearRegression as DaskLinearRegression
from dask_ml.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np 
import plotly.graph_objects as go
from flask import Flask, request, jsonify
import joblib
import os
import warnings

#Suppress future warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
MODEL_PATH = "covidModel.joblib"

#Function to train and save the model
def train_model(client):
    print("Client: ", client)
    print("Dashboard: ", client.dashboard_link)

    #Loading data with Dask
    df = dd.read_csv("CovidLive.csv", encoding="UTF-8", assume_missing=True)

    df = df.rename(columns={col: col.replace(' ', '_') for col in df.columns})
    df = df.rename(columns={
        'Tot_Cases/1M_pop': 'Tot_Cases_per_1M_pop',
        'Deaths/1M_pop': 'Deaths_per_1M_pop',
        'Tests/1M_pop': 'Tests_per_1M_pop',
        '#': 'id'
    })

    #Imputation
    mean_cols = ["Active_Cases", "Total_Deaths", "Total_Recovered", "Tot_Cases_per_1M_pop", 
                 "Deaths_per_1M_pop", "Total_Tests", "Tests_per_1M_pop", "Population"]
    median_cols = ["New_Deaths", "Serious_Critical"]

    for col in mean_cols:
        mean_val = df[col].mean().compute()
        df[col] = df[col].fillna(mean_val)
    for col in median_cols:
        median_val = df[col].quantile(0.5).compute()
        df[col] = df[col].fillna(median_val)

    #Model training
    X = df[['Total_Cases']]
    y = df['Total_Deaths']

    #Data splitting for more accurate performance estimates
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

    #Train the model
    model = DaskLinearRegression()
    model.fit(X_train.values.compute(), y_train.compute()) #Compute to numpy array for DaskML

    #Predictions on the test set
    y_test_computed = y_test.compute()
    y_pred = model.predict(X_test.values.compute())

    #Performance evaluation
    mse = mean_squared_error(y_test_computed, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_computed, y_pred)
    r2 = r2_score(y_test_computed, y_pred)

    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-Squared: {r2:.4f}")

    #Actual Predictions and virtualization 
    pd_df = df.compute()
    predictions = model.predict(pd_df[['Total_Cases']].to_numpy())

    fig = go.Figure([
        go.Scatter(x=pd_df['Total_Cases'], y=pd_df['Total_Deaths'], mode='markers', 
                   text=pd_df['Country'], name='Actual Total Deaths'),
        go.Scatter(x=pd_df['Total_Cases'], y=predictions, name='Predicted Deaths')
    ])
    fig.update_layout(title='Deaths vs Total Cases', xaxis_title='Total Cases', yaxis_title='Deaths')
    fig.write_html("output.html") #Store the visualization

    #Storing the model
    joblib.dump(model, MODEL_PATH)

#Prediction function
def make_prediction(data):
    #Load the pre-trained model
    model = joblib.load(MODEL_PATH)

    total_cases = np.array([[data.get("Total_Cases", 0)]])  #Default to 0 if not provided
    prediction = model.predict(total_cases)[0]

    return {"Predicted deaths": int(prediction)}

#API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or "Total_Cases" not in data:
            return jsonify({"error": "Total cases is required"}), 400
        
        prediction = make_prediction(data)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    #Initialize Dask client
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, dashboard_address=':8787')
    client = Client(cluster)

    print("Client: ", client)
    print("Dashboard: ", client.dashboard_link)

    #Enable force retraining
    force_retrain = input("Force retraining? (y/n): ").lower() == 'y'
    if force_retrain or not os.path.exists(MODEL_PATH):
        print("Training model...")
        train_model(client)
    else:
        print("Model already exists, skipping training.")

    #Start Flask server
    print("Starting Flask API...")
    try:
        app.run(debug=True, host="127.0.0.1", port=5000)
    finally:
        client.close() #Ensure safe closing of rhe cluster
        cluster.close()
