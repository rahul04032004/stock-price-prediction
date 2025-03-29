import yfinance as yf
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to download stock data with retry logic
def fetch_stock_data(symbol, start_date, end_date, retries=3, delay=5):
    for i in range(retries):
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            if df.empty:
                raise ValueError("Yahoo Finance returned empty data.")
            return df
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(delay)  # Wait before retrying
    raise Exception("Failed to fetch stock data after multiple attempts.")

# Set parameters
symbol = "GOOGL"
start_date = "2022-07-16"
end_date = "2025-02-07"

try:
    df = fetch_stock_data(symbol, start_date, end_date)
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# Data Preprocessing
df["HL_PCT"] = (df["High"] - df["Low"]) / df["Low"] * 100
df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100
df = df[["Close", "HL_PCT", "PCT_change", "Volume"]]

# Handle missing values
df.fillna(-99999, inplace=True)

# Forecasting setup
forecast_out = int(np.ceil(0.01 * len(df)))
df["label"] = df["Close"].shift(-forecast_out)

# Prepare training data
X = np.array(df.drop(["label"], axis=1))
if X.shape[0] == 0:
    raise ValueError("Data array is empty. Check if stock data was fetched correctly.")

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df["label"])

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train the model
clf = LinearRegression()
clf.fit(X_train, y_train)

# Model accuracy and predictions
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

# Add predictions to the dataframe
df["Forecast"] = np.nan
last_date = df.index[-1]
next_date = last_date + pd.DateOffset(1)

for prediction in forecast_set:
    while next_date in df.index:
        next_date += pd.DateOffset(1)
    df.loc[next_date] = [np.nan] * (len(df.columns) - 1) + [prediction]
    next_date += pd.DateOffset(1)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Actual Price", color="blue")
plt.plot(df.index, df["Forecast"], label="Predicted Price", color="red", linestyle="dashed")
plt.legend(loc="best")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"{symbol} Stock Price Prediction")
plt.show()

print(f"Model Accuracy (R² Score): {accuracy}")
