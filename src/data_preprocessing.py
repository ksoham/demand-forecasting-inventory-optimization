import pandas as pd

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    df = preprocess_data(df)  # Clean the data
    df = feature_engineering(df)  # Apply feature engineering
    
    return df

import pandas as pd

def feature_engineering(df):
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Extract year, month, and day
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # Create a weekday/weekend feature
    df["Is_Weekend"] = df["Date"].dt.weekday >= 5

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=["Category", "Region", "Seasonality"], drop_first=True)

    # Normalize numerical features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[["Price", "Competitor Pricing", "Discount"]] = scaler.fit_transform(df[["Price", "Competitor Pricing", "Discount"]])

    return df
