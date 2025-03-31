import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


# 1️⃣ Data Preprocessing
def preprocess_data(df):
    """Cleans and preprocesses the dataset."""
    print("\n🔹 Step 1: Data Cleaning...")

    # Convert 'Date' column to datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # Fill missing values
    df.fillna({
        "Demand Forecast": df["Demand Forecast"].median(),
        "Competitor Pricing": df["Competitor Pricing"].median(),
    }, inplace=True)

    return df


# 2️⃣ Feature Engineering
def feature_engineering(df):
    """Creates new features for better model performance."""
    print("\n🔹 Step 2: Feature Engineering...")

    # Extracting date-related features
    if "Date" in df.columns:
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Weekday"] = df["Date"].dt.weekday

    # Creating sales-to-inventory ratio
    if "Units Sold" in df.columns and "Inventory Level" in df.columns:
        df["Sales_Inventory_Ratio"] = df["Units Sold"] / (df["Inventory Level"] + 1)

    # Encoding seasonality
    if "Seasonality" in df.columns:
        df["Is_Winter"] = df["Seasonality"].apply(lambda x: 1 if x == "Winter" else 0)
        df["Is_Summer"] = df["Seasonality"].apply(lambda x: 1 if x == "Summer" else 0)

    return df



def remove_highly_correlated_features(df, threshold=0.9):
    """Removes highly correlated numeric features from the dataset."""
    print("\n🔹 Step 3.1: Removing Highly Correlated Features...")

    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        print("⚠️ No numeric columns found for correlation analysis.")
        return df

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    # Get upper triangle of correlation matrix (excluding diagonal)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns with correlation greater than threshold
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold) and col != "Demand Forecast"]


    print(f"📌 Dropping correlated features: {to_drop}")

    return df.drop(columns=to_drop, errors="ignore")


# 4️⃣ Feature Selection Using Random Forest
def select_features_using_model(df, target_col):
    """
    Selects important features using a Random Forest model.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with features and target variable.
        target_col (str): Target column name.

    Returns:
        pd.DataFrame: DataFrame with selected features.
    """
    # Drop target column to create feature set
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert Date column to numeric if it exists
    if "Date" in X.columns:
        X["Date"] = pd.to_datetime(X["Date"]).astype("int64")  # Convert to timestamp

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Encode categorical columns using one-hot encoding
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)  

    # Train RandomForest model for feature selection
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Get feature importances
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    
    # Select top N features based on importance (e.g., top 10)
    selected_features = feature_importances.nlargest(10).index.tolist()

    print(f"📌 Selected Features: {selected_features}")
    
    return df[selected_features + [target_col]]


# 5️⃣ Chi-Square Test for Categorical Features
def chi_square_feature_selection(df, target_col):
    """Performs Chi-Square test to select categorical features."""
    print("\n🔹 Step 3.3: Performing Chi-Square Test for Categorical Features...")

    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in categorical_features:
        categorical_features.remove(target_col)

    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    
    selected_features = []
    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])
        chi2_score, p_value = chi2(df[[feature]], y)
        if p_value < 0.05:
            selected_features.append(feature)

    print(f"✅ Selected Categorical Features (p < 0.05): {selected_features}")
    
    return selected_features


# 6️⃣ Perform Feature Selection
def perform_feature_selection(df, target_col):
    """Performs feature selection using correlation, Random Forest, and statistical tests."""
    print("\n🔹 Step 4: Performing Feature Selection...\n")
    
    # Debugging step
    print("📌 Columns in DataFrame Before Feature Selection:", df.columns.tolist())
    print("\n🔍 Checking 'Demand Forecast' column type...")
    print(df["Demand Forecast"].dtype)  # Check data type

    # Check for leading/trailing spaces in column names
    print("\n🔍 Checking exact column names...")
    print(list(df.columns))

    # Ensure 'Demand Forecast' is properly referenced
    assert "Demand Forecast" in df.columns, "❌ 'Demand Forecast' column is not accessible!"

    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in dataframe.")
    
    df = remove_highly_correlated_features(df)
    df = select_features_using_model(df, target_col)
    
    selected_categorical_features = chi_square_feature_selection(df, target_col)
    
    df = df[selected_categorical_features + df.select_dtypes(include=["number"]).columns.tolist()]
    
    print("\n✅ Final Feature Set:")
    print(df.columns.tolist())

    return df


# 7️⃣ Load and Preprocess Data
def load_and_preprocess_data(file_path):
    """Loads, cleans, applies feature engineering, and selects features."""
    print("\n🚀 Loading and Preprocessing Data...")
    df = pd.read_csv(file_path)

    df = preprocess_data(df)
    df = feature_engineering(df)
    df = perform_feature_selection(df, target_col="Demand Forecast")

    print("\n✅ Data Preprocessing Completed!")
    return df
