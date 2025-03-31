import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(df):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.
    """
    print("\nChecking column data types:")
    print(df.dtypes)

    print("\nBasic Statistics:")
    print(df.describe(include="all"))  # Includes both numeric and categorical columns

    print("\nChecking for Missing Values:")
    print(df.isnull().sum())

    # Selecting only numeric columns for correlation
    numeric_df = df.select_dtypes(include=["number"])

    # Distribution of Demand Forecast
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Demand Forecast"], bins=30, kde=True)
    plt.title("Distribution of Demand Forecast")
    plt.show()

    # Plot Correlation Heatmap for Numeric Features
    if not numeric_df.empty:
        plt.figure(figsize=(12, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()
    else:
        print("No numeric columns found for correlation heatmap.")

    # Univariate Analysis - Distribution of Numeric Features
    numeric_df.hist(figsize=(12, 10), bins=30, edgecolor="black")
    plt.suptitle("Feature Distributions")
    plt.show()

    # Univariate Analysis - Countplot for Categorical Features
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        sns.countplot(x=df[col], order=df[col].value_counts().index, palette="viridis")
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {col}")
        plt.show()

    return df

  
