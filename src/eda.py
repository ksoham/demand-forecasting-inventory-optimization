import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def perform_eda(df):
    """Perform basic exploratory data analysis."""
    
    # Display basic info
    print("Dataset Info:")
    print(df.info())

    # Show summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Check missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Visualize distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Demand Forecast'], bins=30, kde=True)
    plt.title("Demand Forecast Distribution")
    plt.show()

if __name__ == "__main__":
    file_path = "data/retail_store_inventory.csv"
    df = load_data(file_path)
    perform_eda(df)
