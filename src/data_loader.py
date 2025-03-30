import pandas as pd

def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    df = load_data("data/retail_store_inventory.csv")
    print(df.head())  # Show first 5 rows
