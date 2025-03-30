from data_preprocessing import load_and_preprocess_data

# Define the dataset path
file_path = "data/retail_store_inventory.csv"  # Use this instead of "../data/..."


# Load and preprocess data
df = load_and_preprocess_data(file_path)

# Display basic info
print(df.info())
print(df.head())
