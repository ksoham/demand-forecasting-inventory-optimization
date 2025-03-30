import pandas as pd

def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Clean and preprocess the dataset."""
    
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Fill missing numerical values with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # Handle outliers (clip values beyond 99th percentile)
    for col in num_cols:
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))

    return df

if __name__ == "__main__":
    file_path = "data/retail_store_inventory.csv"
    df = load_data(file_path)
    df_cleaned = clean_data(df)

    # Save the cleaned data
    df_cleaned.to_csv("data/cleaned_retail_inventory.csv", index=False)
    print("Data Cleaning Complete! Cleaned dataset saved.")
import pandas as pd

def load_and_preprocess_data(file_path):
    """Loads the dataset and applies preprocessing steps."""
    df = pd.read_csv(file_path)

    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Convert categorical columns to category type
    categorical_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Fill missing values
    df.fillna(method='ffill', inplace=True)  # Forward fill

    return df
