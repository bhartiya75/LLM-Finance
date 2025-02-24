import pandas as pd

# Load GuruFocus data
print("Loading firm_data.csv...")
data = pd.read_csv("firm_data.csv")
print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns: {data.columns.tolist()}")

# Clean data (remove NaN, format dates, etc.)
print("Cleaning data...")
# Use a more lenient fillna to handle NaN
data = data.fillna('N/A')  # Replace NaN with 'N/A' or another placeholder
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.sort_values('Date')

# Include all columns in the text output (excluding 'text' if it exists)
print("Processing columns...")
columns = [col for col in data.columns if col != 'text']
print(f"Columns to process: {columns}")
data['text'] = data.apply(lambda row: ', '.join([f"{col}: {row[col]}" for col in columns]), axis=1)

# Add a numeric label using the stock price column, with cleaning
stock_price_column = 'Month End Stock Price'  # Adjust this to the actual column name from your file
if stock_price_column not in data.columns:
    raise KeyError(f"Stock price column '{stock_price_column}' not found. Available columns: {data.columns.tolist()}")

# Clean the stock price column: remove non-numeric characters, handle 'N/A', and convert to float
print(f"Cleaning {stock_price_column} column...")
data[stock_price_column] = data[stock_price_column].replace('N/A', pd.NA).str.replace('[^\d.]', '', regex=True)
# Use 'ignore' for to_numeric to handle non-convertible values, then coerce NaN with astype
data[stock_price_column] = pd.to_numeric(data[stock_price_column], errors='ignore').replace('', pd.NA)
# Use astype without errors parameter (defaults to 'raise', but we’ve cleaned data to avoid errors)
data[stock_price_column] = data[stock_price_column].astype(float)
print(f"Sample values after cleaning: {data[stock_price_column].head().tolist()}")

# Drop rows where the stock price is NaN (optional, to ensure valid labels)
data = data.dropna(subset=[stock_price_column])
print(f"After dropping NaN in {stock_price_column}, {len(data)} rows remain")

# Set label as the cleaned stock price
data['label'] = data[stock_price_column]

# Save cleaned text and labels
data[['text', 'label']].to_csv("financial_text_regression.csv", index=False)
print("Done. Check /Users/shekhar/LLM_Finance_Local/financial_text_regression.csv")