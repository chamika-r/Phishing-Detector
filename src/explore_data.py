import pandas as pd

# Load the dataset
df = pd.read_csv('data/emails.csv')

# Basic info
print("Shape of dataset:", df.shape)
print("\nFirst few columns:", df.columns[:5].tolist())
print("\nLast column:", df.columns[-1])
print("\nLabel distribution:")
print(df['Prediction'].value_counts())
print("\nSample of data (first 3 rows, first 5 cols):")
print(df.iloc[:3, :5])