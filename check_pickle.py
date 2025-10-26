import pickle
import pandas as pd

# Load the pickled dataframe
df = pickle.load(open('job_roles_dataset.pkl', 'rb'))

# Print data types
print("Data types:")
print(df.dtypes)

print("\nFirst few rows:")
print(df.head())

print("\nSample row:")
print(df.iloc[0])