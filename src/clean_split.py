import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from sklearn.model_selection import train_test_split

data = pd.read_csv(config.DATA_FILE)

# handle missing values
missing_values = data.isnull().sum()
print(f"checking for missing values in the dataset before handling {missing_values}")

data_shape = data.shape
print(f"the shape of the data is: {data_shape}")

# since we have 17340 observations (rows), we can remove the 3 records that have missing value in the cleaned_review feature
data.dropna(subset=['cleaned_review'], inplace=True)  # remove records with null values in the 'cleaned_review' column

missing_values = data.isnull().sum()
print(f"checking for missing values in the dataset after handling {missing_values}")

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the train and test sets to separate CSV files
train_data.to_csv('../input/train.csv', index=False)
test_data.to_csv('../input/test.csv', index=False)

