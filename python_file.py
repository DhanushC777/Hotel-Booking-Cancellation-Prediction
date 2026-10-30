# HOTEL BOOKING CANCELLATION PREDICTION AND ANALYSIS

# Import the required libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")


# LOAD THE DATASET
# Load the hotel booking dataset from the local CSV file
df = pd.read_csv('hotel_booking.csv')

# Display the first five records
print(df.head(5))

# Display the last five records
print(df.tail(5))

# Display dataset information, including columns, data types, and non-null counts
df.info()

# Display descriptive statistics for numerical columns
print(df.describe())


# REMOVING UNWANTED COLUMNS
# Booking_ID is an identifier and does not help predict booking cancellation
# The reservation date will be used to create year and month features later
df.drop(columns=['Booking_ID'], inplace=True)


# DATA PREPROCESSING AND TYPE CONVERSION
# Remove extra spaces from categorical values
categorical_columns = ['type of meal', 'room type', 'market segment type', 'booking status']

for col in categorical_columns:
    df[col] = df[col].astype('string').str.strip()

# Standardize booking status values
df['booking status'] = df['booking status'].str.replace('_', ' ', regex=False).str.title()

# Convert numerical columns to appropriate numeric data types
numeric_columns = [
    'number of adults', 'number of children', 'number of weekend nights',
    'number of week nights', 'car parking space', 'lead time', 'repeated',
    'P-C', 'P-not-C', 'average price', 'special requests'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert reservation date into datetime format
df['date of reservation'] = pd.to_datetime(
    df['date of reservation'], errors='coerce'
)


# DUPLICATE VALUE DETECTION AND REMOVAL
# Count and display duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicate rows from the dataset
df.drop_duplicates(inplace=True)


# MISSING VALUE ANALYSIS AND HANDLING
# Display the number of missing values in each column before handling
print("\nMissing values before handling:")
print(df.isnull().sum())

# Replace missing numerical values with the median
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Replace missing categorical values with the mode
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove records where the reservation date could not be converted
