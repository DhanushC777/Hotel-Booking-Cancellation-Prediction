
# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("hotel_booking_cancellation.csv")

print("\nDATASET INFORMATION")
print("Shape:", df.shape)
print(df.info())
print("\nFirst five records:")
print(df.head())
print("\nDescriptive statistics:")
print(df.describe(include="all"))


# ============================================================
# 2. DATA PREPROCESSING AND CLEANING
# ============================================================

# Standardize column names.
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace("-", "_", regex=False)
    .str.replace(" ", "_", regex=False)
)

# Remove unwanted spaces and standardize text values.
text_columns = [
    col for col in df.columns
    if pd.api.types.is_object_dtype(df[col].dtype)
    or pd.api.types.is_string_dtype(df[col].dtype)
]
for col in text_columns:
    df[col] = df[col].str.strip()

# Standardize target labels.
df["booking_status"] = df["booking_status"].str.replace(" ", "_")

# Check duplicates using the complete original records, including Booking_ID.
# This prevents removing different bookings that happen to share the same details.
print("\nDuplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Remove Booking_ID because it is only an identifier for analysis and prediction.
df.drop("booking_id", axis=1, inplace=True)

# Convert numeric columns to numeric data types.
numeric_columns = [
    "number_of_adults", "number_of_children",
    "number_of_weekend_nights", "number_of_week_nights",
    "car_parking_space", "lead_time", "repeated",
    "p_c", "p_not_c", "average_price", "special_requests"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert reservation date.
df["date_of_reservation"] = pd.to_datetime(
    df["date_of_reservation"], format="%m/%d/%Y", errors="coerce"
)

# Display missing values before handling.
print("\nMissing values before handling:")
print(df.isnull().sum())

# Fill numerical missing values with median.
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode.
categorical_columns = [
    col for col in df.columns
    if pd.api.types.is_object_dtype(df[col].dtype)
    or pd.api.types.is_string_dtype(df[col].dtype)
]

for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Fill missing dates with the most common date.
if df["date_of_reservation"].isnull().sum() > 0:
    df["date_of_reservation"] = df["date_of_reservation"].fillna(
        df["date_of_reservation"].mode()[0]
    )

# Keep only valid target categories.
df = df[df["booking_status"].isin(
    ["Canceled", "Not_Canceled"]
)].copy()

print("\nMissing values after handling:")
print(df.isnull().sum())

print("\nBooking status categories:")
print(df["booking_status"].unique())


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Extract useful date information.
df["reservation_year"] = df["date_of_reservation"].dt.year
df["reservation_month"] = df["date_of_reservation"].dt.month
df["reservation_day"] = df["date_of_reservation"].dt.day

# Calculate total nights and total guests.
df["total_nights"] = (
    df["number_of_weekend_nights"] +
    df["number_of_week_nights"]
)

df["total_guests"] = (
    df["number_of_adults"] +
    df["number_of_children"]
)

# Remove original date after extracting features.
df.drop("date_of_reservation", axis=1, inplace=True)

print("\nFeature engineering completed.")
print("Created: reservation_year, reservation_month, reservation_day,")
print("total_nights and total_guests.")


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

sns.set_style("whitegrid")

# Correlation heatmap for numerical columns.
plt.figure(figsize=(11, 7))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# 5. FIVE OBJECTIVES OF THE PROJECT
# ============================================================

# OBJECTIVE 1: BOOKING STATUS ANALYSIS
print("\nOBJECTIVE 1: BOOKING STATUS ANALYSIS")
print(df["booking_status"].value_counts())

print("\nBooking status percentage:")
print((df["booking_status"].value_counts(normalize=True) * 100).round(2))

# OBJECTIVE 2: LEAD TIME ANALYSIS
print("\nOBJECTIVE 2: LEAD TIME ANALYSIS")
print(df.groupby("booking_status")["lead_time"].mean())

# OBJECTIVE 3: MARKET SEGMENT ANALYSIS
print("\nOBJECTIVE 3: MARKET SEGMENT ANALYSIS")
segment_rate = (
    df.groupby("market_segment_type")["booking_status"]
    .apply(lambda x: (x == "Canceled").mean() * 100)
    .sort_values(ascending=False)
)
print("Cancellation rate by market segment:")
print(segment_rate.round(2))

# OBJECTIVE 4: AVERAGE PRICE ANALYSIS
print("\nOBJECTIVE 4: AVERAGE PRICE ANALYSIS")
print(df.groupby("booking_status")["average_price"].mean())

# OBJECTIVE 5: SPECIAL REQUESTS ANALYSIS
print("\nOBJECTIVE 5: SPECIAL REQUESTS ANALYSIS")
request_rate = (
    df.groupby("special_requests")["booking_status"]
    .apply(lambda x: (x == "Canceled").mean() * 100)
)
print("Cancellation rate by number of special requests:")
print(request_rate.round(2))


# ============================================================
# 6. DATA VISUALIZATION
# ============================================================

# GRAPH 1: BOOKING STATUS
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="booking_status", color="steelblue")
plt.title("Hotel Booking Status")
plt.xlabel("Booking Status")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.show()

# GRAPH 2: LEAD TIME BY BOOKING STATUS
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="booking_status", y="lead_time")
plt.title("Lead Time by Booking Status")
plt.xlabel("Booking Status")
plt.ylabel("Lead Time (Days)")
plt.tight_layout()
plt.show()

# GRAPH 3: CANCELLATION RATE BY MARKET SEGMENT
plt.figure(figsize=(8, 5))
segment_rate.plot(kind="bar")
plt.title("Cancellation Rate by Market Segment")
plt.xlabel("Market Segment")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# GRAPH 4: AVERAGE PRICE BY BOOKING STATUS
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="booking_status", y="average_price")
plt.title("Average Price by Booking Status")
plt.xlabel("Booking Status")
plt.ylabel("Average Price")
plt.tight_layout()
plt.show()

# GRAPH 5: CANCELLATION RATE BY SPECIAL REQUESTS
plt.figure(figsize=(8, 5))
request_rate.plot(kind="bar")
plt.title("Cancellation Rate by Special Requests")
plt.xlabel("Number of Special Requests")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.show()


# ============================================================
# 7. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

# Use a sample to reduce model training time.
data = df.sample(n=min(10000, len(df)), random_state=42)

# Select input features and target.
X = data.drop("booking_status", axis=1)
Y = data["booking_status"].map({
    "Not_Canceled": 0,
    "Canceled": 1
})

# Convert categorical input columns into numerical columns.
# One-hot encoding avoids assigning a false numeric order to categories.
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Handle any remaining missing values.
X = X.fillna(X.median())

# Split into training and testing data.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Standardize features for Logistic Regression, KNN and SVM.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 8. MODEL 1: LOGISTIC REGRESSION
# ============================================================

model1 = LogisticRegression(max_iter=1000, class_weight="balanced")
model1.fit(X_train, Y_train)
y_pred1 = model1.predict(X_test)

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(Y_test, y_pred1))
print("Precision:", precision_score(Y_test, y_pred1, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred1, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred1, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred1))
print(classification_report(Y_test, y_pred1, zero_division=0))


# ============================================================
# 9. MODEL 2: K-NEAREST NEIGHBORS
# ============================================================

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

print("\nK-NEAREST NEIGHBORS")
print("Accuracy:", accuracy_score(Y_test, y_pred2))
print("Precision:", precision_score(Y_test, y_pred2, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred2, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred2, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred2))
print(classification_report(Y_test, y_pred2, zero_division=0))


# ============================================================
# 10. MODEL 3: SUPPORT VECTOR MACHINE
# ============================================================

model3 = SVC(kernel="linear", cache_size=1000, class_weight="balanced")
model3.fit(X_train, Y_train)
y_pred3 = model3.predict(X_test)

print("\nSUPPORT VECTOR MACHINE")
print("Accuracy:", accuracy_score(Y_test, y_pred3))
print("Precision:", precision_score(Y_test, y_pred3, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred3, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred3, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred3))
print(classification_report(Y_test, y_pred3, zero_division=0))


# ============================================================
# 11. MODEL 4: DECISION TREE
# ============================================================

model4 = DecisionTreeClassifier(
    max_depth=15, random_state=42, class_weight="balanced"
)
model4.fit(X_train, Y_train)
y_pred4 = model4.predict(X_test)

print("\nDECISION TREE")
print("Accuracy:", accuracy_score(Y_test, y_pred4))
print("Precision:", precision_score(Y_test, y_pred4, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred4, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred4, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred4))
print(classification_report(Y_test, y_pred4, zero_division=0))


# ============================================================
# 12. MODEL 5: RANDOM FOREST
# ============================================================

model5 = RandomForestClassifier(
    n_estimators=100, max_depth=20,
    random_state=42, class_weight="balanced", n_jobs=1
)
model5.fit(X_train, Y_train)
y_pred5 = model5.predict(X_test)

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(Y_test, y_pred5))
print("Precision:", precision_score(Y_test, y_pred5, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred5, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred5, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred5))
print(classification_report(Y_test, y_pred5, zero_division=0))


# ============================================================
# 13. MODEL COMPARISON AND BEST-FIT MODEL
# ============================================================

models = [
    "Logistic Regression", "KNN", "SVM",
    "Decision Tree", "Random Forest"
]

predictions = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5]

accuracies = []
precisions = []
recalls = []
f1_scores = []

for prediction in predictions:
    accuracies.append(accuracy_score(Y_test, prediction))
    precisions.append(
        precision_score(Y_test, prediction, zero_division=0)
    )
    recalls.append(
        recall_score(Y_test, prediction, zero_division=0)
    )
    f1_scores.append(f1_score(Y_test, prediction, zero_division=0))

print("\nMODEL PERFORMANCE COMPARISON")
for i in range(len(models)):
    print("\n", models[i])
    print("Accuracy:", round(accuracies[i], 4))
    print("Precision:", round(precisions[i], 4))
    print("Recall:", round(recalls[i], 4))
    print("F1 Score:", round(f1_scores[i], 4))

# Compare model accuracy.
plt.figure(figsize=(10, 5))
sns.barplot(x=models, y=accuracies, color="steelblue")
plt.xticks(rotation=20)
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.show()

# Compare model F1 scores.
plt.figure(figsize=(10, 5))
sns.barplot(x=models, y=f1_scores, color="steelblue")
plt.xticks(rotation=20)
plt.xlabel("Machine Learning Models")
plt.ylabel("F1 Score")
plt.title("Model F1 Score Comparison")
plt.tight_layout()
plt.show()

# Select the model with the highest F1 score.
best_index = f1_scores.index(max(f1_scores))
best = models[best_index]

print("\nBEST-FIT MODEL:", best)
print("BEST F1 SCORE:", round(f1_scores[best_index], 4))
print("BEST MODEL ACCURACY:", round(accuracies[best_index], 4))


# ============================================================
# 14. KEY FINDINGS
# ============================================================

total_bookings = len(df)
canceled_bookings = (df["booking_status"] == "Canceled").sum()
not_canceled_bookings = (df["booking_status"] == "Not_Canceled").sum()

print("\nKEY FINDINGS")
print("1. Total bookings analysed:", total_bookings)
print("2. Canceled bookings:", canceled_bookings)
print("3. Non-canceled bookings:", not_canceled_bookings)
print(
    "4. Overall cancellation rate:",
    round(canceled_bookings / total_bookings * 100, 2), "%"
)
print("5. Market segment with highest cancellation rate:", segment_rate.idxmax())
print("6. Best-fit machine learning model:", best)


# ============================================================
# 15. FINAL CONCLUSION AND RECOMMENDATIONS
# ============================================================

print("\nFINAL CONCLUSION")
print(
    "This project analysed hotel booking data to understand booking "
    "cancellations and predict whether a booking will be canceled. "
    "Data cleaning, missing value handling, duplicate removal, feature "
    "engineering and exploratory data analysis were performed. Five "
    "objectives were investigated using data analysis and visualizations. "
    "Five machine learning models were evaluated, and the model with "
    "the highest F1 score was selected as the best-fit model."
)

print("\nRECOMMENDATIONS")
print(
    "1. Hotel management should monitor booking cancellation patterns.\n"
    "2. Market segments with higher cancellation rates should be reviewed.\n"
    "3. Lead time, room price and special requests can help understand "
    "booking behaviour.\n"
    "4. The selected model can support cancellation-risk analysis.\n"
    "5. The model should be evaluated regularly using newer booking data."
)
