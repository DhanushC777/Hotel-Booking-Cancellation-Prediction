# 🏨 Hotel Booking Cancellation Prediction and Analysis

## 📌 About the Project

Hotel Booking Cancellation Prediction and Analysis is a Data Science and Machine Learning project developed using Python. The project focuses on analysing hotel reservation data, identifying patterns associated with booking cancellations, and predicting whether a hotel booking will be canceled or not.

By applying data preprocessing, Exploratory Data Analysis (EDA), data visualization, feature engineering, and Machine Learning classification algorithms, this project aims to discover meaningful insights into hotel booking behaviour and cancellation trends.

## 🎯 Project Objectives

The main objectives of this project are:

1. To analyse the distribution of canceled and non-canceled hotel bookings.
2. To investigate the relationship between lead time and booking cancellations.
3. To identify cancellation patterns across different market segments.
4. To examine the relationship between average booking prices and cancellation status.
5. To analyse how special requests influence hotel booking cancellations.

## 📂 Dataset Description

- **Dataset Name:** Hotel Booking Cancellation Dataset
- **File Name:** `hotel_booking_cancellation.csv`
- **Target Variable:** `booking_status`
- **Problem Type:** Binary Classification

The dataset contains hotel reservation details such as:

- Number of adults and children
- Number of weekend and weekday nights
- Car parking requirements
- Lead time
- Repeated customer information
- Market segment type
- Average booking price
- Special requests
- Date of reservation
- Booking status

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| Scikit-learn | Machine Learning algorithms |
| VS Code / Jupyter Notebook | Code development and execution |

## 🧹 Data Preprocessing

The following preprocessing techniques are performed:

- Standardized column names.
- Removed unnecessary identifier columns.
- Checked and removed duplicate records.
- Identified and handled missing values.
- Cleaned inconsistent categorical values.
- Converted numerical columns into appropriate data types.
- Converted reservation dates into datetime format.
- Handled missing numerical values using median.
- Handled missing categorical values using mode.
- Removed invalid booking status categories.

## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand the dataset and identify meaningful booking patterns.

The analysis includes:

- Booking status distribution.
- Lead time analysis.
- Market segment cancellation analysis.
- Average price analysis.
- Special requests analysis.
- Correlation analysis using a heatmap.

## 📊 Data Visualization

Different visualization techniques are used to represent the findings clearly:

- Booking Status Count Plot
- Lead Time Box Plot
- Cancellation Rate by Market Segment
- Average Price by Booking Status
- Cancellation Rate by Special Requests
- Correlation Heatmap
- Machine Learning Model Accuracy Comparison
- Model F1 Score Comparison

## ⚙️ Feature Engineering

New features are created to improve the Machine Learning process:

- **Reservation Year:** Extracted from the reservation date.
- **Reservation Month:** Extracted from the reservation date.
- **Reservation Day:** Extracted from the reservation date.
- **Total Nights:** Combination of weekend and weekday nights.
- **Total Guests:** Combination of adults and children.

The original reservation date is removed after extracting useful date-related features.

Categorical input features are converted into numerical representations using One-Hot Encoding.

The target variable is encoded as:

- Canceled = 1
- Not Canceled = 0

## 🤖 Machine Learning Models

Five classification algorithms are implemented and compared:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree Classifier
5. Random Forest Classifier

The dataset is sampled using up to 10,000 records to reduce model training time.

The dataset is divided into training and testing sets using an 80:20 ratio. Stratified splitting is applied to maintain the class distribution.

Feature scaling is performed using StandardScaler for Machine Learning model training.

## 📈 Model Evaluation

The performance of each Machine Learning model is evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix
- Classification Report

The performance of all five algorithms is compared, and the best-fit model is selected based on the highest F1 Score.

## 📁 Project Structure

```text
Hotel-Booking-Cancellation-Prediction/
│
├── hotel_booking_cancellation.csv
│
├── prog_file.py
│
└── README.md
```

## 🚀 How to Run the Project

**Step 1:** Clone or download this repository.

**Step 2:** Install the required Python libraries.

```bash
pip install pandas matplotlib seaborn scikit-learn
```

**Step 3:** Keep the dataset `hotel_booking_cancellation.csv` in the same folder as the Python script.

**Step 4:** Run the Python program.

```bash
python prog_file.py
```

**Step 5:** Execute the program and analyse the generated visualizations, model performance metrics, and key findings.

## 💡 Key Outcomes

- Understand hotel booking and cancellation patterns.
- Identify factors associated with booking cancellations.
- Analyse cancellation rates across different market segments.
- Discover useful insights through data analysis and visualization.
- Compare five Machine Learning classification algorithms.
- Identify the best-performing model based on F1 Score.

## 📝 Conclusion

This project demonstrates the application of Data Science and Machine Learning techniques to hotel booking data. By performing data preprocessing, Exploratory Data Analysis, feature engineering, visualization, and classification, the project aims to understand hotel booking cancellation behaviour and predict the cancellation status of reservations.

Five Machine Learning algorithms are evaluated using different performance metrics, and the model with the highest F1 Score is selected as the best-fit model.

The findings can help hotel management understand customer booking behaviour, identify cancellation patterns, and make informed decisions to manage booking cancellations.

## 🔮 Future Scope

- Implement advanced Machine Learning algorithms.
- Improve model performance through hyperparameter tuning.
- Develop an interactive dashboard using Power BI.
- Deploy the selected prediction model as a web application.
- Integrate additional booking datasets to improve prediction performance.

## 👨‍💻 Author

**Name:** Dhanush  
**Course:** Computer Science and Engineering  
**Project Domain:** Data Science and Machine Learning

---

⭐ If you find this project interesting, don't forget to star the repository!
