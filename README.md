# 🏨 Hotel Booking Cancellation Prediction and Analysis

## 📌 About the Project

Hotel Booking Cancellation Prediction and Analysis is a Data Science and Machine Learning project developed using Python. The project focuses on analysing hotel reservation data, identifying the major factors influencing booking cancellations, and predicting whether a hotel booking will be canceled or not.

By applying data preprocessing, Exploratory Data Analysis (EDA), data visualization, feature engineering, and Machine Learning classification algorithms, this project aims to discover meaningful patterns in hotel booking behaviour.

## 🎯 Project Objectives

The main objectives of this project are:

1. To analyse the distribution of canceled and non-canceled hotel bookings.
2. To investigate the relationship between lead time and booking cancellations.
3. To identify cancellation patterns across different market segments.
4. To examine the relationship between room types, average prices, and booking cancellations.
5. To analyse how special requests and repeat customers influence booking status.

## 📂 Dataset Description

- **Dataset Name:** Hotel Booking Cancellation Dataset
- **File Name:** `hotel_booking_cancellation.csv`
- **Total Records:** 36,285
- **Total Attributes:** 17
- **Target Variable:** `booking status`
- **Problem Type:** Binary Classification

The dataset contains hotel reservation details such as:

- 👥 Number of adults and children
- 🌙 Weekend and weekday nights
- 🍽️ Type of meal
- 🛏️ Room type
- ⏳ Lead time
- 🏢 Market segment type
- 💰 Average price
- ⭐ Special requests
- 📅 Date of reservation
- 📊 Booking status

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| VS Code / Jupyter Notebook | Code development and execution |

## 🧹 Data Preprocessing

The following preprocessing techniques are performed:

- Removed unnecessary columns.
- Checked and removed duplicate records.
- Identified and handled missing values.
- Cleaned inconsistent categorical values.
- Converted columns into appropriate data types.
- Handled invalid numerical values.
- Extracted useful information from reservation dates.

## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand the dataset and discover meaningful patterns.

The analysis includes:

- Booking status distribution.
- Lead time analysis.
- Average price distribution.
- Market segment analysis.
- Correlation and covariance analysis.
- Skewness and kurtosis analysis.
- Outlier detection using the IQR method.

## 📊 Data Visualization

Different visualization techniques are used to represent the findings clearly:

- Count Plots
- Histograms
- Box Plots
- Bar Charts
- Heatmaps
- Correlation Matrix

## ⚙️ Feature Engineering

New features are created to improve the Machine Learning process:

- **Total Nights:** Combination of weekend and weekday nights.
- **Total Guests:** Combination of adults and children.
- **Reservation Year:** Extracted from the reservation date.
- **Reservation Month:** Extracted from the reservation date.
- **Booking Status Encoding:**
  - Canceled = 1
  - Not Canceled = 0

Categorical features are transformed using One-Hot Encoding.

## 🤖 Machine Learning Models

Four classification algorithms are implemented and compared:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest Classifier

The dataset is divided into training and testing sets using an 80:20 ratio. Feature scaling is applied to the models where required.

## 📈 Model Evaluation

The performance of each model is evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix
- Classification Report

The models are compared to identify the best-performing algorithm based on the F1 score.

## 📁 Project Structure

```text
Hotel-Booking-Cancellation-Prediction/
│
├── hotel_booking_cancellation.csv
│
├── Hotel_Booking_Project_Reference_Style.ipynb
│
├── Hotel_Booking_Project_Reference_Style.py
│
├── hotel_booking_model_comparison.csv
│
└── README.md
```

## 🚀 How to Run the Project

**Step 1:** Clone or download this repository.

**Step 2:** Install the required Python libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Step 3:** Keep the dataset `hotel_booking_cancellation.csv` in the project folder.

**Step 4:** Run the Python script or open the notebook in your preferred Python environment.

**Step 5:** Execute the code and analyse the generated visualizations and Machine Learning results.

## 💡 Key Outcomes

- Understand hotel booking and cancellation patterns.
- Identify factors associated with booking cancellations.
- Discover useful insights through data analysis and visualization.
- Compare four Machine Learning classification algorithms.
- Determine the best-performing model using evaluation metrics.

## 📝 Conclusion

This project demonstrates the application of Data Science and Machine Learning techniques to hotel booking data. By performing data preprocessing, Exploratory Data Analysis, feature engineering, visualization, and classification, the project aims to understand booking cancellation behaviour and predict the cancellation status of hotel reservations.

The findings can help hotels better understand customer booking patterns and make informed decisions to manage cancellations.

## 🔮 Future Scope

- Implement advanced Machine Learning algorithms.
- Improve model performance through hyperparameter tuning.
- Develop an interactive dashboard using Power BI.
- Deploy the prediction model as a web application.
- Integrate real-time booking information for cancellation prediction.

## 👨‍💻 Author

**Name:** Dhanush  
**Course:** Computer Science and Engineering  
**Project Domain:** Data Science and Machine Learning

---

⭐ If you find this project interesting, don't forget to star the repository!
