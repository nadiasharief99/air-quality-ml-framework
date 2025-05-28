# A Comprehensive Machine Learning Framework for Predicting and Analyzing Urban Air Quality in India

**Author:** Nadia Sharief  
**Degree:** Master of Science in Data Science  
**Date of Completion:** March 25, 2025

---

## 📘 Introduction

Air pollution is a major environmental health issue, with increasing evidence linking exposure to harmful pollutants such as nitrogen dioxide (NO₂) and particulate matter (PM) to severe health conditions like respiratory diseases, cardiovascular issues, and cancer. Urban regions with higher pollution levels tend to impact vulnerable populations more severely, compounding health disparities.

This research assesses the impact of air pollution on public health outcomes—particularly cancer incidence and survival—by predicting air quality levels using machine learning (ML). AQI (Air Quality Index) values are categorized into buckets such as "Good," "Moderate," and "Unhealthy" to explore their correlation with health risks.

Multiple ML models are used for AQI prediction, including:
- K-Nearest Neighbors (KNN)
- Gaussian Naïve Bayes (GNB)
- Support Vector Machine (SVM)
- Random Forest (RF)
- XGBoost

To handle class imbalance in AQI categories, SMOTE (Synthetic Minority Over-sampling Technique) is applied. Evaluation metrics include accuracy, precision, recall, F1-score, RMSE, and MAE.

The study’s findings aim to support data-driven public health policies and inform interventions that reduce exposure to harmful pollutants—especially in underserved communities.

---

## 🔧 Methodology Applied

### 🧹 Data Preparation and Preprocessing
- **Data Sources**: CPCB & Kaggle (AQI, NO₂, PM for 2015–2020)
- **Imputation Techniques**: Compared mean, KNN, and iterative imputation; KNN was chosen for its lower RMSE and better data integrity.
- **Data Cleaning**: Removed outliers and corrected erroneous values.
- **AQI Categorization**: Mapped AQI values to health-related buckets (Good, Moderate, Unhealthy, etc.).
- **Consistency Checks**: Verified data types and unique values for clean processing.

### 📊 Exploratory Data Analysis (EDA)
- **Feature Insights**: Assessed how pollutants affect AQI levels.
- **Correlation Analysis**: Used heatmaps to visualize relationships.
- **Temporal Trends**: Analyzed annual and seasonal patterns in AQI/pollutants.
- **City-Level Analysis**: Focused on six most polluted Indian cities.

### 🔁 Feature Transformation & Normalization
- Addressed skewness using log and Yeo-Johnson transformations.
- Standardized features using `StandardScaler`.

### ⚖️ Handling Class Imbalance
- Detected imbalanced AQI buckets.
- Applied **SMOTE** to balance class distribution.

---

## 🌍 Geospatial Analysis (within EDA)

### 🗺 City-Level AQI Mapping
- Obtained geolocation coordinates via geocoding.
- Used **Folium** to create interactive AQI heatmaps.

### 📌 Spatial AQI Trends
- Visualized regional disparities in pollution.
- Tracked AQI fluctuations over time using yearly and monthly maps.

---

## 🤖 Model Development & Evaluation

### ⚙️ Model Setup
- Defined classification models: KNN, Naive Bayes, SVM, Random Forest, XGBoost.
- Evaluated class distributions in train/test datasets.

### 📈 Model Evaluation
- **Without SMOTE**: Trained models on normalized datasets.
- **With SMOTE**: Improved predictions for minority AQI classes.

---

## 🗺 Choropleth Map: Air Pollution Deaths (2015–2020)
- Created a global choropleth map showing country-level mortality attributed to outdoor air pollution.
  
### Data Loading and Combining
```
import pandas as pd

data1 = pd.read_excel("C:\\Users\\nadia\\Downloads\\data1.xlsx")
data2 = pd.read_excel("C:\\Users\\nadia\\Downloads\\data2.xlsx")
data3 = pd.read_excel("C:\\Users\\nadia\\Downloads\\data3.xlsx")
data4 = pd.read_excel("C:\\Users\\nadia\\Downloads\\data4.xlsx")
data5 = pd.read_excel("C:\\Users\\nadia\\Downloads\\data5.xlsx")

# Combine all datasets 
df_all = pd.concat([data1, data2, data3, data4, data5], ignore_index=True)
print("Combined dataset shape:", df_all.shape)

print("First few rows of the combined dataset:")
print(df_all.head())

```
```
Combined dataset shape: (915, 4)
First few rows of the combined dataset:
  GeometryCode  Max of Year        First Location           First Tooltip
0          AFG         2015           Afghanistan  20 688 [16 534-25 300]
1          AGO         2015                Angola      8527 [2150-16 192]
2          ALB         2015               Albania        2250 [1643-2862]
3          ARE         2015  United Arab Emirates        1607 [1251-1962]
4          ARG         2015             Argentina  18 729 [12 845-25 270]
```
### Data Cleaning and Aggregation

```
import re

# Defining a function to extract the death count 
def extract_deaths(text):
    """
    Extracts the first numeric value from the input text.
    For example, "20 688 [16 534-25 300]" becomes 20688.
    """
    # Remove spaces and then use regex to extract digits.
    text_clean = text.replace(" ", "")
    match = re.search(r'(\d+)', text_clean)
    if match:
        return int(match.group(1))
    return None

# Apply the function to create a new deaths column
df_all['Deaths'] = df_all['First Tooltip'].astype(str).apply(extract_deaths)

# Rename "First Location" to "Country" for clarity
df_all.rename(columns={'First Location': 'Country'}, inplace=True)

# Preview the data with the new 'Deaths' column
print("Data with Deaths column:")
print(df_all[['Country', 'Max of Year', 'First Tooltip', 'Deaths']].head())

# Aggregate the total deaths by country (if multiple rows exist per country)
country_deaths = df_all.groupby("Country", as_index=False)["Deaths"].sum()

print("Aggregated deaths by country:")
print(country_deaths.head())
```
```
Data with Deaths column:
                Country  Max of Year           First Tooltip  Deaths
0           Afghanistan         2015  20 688 [16 534-25 300]   20688
1                Angola         2015      8527 [2150-16 192]    8527
2               Albania         2015        2250 [1643-2862]    2250
3  United Arab Emirates         2015        1607 [1251-1962]    1607
4             Argentina         2015  18 729 [12 845-25 270]   18729
Aggregated deaths by country:
               Country  Deaths
0          Afghanistan  106671
1              Albania   13457
2              Algeria   81529
3               Angola   42985
4  Antigua and Barbuda     106
```
