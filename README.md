# air-quality-ml-framework

A Comprehensive Machine Learning Framework for Predicting and Analyzing Urban Air Quality in India
Author: Nadia Sharief

Institution: University of Arizona

Degree: Master of Science in Data Science

Date of Completion: March 25, 2025

Introduction:
Air pollution is a major environmental health issue, with increasing evidence linking exposure to harmful pollutants such as nitrogen dioxide (NO₂) and particulate matter (PM) to severe health problems, including respiratory diseases, cardiovascular issues, and cancer. Urban areas, particularly those with high pollution levels, disproportionately affect vulnerable populations, exacerbating health disparities. Understanding how varying levels of air pollution influence public health is crucial, especially in communities facing environmental injustices.

This research aims to assess the impact of air pollution on health outcomes, with a particular focus on cancer incidence and survival. Machine learning (ML) techniques are employed to predict air quality levels, categorized into AQI (Air Quality Index) buckets, which represent different health risk categories. By classifying AQI levels into categories such as "Good," "Moderate," and "Unhealthy," the study investigates how exposure to different pollution levels correlates with health risks.

The study utilizes a range of machine learning models, including K-Nearest Neighbors (KNN), Gaussian Naïve Bayes (GNB), Support Vector Machine (SVM), Random Forest (RF), and XGBoost, to predict AQI buckets based on environmental data from 2015 to 2020. To address class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) is applied to ensure robust model performance across all AQI categories. Various performance metrics, including accuracy, precision, recall, F1-score, and error metrics such as RMSE and MAE, are used to evaluate model effectiveness.

By exploring the relationship between air pollution and health outcomes, this research aims to provide valuable insights for public health policy, particularly in regions where air pollution is a critical concern. The findings will assist policymakers and healthcare professionals in designing interventions to reduce exposure to harmful pollutants, especially in underserved communities.

Methodology Applied in this Project
Data Preparation and Preprocessing
Data Sources: Air quality data from CPCB and Kaggle (AQI, NO₂, PM for 2015-2020).
Data Imputation: Compared mean, KNN, and iterative imputation methods. Chose KNN for its lower RMSE and ability to maintain non-negative data.
Data Cleaning: Identified and corrected outliers and erroneous values affecting model performance.
Mapping to AQI Categories: Categorized AQI values into Good, Moderate, Unhealthy, etc..
Data Consistency: Verified data types and unique values to maintain consistency.
Exploratory Data Analysis (EDA)
Feature Exploration: Examined the influence of pollutants on AQI levels.
Correlation Heatmap: Visualized relationships between pollutants and AQI.
Trend Analysis: Analyzed annual and seasonal variations in AQI and pollutants, including trends for the six most polluted cities in India.
Pollutant Contributions: Investigated city-wise AQI contributions and pollutant trends over time.
Feature Transformation & Normalization:
Assessed and addressed skewness in features.
Applied log and Yeo-Johnson transformations for normalization.
Used StandardScaler to normalize training and testing datasets.
Class Imbalance Handling: Identified class imbalance, leading to the application of SMOTE.
Geospatial Analysis (Within EDA):
City-Level AQI Mapping:

Used geocoding to obtain latitude/longitude coordinates for cities.
Plotted AQI bucket distributions on Folium maps, highlighting pollution hotspots.
Spatial AQI Trends:

Visualized geographic variations in AQI levels.
Analyzed regional disparities in pollution exposure.
Mapped yearly and monhtly trends to observe AQI fluctuations.
Model Development
Class Distribution Analysis: Evaluated class distribution in training and testing sets for AQI prediction.
Model Definition: Defined ML models (KNN, Naive Bayes, Random Forest, XGBoost, SVM) for AQI prediction.
Model Evaluation:
Without SMOTE: Assessed model performance on normalized data.
With SMOTE: Applied SMOTE to address class imbalance, improving prediction for underrepresented AQI categories.
Choropleth Map of Deaths from Outdoor Air Pollution by Country (2015–2020)
Data Loading and Combining

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

