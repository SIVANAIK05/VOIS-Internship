# Crop Recommendation Analysis

## Project Overview

This project focuses on the exploratory analysis of a crop recommendation dataset using Python.

The analysis examines different agricultural and environmental parameters such as Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, soil pH, and rainfall. The project uses data visualization and statistical analysis to understand the characteristics and patterns present in the dataset.

---

## Objective

The main objectives of this project are:

- Explore the crop recommendation dataset.
- Understand the structure and characteristics of the data.
- Perform statistical analysis on the dataset.
- Check for missing values.
- Identify and analyze potential outliers.
- Visualize the distribution of agricultural parameters.
- Study relationships between environmental parameters.
- Compare agricultural parameters across different crop types.

---

## Dataset

The project uses the `Crop_recommendation.csv` dataset.

### Features

The dataset contains the following parameters:

- **N** – Nitrogen content
- **P** – Phosphorus content
- **K** – Potassium content
- **Temperature** – Temperature condition
- **Humidity** – Humidity condition
- **pH** – Soil pH value
- **Rainfall** – Rainfall condition
- **Label** – Recommended crop

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Analysis Performed

### 1. Data Exploration

The project performs:

- Dataset loading
- First and last records inspection
- Dataset shape analysis
- Dataset information
- Descriptive statistics
- Data type inspection

### 2. Missing Value Analysis

The dataset is checked for missing values using:

```python
df.isnull().sum()
