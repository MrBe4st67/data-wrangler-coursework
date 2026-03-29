# Data Wrangler

This project is a Streamlit app for uploading, cleaning, transforming, visualizing, and exporting tabular data.

I made it for the coursework. The idea of the app is to give the user one place where they can upload a dataset, look through it, clean it step by step, create charts, and then export the final result together with a workflow report.

The app works with:
- CSV files
- Excel files (xlsx)
- JSON files

# What the app does

The app is split into 4 pages:

# 1. Upload & Overview
This page is used to upload the dataset and look at the first profile of the data.

It shows:
- number of rows
- number of columns
- column names
- data types
- first rows of the dataset
- numeric summary
- categorical summary
- missing values by column
- duplicate count
- outlier summary

It also has a Reset session button.

# 2. Cleaning & Preparation
This is the main page of the app.

It includes tools for:

# Missing values
- drop rows with missing values
- drop columns above a missing-value threshold
- fill missing values with:
  - mean
  - median
  - mode
  - most frequent value
  - constant value
  - forward fill
  - backward fill

# Duplicates
- detect full-row duplicates
- detect duplicates by selected key columns
- preview duplicate groups
- remove duplicates and keep first or last

# Data types and parsing
- convert columns to numeric
- convert columns to category
- convert columns to datetime
- clean dirty numeric strings like currency values or percentages

# Categorical tools
- trim whitespace
- convert to lower case
- convert to title case
- replace values using a mapping table
- group rare categories into 'Other'
- one-hot encode one column

# Numeric cleaning
- detect outliers with IQR
- detect outliers with z-score
- winsorize values at selected quantiles
- remove outlier rows

# Normalization and scaling
- min-max scaling
- z-score standardization

# Column operations
- rename columns
- drop columns
- create new columns with simple formulas
- bin numeric columns into categories

# Validation rules
- numeric range check
- allowed categories check
- non-null constraint
- export violations as CSV

# Transformation log
The app keeps a transformation log in session state.  
It also supports:
- undo last step
- reset all transformations back to the original uploaded file

# 3. Visualization Builder
This page is used to build charts from the cleaned dataset.

The user can apply:
- category filters
- numeric range filters

The app supports these chart types:
- histogram
- box plot
- scatter plot
- line chart
- bar chart
- heatmap / correlation matrix

It also supports:
- optional grouping
- aggregation
- top N categories for bar charts

All charts are built with matplotlib.

# 4. Export & Report
This page is used to export the final results.

The user can download:
- cleaned dataset as CSV
- cleaned dataset as Excel
- transformation report as JSON
- transformation report as CSV
- JSON recipe of the workflow
- replay snippet in Python format

# Main libraries used

This project mainly uses:
- Streamlit for the web app interface
- Panda for data handling
- NumPy for numerical operations
- Matplotlib for visualizations

# How the code is organized

The app is written in one main file: 'app.py'

The code is split into small functions, for example:
- state and reset helpers
- file loading and profiling helpers
- cleaning helpers
- validation helpers
- rendering functions for each page

This made it easier to keep the project more organized and easier to debug.

# How to run the app

1. Install the required libraries:

pip install -r requirements.txt

2. Run the app:

streamlit run app.py

3. Open the local Streamlit link in your browser.

# Data Sources

- De Cock, D. (2011). Ames, Iowa Housing Dataset. Used in Kaggle “House Prices: Advanced Regression Techniques” competition. Avaible from:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=train.csv

- IBM. (n.d.). Telco Customer Churn Dataset. Retrieved from GitHub dataset repository. Avaible from:
https://github.com/hemanth-HN/OR568/blob/master/train.csv

# Notes

- The app keeps a working copy of the dataset in session state.
- The transformation log is updated after each real change.
- Cached loading and profiling are used to avoid unnecessary reruns.
- The app was tested with mixed-type datasets and is designed for coursework-style data preparation workflows.

# Project purpose

The main purpose of this project is to show the full workflow of data preparation:

upload -> inspect -> clean -> transform -> visualize -> export