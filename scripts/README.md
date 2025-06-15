ACIS Insurance Data Analysis

This project provides a comprehensive exploratory data analysis (EDA) and visualization of an insurance claims and premium dataset. The aim is to uncover patterns, trends, and insights related to total premiums, total claims, and loss ratios, considering various demographic and vehicle-related factors, as well as monthly performance.
🌟 Features

    Data Loading & Inspection: Efficiently loads data from a pipe-separated file and provides initial insights into its structure and content.

    Descriptive Statistics: Generates summary statistics for numerical columns to understand data distribution.

    Missing Value Analysis: Identifies and reports missing values across the dataset.

    Distribution Visualization:

        Histograms: Visualizes the distribution of key numerical features like TotalPremium, TotalClaims, and CustomValueEstimate, with logarithmic scaling for skewed data.

        Bar Charts: Displays the frequency distribution of categorical variables such as Gender, VehicleType, and Province.

    Claims & Premium Analysis:

        Log Transformation of Claims: Applies a log transformation to TotalClaims to reduce skewness and visualize its distribution more clearly using a box plot.

        Total Claims by Province: Shows the sum of claims for each province.

        Total Claims by Vehicle Type: Illustrates the sum of claims for different vehicle types.

    Monthly Trend Analysis:

        Analyzes the monthly changes in TotalPremium and TotalClaims.

        Visualizes the relationship between these monthly changes using a scatter plot and a correlation heatmap.

        Plots the overall monthly trend of total claims over time.

    Loss Ratio Calculation & Visualization:

        Calculates the Loss Ratio (TotalClaims / TotalPremium).

        Visualizes the loss ratio by Province and VehicleType to identify areas of higher or lower profitability.

🚀 Getting Started

Follow these steps to set up and run the analysis on your local machine.
Prerequisites

Ensure you have Python 3.x installed.
Installation

    Clone the repository:

    git clone https://github.com/tigrm/ACIS-Insurance-Analysis.git
    cd ACIS-Insurance-Analysis

    Install dependencies:

    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    pip install pandas numpy matplotlib seaborn

Data Setup

    Create data directory: Inside the ACIS-Insurance-Analysis directory, create a folder structure notebooks/data.

    mkdir -p notebooks/data

    Place the dataset: Download or place your MachineLearningRating_v3.txt file into the ACIS-Insurance-Analysis/notebooks/data/ directory. The file should be pipe-separated (|).

    Your project structure should look similar to this:

    ACIS-Insurance-Analysis/
    ├── your_analysis_script.py # The Python script containing your code
    ├── notebooks/
    │   └── data/
    │       └── MachineLearningRating_v3.txt
    └── README.md

Running the Analysis

Execute your insurance_history_data_EDA.ipynb script from the root directory of the ACIS-Insurance-Analysis project:

python insurance_history_data_EDA.ipynb


The script will print various data insights to the console and display several plots.
📊 Visualizations

The project generates the following types of visualizations:

    Histograms: For numerical data distributions.

    Bar Charts: For categorical data frequencies and aggregated totals (e.g., Total Claims by Province/Vehicle Type).

    Scatter Plots: To analyze relationships between monthly changes in premiums and claims.

    Heatmaps: To visualize correlations.

    Box Plots: To inspect the distribution of log-transformed claims.

    Line Plots: To observe monthly trends in claims.
