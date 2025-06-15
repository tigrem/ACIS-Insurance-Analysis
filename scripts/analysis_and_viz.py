import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def inspect_data(df):
    """
    Prints the first few rows (head) and a concise summary (info) of the DataFrame.
    """
    print("\n--- Data Head ---")
    print(df.head())
    print("\n--- Data Info ---")
    df.info()

def get_descriptive_statistics(df):
    """
    Computes and prints descriptive statistics (e.g., count, mean, std, min, max)
    for numerical columns in the DataFrame.
    """
    print("\n--- Descriptive Statistics ---")
    print(df.describe())

def check_data_types(df):
    """
    Prints the data types of each column in the DataFrame, which is useful
    for identifying if columns are correctly parsed (e.g., numbers as numbers).
    """
    print("\n--- Data Types ---")
    print(df.dtypes)

def check_missing_values(df):
    """
    Checks for and prints the count of missing values (NaN) for each column in
    the DataFrame, displaying only columns that have at least one missing value.
    """
    print("\n--- Missing Values ---")
    missing_values = df.isnull().sum()
    # Display only columns with missing values (count > 0)
    print(missing_values[missing_values > 0])


def plot_numerical_histograms(df, numerical_cols):
    """
    Generates and displays histograms for a list of specified numerical columns.
    If a column's maximum value is very large, a logarithmic y-scale is applied
    to better visualize skewed distributions.
    """
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols):
        plt.subplot(2, 2, i + 1)
        # Plot histogram, dropping NaNs to avoid errors
        plt.hist(df[col].dropna(), bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        # Apply log scale if max value is large (e.g., > 1000) for better visualization
        if not df[col].dropna().empty and df[col].dropna().max() > 1000:
            plt.yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
    plt.tight_layout()
    plt.show()

def plot_categorical_bar_charts(df, categorical_cols):
    """
    Generates and displays bar charts for a list of specified categorical columns.
    Each bar chart shows the frequency of each category.
    """
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_cols):
        plt.subplot(2, 2, i + 1)
        # Calculate value counts for each category and plot as a bar chart
        df[col].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
        plt.title(f'Bar Chart of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_log_claims_boxplot(df, claims_col='TotalClaims'):
    """
    Creates and displays a box plot for the logarithmically transformed claims data.
    The log1p (log(1+x)) transformation is used to handle zero claims and reduce skewness,
    making the distribution more symmetric and easier to visualize outliers.
    """
    # Ensure the claims column is numeric, coercing errors to NaN.
    df[claims_col] = pd.to_numeric(df[claims_col], errors='coerce')
    df['Log' + claims_col] = np.log1p(df[claims_col].dropna())

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['Log' + claims_col].dropna(), color='yellow') # Box plot of the transformed data
    plt.title(f'Box Plot of Log Transformed {claims_col}')
    plt.xlabel(f'Log(1 + {claims_col})')
    plt.grid(True)
    plt.show()

def plot_total_claims_by_province(df, claims_col='TotalClaims', province_col='Province'):
    """
    Generates and displays a bar plot showing the sum of total claims for each province.
    Filters out rows with missing or zero claims.
    """
    df[claims_col] = pd.to_numeric(df[claims_col], errors='coerce')
    # Filter for non-null claims and province, and only claims greater than zero.
    data_filtered = df.dropna(subset=[claims_col, province_col])
    non_zero_claims_df = data_filtered[data_filtered[claims_col] > 0]

    if not non_zero_claims_df.empty:
        plt.figure(figsize=(12, 6))
        # Create a bar plot, summing claims for each province.
        sns.barplot(x=province_col, y=claims_col, data=non_zero_claims_df, estimator=sum, palette='viridis')
        plt.title(f'Total Claims by {province_col}')
        plt.xlabel(province_col)
        plt.ylabel(f'Total {claims_col}')
        plt.xticks(rotation=45) # Rotate x-axis labels for readability
        plt.grid(True)
        plt.show()
    else:
        print(f"No non-zero {claims_col} to plot for {province_col}.")

def plot_total_claims_by_vehicle_type(df, claims_col='TotalClaims', vehicle_col='VehicleType', premium_col='TotalPremium'):
    """
    Generates and displays a bar plot showing the sum of total claims for each vehicle type.
    Filters out rows with missing or zero claims/premiums.
  """
    # Filter for rows where claims and premium are not null and are positive.

    data_filtered = df[(df[claims_col].notnull()) &
                       (df[claims_col] > 0) &
                       (df[premium_col].notnull()) &
                       (df[premium_col] > 0)].copy()

    # Ensure claims column is numeric for aggregation.
    data_filtered[claims_col] = pd.to_numeric(data_filtered[claims_col], errors='coerce')

    if not data_filtered.empty:
        plt.figure(figsize=(12, 6))
        # Create a bar plot, summing claims for each vehicle type.
        sns.barplot(x=vehicle_col, y=claims_col, data=data_filtered, estimator=sum, palette='coolwarm')
        plt.title(f'Total Claims by {vehicle_col}')
        plt.xlabel(vehicle_col)
        plt.ylabel(f'Total {claims_col}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
    else:
        print("No valid data to plot for Total Claims by Vehicle Type.")

# --- Data Analysis Functions ---

def analyze_monthly_trends(df, premium_col='TotalPremium', claims_col='TotalClaims', month_col='TransactionMonth'):
    """
    Analyzes and visualizes monthly trends for total premium and total claims.
    Includes a scatter plot of monthly changes and a correlation heatmap between changes.
    """
    # Convert relevant columns to numeric and datetime types.
    df[premium_col] = pd.to_numeric(df[premium_col], errors='coerce')
    df[claims_col] = pd.to_numeric(df[claims_col], errors='coerce')
    df[month_col] = pd.to_datetime(df[month_col])

    # Aggregate total premium and claims by month.
    monthly_data = df.groupby(df[month_col].dt.to_period('M')).agg(
        TotalPremium=(premium_col, 'sum'),
        TotalClaims=(claims_col, 'sum')
    ).reset_index()

    # Calculate month-over-month changes.
    monthly_data['PremiumChange'] = monthly_data['TotalPremium'].diff()
    monthly_data['ClaimsChange'] = monthly_data['TotalClaims'].diff()

    # Drop the first row which will have NaN for changes. Use .copy() to avoid warnings.
    monthly_data = monthly_data.dropna().copy()

    if not monthly_data.empty:
        # Scatter plot of Monthly Changes in Premium vs. Claims.
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=monthly_data, x='PremiumChange', y='ClaimsChange', color='blue', alpha=0.7)
        plt.title('Scatter Plot of Monthly Changes: TotalPremium vs TotalClaims')
        plt.xlabel('Monthly Change in TotalPremium')
        plt.ylabel('Monthly Change in TotalClaims')
        plt.grid(True)
        plt.axhline(0, color='red', linestyle='--') # Horizontal line at y=0
        plt.axvline(0, color='red', linestyle='--') # Vertical line at x=0
        plt.show()

        # Calculate and plot the correlation matrix for monthly changes.
        correlation_matrix = monthly_data[['PremiumChange', 'ClaimsChange']].corr()
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Monthly Changes in TotalPremium and TotalClaims')
        plt.show()
    else:
        print("Not enough monthly data to analyze trends after calculating changes.")

def calculate_and_plot_loss_ratio(df, claims_col='TotalClaims', premium_col='TotalPremium',
                                  province_col='Province', vehicle_col='VehicleType'):
    """
    Calculates the loss ratio (Total Claims / Total Premium) and visualizes it
    by province and by vehicle type using bar plots. Filters for positive claims and premiums.
    """
    # Filter data for non-null and positive claims and premium.
    data_filtered = df[(df[claims_col].notnull()) & (df[premium_col].notnull()) &
                       (df[claims_col] > 0) & (df[premium_col] > 0)].copy()

    if not data_filtered.empty:
        data_filtered['LossRatio'] = data_filtered[claims_col] / data_filtered[premium_col]

        # Bar plot for Loss Ratio by Province.
        plt.figure(figsize=(12, 6))
        sns.barplot(x=province_col, y='LossRatio', data=data_filtered, palette='viridis')
        plt.title(f'Loss Ratio by {province_col}')
        plt.xlabel(province_col)
        plt.ylabel('Loss Ratio')
        plt.xticks(rotation=45)
        # Note: A single legend for a simple bar plot might be redundant.
        # It's included here to match the original code's intention.
        plt.legend(['Loss Ratio'])
        plt.grid(True)
        plt.show()

        # Bar plot for Loss Ratio by VehicleType.
        plt.figure(figsize=(12, 6))
        sns.barplot(x=vehicle_col, y='LossRatio', data=data_filtered, palette='plasma')
        plt.title(f'Loss Ratio by {vehicle_col}')
        plt.xlabel(vehicle_col)
        plt.ylabel('Loss Ratio')
        plt.xticks(rotation=45)
        plt.legend(['Loss Ratio'])
        plt.grid(True)
        plt.show()
    else:
        print("No valid data to calculate Loss Ratio after filtering.")

def plot_monthly_claims_trend(df, claims_col='TotalClaims', month_col='TransactionMonth'):
    """
    Plots the monthly claims trend over time as a line plot.
    """
    df[month_col] = pd.to_datetime(df[month_col])
    df[claims_col] = pd.to_numeric(df[claims_col], errors='coerce') # Ensure numeric type

    # Aggregate total claims by month.
    monthly_claims = df.groupby(df[month_col].dt.to_period('M')).agg(
        TotalClaims=(claims_col, 'sum')
    ).reset_index()

    if not monthly_claims.empty:
        # Convert PeriodIndex to Timestamp for accurate plotting on x-axis.
        monthly_claims['TransactionMonth'] = monthly_claims['TransactionMonth'].dt.to_timestamp()

        # Line plot for Monthly Claims Trend.
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=monthly_claims['TransactionMonth'], y=monthly_claims['TotalClaims'], marker='o', color='purple')
        plt.title('Monthly Claims Trend')
        plt.xlabel('Month')
        plt.ylabel('Total Claims')
        plt.xticks(rotation=45) # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.show()
    else:
        print("No monthly claims data to plot trend.")

# This block allows you to test the functions in this script independently.
if __name__ == '__main__':
    print("--- Running analysis_and_viz.py tests with dummy data ---")
    # Create a dummy DataFrame for testing all functions in this module.
    data_test = {
        'TotalPremium': np.random.rand(200) * 10000,
        'TotalClaims': np.random.rand(200) * 5000,
        'CustomValueEstimate': np.random.rand(200) * 20000,
        'Gender': np.random.choice(['Male', 'Female'], 200),
        'VehicleType': np.random.choice(['Car', 'Truck', 'Motorcycle'], 200),
        'Province': np.random.choice(['A', 'B', 'C', 'D'], 200),
        'TransactionMonth': pd.to_datetime(pd.date_range(start='2022-01-01', periods=200, freq='D')),
    }
    test_df = pd.DataFrame(data_test)
    # Introduce some NaN values and zeros for robust testing of filters.
    test_df.loc[[10, 20, 50, 55], ['TotalClaims', 'TotalPremium']] = np.nan
    test_df.loc[60, 'TotalClaims'] = 0
    test_df.loc[65, 'TotalPremium'] = 0

    # Run profiling functions
    inspect_data(test_df.copy())
    get_descriptive_statistics(test_df.copy())
    check_data_types(test_df.copy())
    check_missing_values(test_df.copy())

    numerical_cols_test = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
    categorical_cols_test = ['Gender', 'VehicleType', 'Province']

    # Run visualization functions
    plot_numerical_histograms(test_df.copy(), numerical_cols_test)
    plot_categorical_bar_charts(test_df.copy(), categorical_cols_test)
    plot_log_claims_boxplot(test_df.copy())
    plot_total_claims_by_province(test_df.copy())
    plot_total_claims_by_vehicle_type(test_df.copy())

    # Run analysis functions
    analyze_monthly_trends(test_df.copy())
    calculate_and_plot_loss_ratio(test_df.copy())
    plot_monthly_claims_trend(test_df.copy())

    print("--- analysis_and_viz.py tests complete ---")
