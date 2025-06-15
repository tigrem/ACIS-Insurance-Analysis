import os
import warnings
import sys
import pandas as pd

DATA_DIR = 'notebooks/data'

def setup_environment(base_path='.'):

    try:

        os.chdir(base_path)

        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())

        warnings.filterwarnings("ignore")

    except OSError as e:
        print(f"Error setting up environment: {e}. Please ensure the path '{base_path}' is correct.")


def load_data(file_name='MachineLearningRating_v3.txt', data_dir=DATA_DIR):

    file_path = os.path.join(data_dir, file_name)
    try:
        # Read the data using pandas, specifying the separator.
        data = pd.read_csv(file_path, sep='|')
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return pd.DataFrame() # Return an empty DataFrame on FileNotFoundError
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return pd.DataFrame()

# This block allows you to test the functions in this script independently.
if __name__ == '__main__':
    test_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    setup_environment(test_project_root)
    test_data_dir = os.path.join(test_project_root, 'notebooks', 'data')
    df = load_data(data_dir=test_data_dir)
    if not df.empty:
        print("\nData head (from data_handler.py test):")
        print(df.head())
