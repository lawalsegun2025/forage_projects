# ------- BEFORE STARTING - SOME BASIC TIPS
# You can add a comment within a Python file by using a hashtag '#'
# Anything that comes after the hashtag on the same line, will be considered
# a comment and won't be executed as code by the Python interpreter.

# --- 1) IMPORTING PACKAGES
# The first thing you should always do in a Python file is to import any
# packages that you will need within the file. This should always go at the top
# of the file

# --- 2) DEFINE GLOBAL CONSTANTS
# Constants are variables that should remain the same througout the entire running
# of the module. You should define these after the imports at the top of the file.
# You should give global constants a name and ensure that they are in all upper
# case, such as: UPPER_CASE

# --- 3) ALGORITHM CODE
# Next, we should write our code that will be executed when a model needs to be 
# trained. There are many ways to structure this code and it is your choice 
# how you wish to do this. The code in the 'module_helper.py' file will break
# the code down into independent functions, which is 1 option. 
# Include your algorithm code in this section below:
 
# --- 4) MAIN FUNCTION
# Your algorithm code should contain modular code that can be run independently.
# You may want to include a final function that ties everything together, to allow
# the entire pipeline of loading the data and training the algorithm to be run all
# at once


# Import all required libraries

import pandas as pd
from sklearn.ensemble import RandimForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# Load data
def load_data(path: str = "/path/to/csv/"):
    """
    This function takes a path string to a CSV file and loads it into
    a pandas DataFrame.

    : param     path (optional): str, relative path of the CSV file

    : return    df: pd.DataFrame           
    """

    df = pd.read_cdv(f"{path}")
    df.drop(columns=["Unnamed: 0"], inplace=True, error='ignore')
    return df

# Create target varable and predictor variables
def create_target_and_predictors(
        data: pd.DataFrame = None,
        target: str = "estimated_stock_pct"
):
    """
    This function takes in a pandas DataFrame and splits the columns
    into a target column and a set of predictor variables, i.e X and y.
    These tow splits of the data will be used to train a supervided
    machine learning model

    :param      data: pd.DataFrame, dataframe containing data for the model

    :param      target: str (optional), target variable that you want to predict

    :return     X: pd.DataFrame
                y: pd.Series
    """

    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Targe: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

