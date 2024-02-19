import logging
from zenml import pipeline, step
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated

@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"]
]:
    """
    Returns X_train, X_test and y_train and y_test
    Handles the call to cleaning ,transforming and splitting the data 
    """
    try:
        process_strategy = DataPreProcessStrategy() ## initialize the process strategy class here as it is the first step
        data_cleaning = DataCleaning(data, process_strategy)  ## call the data cleaning strategy with the data and the process
        processed_data = data_cleaning.handle_data() ## call methid to do the actual step

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data() ## data splitting step handles here
        logging.info("Data cleaning and processing completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error while processing data {}".format(e))
        raise e





