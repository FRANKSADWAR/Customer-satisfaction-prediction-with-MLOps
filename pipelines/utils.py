import pandas as pd
import numpy as np
import logging

from src.data_cleaning import DataCleaning, DataPreProcessStrategy

def get_data_for_test():
    try:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        df = df.sample(n=100)

        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning  = DataCleaning(df,strategy=preprocess_strategy)
        df = data_cleaning.handle_data()
        
        df.drop(["review_score"],axis=1, inplace=True)
        result = df.to_json(orient="split") 
        return result
    except Exception as e:
        logging.error("Error in transforming the prediction data: {}".format(e))
        raise e

