import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  abc import ABC, abstractmethod
from typing import Union

class DataStrategy(ABC):
    """
    Abstract classs defining strategy for handling the data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for processing the data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(["order_approved_at","order_delivered_carrier_date",
                              "order_delivered_customer_date",
                              "order_estimated_delivery_date",
                              "order_purchase_timestamp"],
                             axis=1)
            
            ## impute missing values on these columns using the median value
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)

            ## where noe review was provided, fill it up using the no review string
            data["review_comment_message"].fillna("No review",inplace=True)

            ## select only the numerical columns
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix","order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data 
        
        except Exception as e:
            logging.error("Error in cleaning the data {}".format(e))
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Splitting the dataset into test and train sets
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        try:
            x =data.drop("review_score",axis=1)
            y =data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in processing the data: {}".format(e))
            raise e
        
class DataCleaning: ## maybe call this a different name such as DataTransformation
    """
    class that implements the data strategy
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        """
        Initializes the DataCleaning class with a specific strategy
        """
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Handle the data according to the strategy selected
        """    
        try: 
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling the data strategy selected: {}".format(e))
            raise e