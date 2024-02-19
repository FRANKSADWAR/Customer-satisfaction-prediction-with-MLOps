import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train, y_train,**kwargs):
        """
        Trains the model based on X_train and y_train
        """
        print("----Training model-----")

### Linear regression is the baseline model
class LinearRegressionModel(Model):
    """
    Linear regression model
    """

    def train(self, X_train, y_train):
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error encountered in training the model: {}".format(e))
            raise e

        


