from abc import ABC, abstractmethod
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining the strategy for evaluation of the models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) :
        pass
    

class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Computing MSE")   
            mse = mean_squared_error(y_true, y_pred)
            logging.info("The mse for the model is {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in computing the MSE: {}".format(e))
            raise e
        
class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Computing the R2 score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in computing the R2 score {}".format(e))
            raise e    
        
class RMSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Computing the RMSE") 
            sq_mse = mean_squared_error(y_true,y_pred)
            rmse = np.sqrt(sq_mse)
            logging.info("RMSE for the model is :{}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in computing the RMSE of the model {}".format(e))
            raise e     