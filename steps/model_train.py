import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from zenml.client import Client
import mlflow 

experiment_tracker = Client().active_stack.experiment_tracker ## use this for tracking experiments


@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame, 
                y_train: pd.DataFrame,
                config: ModelNameConfig
                ) -> RegressorMixin:
    """
    Trains the model on the ingested data

    Args:
        X_train, y_train: the training data
        X_test, y_test: the testing data (should never be seen by the model)
        config: model configurations i.e name and hyperparameter configurations
    
    Returns:
        The Regressor/trained model
    """

    model = None
    try:
        if config.model_name== "LinearRegression":
            #mlflow.autolog(log_models=True)

            mlflow.sklearn.autolog() 
            # the training code
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            logging.info("Finished training the model")
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error training the model {}".format(e))
        raise e    