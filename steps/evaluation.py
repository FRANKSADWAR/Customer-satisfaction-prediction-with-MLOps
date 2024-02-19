import logging
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from zenml import step
from src.evaluation import MSE, R2, RMSE
from sklearn.base  import RegressorMixin
from typing import Union, Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: Union[pd.DataFrame,pd.Series])-> Tuple[
    Annotated[float,"r2"],
    Annotated[float, "rmse"]
]:
    try:
        prediction = model.predict(X_test)
        mse_class = MSE() ## instantiate the class
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("MSE",mse)

        r2_class = R2() ## instantiate the class
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("R2",r2)

        rmse_class = RMSE() ## instantiate the class
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("RMSE",rmse)

        return r2, rmse
    
    except Exception as e:
        logging.error("Unable to compute the model metrics successfully {}".format(e))
        raise e    