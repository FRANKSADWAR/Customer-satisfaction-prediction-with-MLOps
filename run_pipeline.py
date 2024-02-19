from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
import mlflow
import subprocess
import logging

if __name__=="__main__":
    mlflow_uri = Client().active_stack.experiment_tracker.get_tracking_uri()
    train_pipeline(data_path="/home/billy/Documents/ML_Projects/customer_satisfaction/data/olist_customers_dataset.csv")
    command = 'mlflow ui --backend-store-uri "{}" '.format(mlflow_uri)
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("Error encountered in launching the MLFLOW UI backend {}".format(e))
        raise e
