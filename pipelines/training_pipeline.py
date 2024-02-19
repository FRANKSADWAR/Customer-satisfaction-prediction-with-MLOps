from zenml import pipeline
import logging
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.model_train import train_model



@pipeline
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, y_train)
    r2, rmse = evaluate_model(model, X_test, y_test)