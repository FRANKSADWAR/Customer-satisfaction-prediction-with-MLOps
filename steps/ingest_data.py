import logging
import pandas as pd
from zenml import step, pipeline

class IngestData:
    """
    Ingesting data from the data path
    """

    def __init__(self,data_path: str):
        self.data_path = data_path
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path: str) -> pd.DataFrame:  ## renamed from ingest_data to ingest_df
    """
    Ingesting data from the data_path

    Args:
        data_path: path to the dataset
    Returns:
        pd.DataFrame: the ingested data in a dataframe    
    """
    try:
        logging.info("------Reading the data from file")
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        logging.info("Finished reading the dat from file****")
        return df
    except Exception as e:
        logging.error(f"Eror while ingesting data {e}")   
        raise e