import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.logger import logger  

def load_data():
    try:
        logger.info("Starting to load dataset...")
        df = pd.read_csv('notebooks/stud.csv')
        logger.info(f"Dataset loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error("Error while loading dataset", exc_info=True)
        raise e